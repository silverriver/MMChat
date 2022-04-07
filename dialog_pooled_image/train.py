import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file', default='weibo_chat/dialog_pooled_image/config.json')
parser.add_argument('--gpu', help='which gpu to use', type=str, default='0')
parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if True:
    import torch
    import random
    import traceback
    import model.utils as utils
    import model.dataset as dataset
    from model.model_multi_input import MultiInputModel
    from model.trainer_multi_input import Trainer
    from model.text import Vocab
    from torch.nn.parallel import DistributedDataParallel
    from parlai.core.image_featurizers import ImageLoader

config = utils.load_config(args.config)
config_path = os.path.dirname(args.config)
logger = utils.get_logger(os.path.join(config_path, 'main.log'))

train_dir = os.path.join(config_path, config['train_dir'])
eval_dir = os.path.join(config_path, config['eval_dir'])
log_dir = os.path.join(config_path, config['log_dir'])
best_model = os.path.join(config_path, config['best_dir'])


# helpers -----------------------------------------------------
def save_func(epoch, device):
    filename = utils.get_ckpt_filename('model', epoch)
    torch.save(trainer.state_dict(), os.path.join(train_dir, filename))
    if os.path.exists(os.path.join(train_dir, utils.get_ckpt_filename('model', epoch-80))):
        os.remove(os.path.join(train_dir, utils.get_ckpt_filename('model', epoch-80)))


def sample_text_func(epoch, device):
    n_samples = 8
    samples_idxs = random.sample(range(len(valid_dataset)), n_samples)
    samples = [valid_dataset[idx] for idx in samples_idxs]
    for i, data in enumerate(samples):
        contexts = [torch.tensor([data['post']], dtype=torch.long, device=device)]

        prediction = trainer.model.predict(contexts)[0]
        post_str = vocab.ids2string(data['post'][1:-1])
        resp_str = vocab.ids2string(data['resp'][1:-1])
        pred_str = vocab.ids2string(prediction)

        logger.info('-------epoch {} sample {}---------'.format(epoch, i))
        logger.info('post: {}'.format(post_str))
        logger.info('resp: {}'.format(resp_str))
        logger.info('pred: {}'.format(pred_str))
# helpers -----------------------------------------------------


try:
    if args.local_rank == -1 or args.local_rank == 0:
        logger.info('pytorch version: {}'.format(torch.__version__))
        for i in config:
            logger.info('{}: {}'.format(i, config[i]))
        for i in vars(args):
            logger.info('{}: {}'.format(i, getattr(args, i)))

        dirs = [train_dir, eval_dir, log_dir, best_model]

        for d in dirs:
            if not os.path.isdir(d):
                logger.info('cannot find {}, mkdiring'.format(d))
                os.makedirs(d)

    # code for distributed training
    distributed = (args.local_rank != -1)
    if distributed:
        print(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.manual_seed(config.seed)
    else:
        device = torch.device("cuda", 0)

    logger.info('loading dataset on rank {}'.format(args.local_rank))
    vocab = Vocab(config.vocab_path)
    train_dataset = dataset.DialogDataset([config.train_dialog_data], [config.train_img_index_data], config.img_feat_dir, vocab, logger,
                                          config.max_context_len, config.max_resp_len, feat_num=config.img_feat_num)
    valid_dataset = dataset.DialogDataset([config.valid_dialog_data], [config.valid_img_index_data], config.img_feat_dir, vocab, logger,
                                          config.max_context_len, config.max_resp_len, feat_num=config.img_feat_num)

    logger.info('Building models')
    model = MultiInputModel(config, vocab)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)

    latest_ckpt = utils.get_latest_ckpt(train_dir)
    if latest_ckpt is None:  # start from scratch
        logger.info('start from CGPT weights')
        cgpt_model = torch.load(config.cgpt_parameters_dir, map_location=device)
        cgpt_model.pop('decoder.pre_softmax.weight')
        b = list(cgpt_model.keys())
        for i in b:
            cgpt_model[i.split('.', 1)[1]] = cgpt_model.pop(i)
        cgpt_model['embeddings.weight'] = torch.cat(
            [cgpt_model['embeddings.weight'], (torch.randn((len(vocab.spec_tokens) - len(vocab.pretrained_spec_tokens), config.embeddings_size), dtype=torch.float32) * 0.01).to(device, non_blocking=True)], 
            dim=0)
        cgpt_model['img_feat_proj.weight'] = model.transformer_module.img_feat_proj.weight.to(device, non_blocking=True)
        cgpt_model['img_feat_proj.bias'] = model.transformer_module.img_feat_proj.bias.to(device, non_blocking=True)

        model.transformer_module.load_state_dict(cgpt_model, strict=True)
        logger.info('CGPT weights loaded from {}'.format(config.cgpt_parameters_dir))

    trainer = Trainer(model, train_dataset, valid_dataset, config, log_dir, logger, device, vocab.special_tokens_ids,
                      distributed=distributed)

    if distributed:
        trainer.model.transformer_module = DistributedDataParallel(
            trainer.model.transformer_module, device_ids=[args.local_rank], output_device=args.local_rank)

    start_epoch = 0
    if latest_ckpt is not None:
        logger.info('Weights loading from {}'.format(os.path.join(train_dir, latest_ckpt)))
        start_epoch = utils.get_epoch_from_ckpt(latest_ckpt)
        trainer.load_state_dict(torch.load(os.path.join(train_dir, latest_ckpt), map_location=device))

    try:
        if args.local_rank in [-1, 0]:
            trainer.train(start_epoch, config.n_epochs, after_epoch_funcs=[save_func])
        else:
            trainer.train(start_epoch, config.n_epochs)
        # model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[sample_text_func], risk_func=f1_risk)
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(trainer.state_dict(), os.path.join(train_dir, 'interrupt.pt'))
        raise e
except BaseException:
    logger.error(traceback.format_exc())
