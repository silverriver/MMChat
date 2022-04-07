import os
import argparse
import json

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file',    default='dialog_model/infer_config.json')
parser.add_argument('--test_data', help='config file', default='data/dialog_test.json')
parser.add_argument('--out_file', help='out_file', default='infer_out.txt')
parser.add_argument('--infer_ckpt', help='out_file', default='model-80.ckpt')
parser.add_argument('--infer_bs', help='out_file', type=int, default=200)
parser.add_argument('--gpu', help='which gpu to use', type=str, default='6')
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
    from torch.utils.data import DataLoader
    from model.text import Vocab
    from tqdm import tqdm

config = utils.load_config(args.config)
config_path = os.path.dirname(args.config)
logger = utils.get_logger(os.path.join(config_path, 'infer.log'))

train_dir = os.path.join(config_path, config['train_dir'])
eval_dir = os.path.join(config_path, config['eval_dir'])

try:
    if args.local_rank == -1 or args.local_rank == 0:
        logger.info('pytorch version: {}'.format(torch.__version__))
        for i in config:
            logger.info('{}: {}'.format(i, config[i]))
        for i in vars(args):
            logger.info('{}: {}'.format(i, getattr(args, i)))

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

    vocab = Vocab(config.vocab_path)
    test_dataset = dataset.DialogDataset(
        [args.test_data], vocab, logger, config.max_context_len, config.max_resp_len)
    sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if distributed else None

    test_dataloader = DataLoader(test_dataset, sampler=sampler, pin_memory=True,
                                 batch_size=args.infer_bs, collate_fn=dataset.PadBatchSeq(vocab.pad_token_id))

    logger.info('Building models')
    model = MultiInputModel(config, vocab).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    latest_ckpt = args.infer_ckpt
    logger.info('Weights loading from {}'.format(os.path.join(train_dir, latest_ckpt)))
    weights = torch.load(os.path.join(train_dir, latest_ckpt), map_location=device)
    if 'model' in weights:
        weights = weights['model']

    weight_keys = list(weights.keys())
    for key in weight_keys:
        if key.startswith('transformer_module.module'):
            weights['transformer_module' + key[len('transformer_module.module'):]] = weights[key]
            weights.pop(key)

    model.load_state_dict(weights, strict=True)

    with torch.no_grad():
        model.eval()
        res = []
        with open(os.path.join(eval_dir, latest_ckpt + os.path.basename(args.out_file) + str(args.local_rank)), 'w', encoding='utf-8') as f:
            if args.local_rank == -1 or args.local_rank == 0:
                ITER = tqdm(test_dataloader, dynamic_ncols=True, total=len(test_dataloader))
            else:
                ITER = test_dataloader

            for data in ITER:
                bs = data['resp'].shape[0]
                context, resp = data['context'].to(device, non_blocking=True), data['resp'].to(device, non_blocking=True)
                context_seg_id = data['context_seg_id'].to(device, non_blocking=True)

                prediction = model.predict_beam([(context, context_seg_id)])
                for i in range(bs):
                    post_str = data['context'][i].tolist()
                    post_str = vocab.ids2string_wo_eos(post_str)
                    resp_str = data['resp'][i].tolist()[1:]
                    resp_str = vocab.ids2string_wo_eos(resp_str)
                    pred_strs = []
                    for j in prediction[i]:
                        pred_strs.append(vocab.ids2string_wo_eos(j))
                    tmp = dict()
                    tmp['context'] = post_str
                    tmp['response'] = resp_str
                    tmp['preds'] = pred_strs
                    tmp['dialog_index'] = data['dialog_index'][i].item()
                    tmp['utt_index'] = data['utt_index'][i].item()
                    print(json.dumps(tmp, ensure_ascii=False), file=f)
except BaseException:
    logger.error(traceback.format_exc())
