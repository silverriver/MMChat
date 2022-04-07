#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import os
import random
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torch.tensor
from .dataset import PadBatchSeq
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .optim import Adam, NoamOpt
from .loss import LabelSmoothingLoss
from metrics.eval_distinct import eval_distinct
from metrics.eval_bleu import eval_bleu
from metrics.eval_f1 import eval_f1


class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, config, log_dir, logger, device=torch.device('cuda'),
                 ignore_idxs=[], distributed=False):
        self.config = config
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.valid_dataset = valid_dataset
        self.rank = torch.distributed.get_rank() if distributed else -1
        if self.rank in [-1, 0]:
            self.train_writer = SummaryWriter(os.path.join(log_dir, 'train'), flush_secs=60)
            self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid'))
        self.ignore_idxs = ignore_idxs
        self.model = model.to(device, non_blocking=True)
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=self.model.vocab.pad_token_id).to(device, non_blocking=True)
        self.ppl_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=self.model.vocab.pad_token_id).to(device, non_blocking=True)
        self.criterion = LabelSmoothingLoss(n_labels=len(self.model.vocab), smoothing=config.label_smoothing,
                                            ignore_index=self.model.vocab.pad_token_id).to(device, non_blocking=True)
        base_optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.config.embeddings_size, config.lr, config.lr_warmup, base_optimizer)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else torch.utils.data.RandomSampler(train_dataset)
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if distributed else None

        self.train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=self.train_sampler,
                                           num_workers=config.n_jobs, pin_memory=True,
                                           collate_fn=PadBatchSeq(self.model.vocab.pad_token_id))
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=config.eval_batch_size, sampler=self.valid_sampler,
                                           num_workers=config.n_jobs, pin_memory=True,
                                           collate_fn=PadBatchSeq(self.model.vocab.pad_token_id))

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=True)
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _eval_train(self, epoch):
        self.model.train()
        step_count, lm_loss, s2s_loss = 0, 0, 0
        valid_lm_loss, valid_s2s_loss = -1, -1
        total = len(self.train_dataloader)
        if self.rank == -1 or self.rank == 0:
            ITER = tqdm(enumerate(self.train_dataloader), dynamic_ncols=True, total=total)
        else:
            ITER = enumerate(self.train_dataloader)

        for i, data in ITER:
            context = data['context'].to(self.device, non_blocking=True)
            context_seg_id = data['context_seg_id'].to(self.device, non_blocking=True)
            resp = data['resp'].to(self.device, non_blocking=True)
            img_feat = data['img_feat'].to(self.device, non_blocking=True)
            img_feat = torch.max(img_feat, dim=1, keepdim=True)[0]

            enc_contexts = []

            # lm loss
            post_rep = self.model.encode(context.clone(), context_seg_id)

            post_outputs = self.model.generate(post_rep[0])
            ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1).bool()
            context.masked_fill_(ignore_mask, self.model.vocab.pad_token_id)
            prevs, nexts = post_outputs[:, :-1, :].contiguous(), context[:, 1:].contiguous()
            batch_lm_loss = self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1))

            # s2s loss
            if hasattr(self.model.transformer_module, 'img_feat_proj'):
                enc_contexts.append((post_rep[0] + self.model.transformer_module.img_feat_proj(img_feat) * self.config.img_feat_weight, post_rep[1]))
            else:
                enc_contexts.append((post_rep[0] + self.model.transformer_module.module.img_feat_proj(img_feat) * self.config.img_feat_weight, post_rep[1]))

            prevs, nexts = resp[:, :-1].contiguous(), resp[:, 1:].contiguous()
            outputs = self.model.decode(prevs, enc_contexts)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_s2s_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            # optimization
            full_loss = (batch_lm_loss * self.config.lm_weight + batch_s2s_loss) / self.config.batch_split
            full_loss.backward()

            lm_loss += batch_lm_loss.item()
            s2s_loss += batch_s2s_loss.item()
            step_count += 1

            if (i + 1) % self.config.batch_split == 0:
                if self.config.clip_grad is not None:
                    for group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(group['params'], self.config.clip_grad)
                # update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                # shit log if you are node 0 in every step
                if self.rank == -1 or self.rank == 0:
                    lm_loss /= step_count
                    s2s_loss /= step_count
                    ITER.set_postfix_str(("lm_loss(t/v) {:>4.4f}/{:>4.4f}, s2s_loss {:>4.4f}/{:>4.4f}, " +
                        "total_loss {:>4.4f}/{:>4.4f}, step {}").format(lm_loss, valid_lm_loss, s2s_loss, valid_s2s_loss, lm_loss + s2s_loss,
                                                                        valid_s2s_loss + valid_lm_loss, self.optimizer.curr_step()))
                    self.train_writer.add_scalar('loss/lm_loss', lm_loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('loss/s2s_loss', s2s_loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('loss/total_loss', lm_loss + s2s_loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('lr/lr', self.optimizer.rate(), self.optimizer.curr_step())

                # only valid on dev and sample on dev data at every eval_steps
                if self.optimizer.curr_step() % self.config.eval_steps == 0:
                    valid_lm_loss, valid_s2s_loss, bleu, dist, f1, s2s_nll_loss, total_tokens = self._eval_test()
                    if self.rank != -1:
                        torch.distributed.all_reduce(valid_lm_loss, op=torch.distributed.reduce_op.SUM)
                        torch.distributed.all_reduce(valid_s2s_loss, op=torch.distributed.reduce_op.SUM)
                        torch.distributed.all_reduce(bleu, op=torch.distributed.reduce_op.SUM)
                        torch.distributed.all_reduce(dist, op=torch.distributed.reduce_op.SUM)
                        torch.distributed.all_reduce(f1, op=torch.distributed.reduce_op.SUM)
                        torch.distributed.all_reduce(s2s_nll_loss, op=torch.distributed.reduce_op.SUM)
                        torch.distributed.all_reduce(total_tokens, op=torch.distributed.reduce_op.SUM)
                        valid_lm_loss /= torch.distributed.get_world_size()
                        valid_s2s_loss /= torch.distributed.get_world_size()
                        bleu /= torch.distributed.get_world_size()
                        dist /= torch.distributed.get_world_size()
                        f1 /= torch.distributed.get_world_size()

                    # but only shit log if you are node 0
                    if self.rank == -1 or self.rank == 0:
                        valid_lm_loss = valid_lm_loss.item()
                        valid_s2s_loss = valid_s2s_loss.item()
                        self.valid_writer.add_scalar('loss/lm_loss', valid_lm_loss, self.optimizer.curr_step())
                        self.valid_writer.add_scalar('loss/s2s_loss', valid_s2s_loss, self.optimizer.curr_step())
                        self.valid_writer.add_scalar(
                            'loss/total_loss', valid_s2s_loss + valid_lm_loss, self.optimizer.curr_step())
                        self.valid_writer.add_scalar('metric/bleu', bleu, self.optimizer.curr_step())
                        self.valid_writer.add_scalar('metric/dist', dist, self.optimizer.curr_step())
                        self.valid_writer.add_scalar('metric/f1', f1, self.optimizer.curr_step())
                        self.valid_writer.add_scalar('metric/nll', s2s_nll_loss / total_tokens, self.optimizer.curr_step())
                        self.valid_writer.add_scalar('metric/ppl', torch.exp(s2s_nll_loss / total_tokens), self.optimizer.curr_step())

                        log_str = ('epoch {:>3}, t_lm_loss {:>4.4f}, t_s2s_loss {:>4.4f}, ' +
                                   'v_lm_loss {:>4.4f}, v_s2s_loss {:>4.4f}, bleu {:>4.4f}, dist {:>4.4f}, f1 {:>4.4f}, lr {:>.6}, step {}').format(
                            epoch, lm_loss, s2s_loss, valid_lm_loss, valid_s2s_loss, bleu, dist, f1, self.optimizer.rate(),
                            self.optimizer.curr_step())
                        self.logger.info(log_str)

                        # and only predicts sample on node 0
                        sample_dialog = self._pred_sample(5)
                        for j, d in enumerate(sample_dialog):
                            self.logger.info('--epoch {} step{} sample {}--'.format(
                                epoch, self.optimizer.curr_step(), j))
                            self.logger.info('context: {}'.format(d['context']))
                            self.logger.info('resp: {}'.format(d['resp']))
                            self.logger.info('pred: {}'.format(d['pred']))
                            self.train_writer.add_text('dialog', 'Context: {}\n  Resp: {}\n  Pred: {}\n'.format(
                                d['context'], d['resp'], d['pred']), self.optimizer.curr_step())
                    self.model.train()
                lm_loss, s2s_loss, step_count = 0, 0, 0

    def _eval_test(self):
        s2s_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        lm_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        bleu = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        f1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        dist = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        s2s_nll_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        total_tokens = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        all_refs_ids = []
        all_pred_ids = []
        with torch.no_grad():
            self.model.eval()

            if self.rank == -1 or self.rank == 0:
                ITER = tqdm(enumerate(self.valid_dataloader), dynamic_ncols=True, total=len(self.valid_dataloader))
            else:
                ITER = enumerate(self.valid_dataloader)
            for i, data in ITER:
                bs = data['resp'].shape[0]
                context = data['context'].to(self.device, non_blocking=True)
                context_seg_id = data['context_seg_id'].to(self.device, non_blocking=True)
                resp = data['resp'].to(self.device, non_blocking=True)
                img_feat = data['img_feat'].to(self.device, non_blocking=True)
                img_feat = torch.max(img_feat, dim=1, keepdim=True)[0]
                enc_contexts = []

                # lm loss
                post_rep = self.model.encode(context.clone(), context_seg_id)

                context_outputs = self.model.generate(post_rep[0])
                ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1).bool()
                context.masked_fill_(ignore_mask, self.model.vocab.pad_token_id)
                prevs, nexts = context_outputs[:, :-1, :].contiguous(), context[:, 1:].contiguous()
                batch_lm_loss = self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1))

                # s2s loss

                if hasattr(self.model.transformer_module, 'img_feat_proj'):
                    enc_contexts.append((post_rep[0] + self.model.transformer_module.img_feat_proj(img_feat) * self.config.img_feat_weight, post_rep[1]))
                else:
                    enc_contexts.append((post_rep[0] + self.model.transformer_module.module.img_feat_proj(img_feat) * self.config.img_feat_weight, post_rep[1]))
                prevs, nexts = resp[:, :-1].contiguous(), resp[:, 1:].contiguous()
                outputs = self.model.decode(prevs, enc_contexts)
                batch_s2s_nll_loss = self.ppl_criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
                pad_mask = ~nexts.view(-1).eq(self.model.vocab.pad_token_id).bool()
                s2s_nll_loss = s2s_nll_loss + torch.sum(batch_s2s_nll_loss * pad_mask)
                total_tokens = total_tokens + torch.sum(pad_mask)

                outputs = F.log_softmax(outputs, dim=-1)
                batch_s2s_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

                lm_loss = (i * lm_loss + batch_lm_loss) / (i + 1)
                s2s_loss = (i * s2s_loss + batch_s2s_loss) / (i + 1)

                # blue, dist, f1
                preds, lens = self.model.top_k_top_p_search(enc_contexts, top_p=self.config.top_p, top_k=self.config.top_k, sample_size=1)
                for j in range(bs):
                    all_refs_ids.append([self.ids_cut(data['resp'][j])])
                    all_pred_ids.append(self.ids_cut(preds[j], lens[j]))
            bleu += eval_bleu(all_refs_ids, all_pred_ids)
            dist += eval_distinct(all_pred_ids)
            f1 += eval_f1(all_refs_ids, all_pred_ids)

        return lm_loss, s2s_loss, bleu, dist, f1, s2s_nll_loss, total_tokens

    def _pred_sample(self, n_sample):
        with torch.no_grad():
            self.model.eval()
            samples_idxs = random.sample(range(len(self.valid_dataset)), n_sample)
            samples = PadBatchSeq(self.model.vocab.pad_token_id)([self.valid_dataset[idx] for idx in samples_idxs])
            context, context_seg_id = samples['context'].to(self.device, non_blocking=True), samples['context_seg_id'].to(self.device, non_blocking=True)
            img_feat = samples['img_feat'].to(self.device, non_blocking=True)
            img_feat = torch.max(img_feat, dim=1, keepdim=True)[0]

            post_rep = self.model.encode(context.clone(), context_seg_id, img_feat=img_feat)
            enc_contexts = []

            if hasattr(self.model.transformer_module, 'img_feat_proj'):
                enc_contexts.append((post_rep[0] + self.model.transformer_module.img_feat_proj(img_feat) * self.config.img_feat_weight, post_rep[1]))
            else:
                enc_contexts.append((post_rep[0] + self.model.transformer_module.module.img_feat_proj(img_feat) * self.config.img_feat_weight, post_rep[1]))

            prediction = self.model.beam_search(enc_contexts)

            # prediction, _ = self.model.top_k_top_p_search(
            #     [self.model.encode(samples['context'].to(self.device), samples['context_seg_id'].to(self.device))],
            #     top_k=self.config.top_k, top_p=self.config.top_p, sample_size=1)
            # prediction = prediction.cpu().tolist()
            res = []
            for j in range(len(samples_idxs)):
                post_str = samples['context'][j].tolist()
                post_str = self.model.vocab.ids2string_wo_eos(post_str)
                resp_str = samples['resp'][j].tolist()[1:]
                resp_str = self.model.vocab.ids2string_wo_eos(resp_str)
                pred_str = self.model.vocab.ids2string_wo_eos(prediction[j])
                res.append({"context": post_str, "resp": resp_str, "pred": pred_str})
        return res

    def ids_cut(self, ids, length=None):
        """
        Cut a list of ids to its real length. (ids[0] & eos would be cut)
        :param ids: A list of ids. Note: ids[0] would be ignored.
        :param length: Length of ids including eos (but eos would be cut). If is None,
            length would be inferred from ids.
        :return: Result id list.
        """
        if type(ids) is not list:
            ids = ids.tolist()
        if length is None:
            length = ids.index(self.model.vocab.eos_token_id)
        return ids[1: length]

    def test(self):
        self._eval_test()

    def train(self, start_epoch, epochs, after_epoch_funcs=[]):
        for epoch in range(start_epoch + 1, epochs):
            self.logger.info('Training on process {}, epoch {}, step {}'.format(
                self.rank, epoch, self.optimizer.curr_step()))
            if self.train_sampler and hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)
            self._eval_train(epoch)
            for func in after_epoch_funcs:
                func(epoch, self.device)
