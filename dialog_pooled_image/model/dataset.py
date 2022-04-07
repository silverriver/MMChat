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

from torch.utils.data import Dataset
import torch
import json
import lmdb
import os
import pickle


class DialogDataset(Dataset):
    def __init__(self, dialog_paths, img_index_paths, img_feat_dir, vocab, logger, max_context_len=2048, max_resp_len=2048, feat_num=50):
        self.logger = logger
        self.vocab = vocab
        self.max_context_len = max_context_len
        self.max_resp_len = max_resp_len
        self.feat_num = 100
        self.feat_env = lmdb.open(img_feat_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

        with self.feat_env.begin(write=False) as txn:
            self.feat_ids = pickle.loads(txn.get('keys'.encode()))
        self.feat_ids_set = set(self.feat_ids)
        self.feat_num = 50

        self.img_index = self.make_img_dataset(img_index_paths, logger)

        self.dialog_data, self.dialg_idx2session_idx = DialogDataset.make_dialog_dataset(dialog_paths, vocab, logger, max_context_len, max_resp_len, self.img_index)

    def make_img_dataset(self, paths, logger):
        logger.info('reading img index from {}'.format(paths))
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                res = [json.loads(i) for i in f.readlines()]
            for i in range(len(res)):
                imgs = [j.split('.')[0].encode() for j in res[i]['weibo_img_path'].split(';')]
                dataset.append([j for j in imgs if j in self.feat_ids_set])

        logger.info('{} img index loaded'.format(len(dataset)))
        return dataset

    @staticmethod
    def make_dialog_dataset(paths, vocab, logger, max_context_len, max_resp_len, img_index):
        logger.info('reading data from {}'.format(paths))
        dialog_idx2cap_idx = []
        dataset = []
        count = -1
        filtered_count = 0
        for path in paths:
            lines = []
            with open(path, 'r', encoding='utf8') as f:
                for line in [json.loads(i) for i in f.readlines()]:
                    lines.append([vocab.string2ids(i) for i in line])

            for line_index, line in enumerate(lines):
                count += 1
                if len(img_index[count]) == 0:
                    filtered_count += 1
                    continue
                for index in range(1, len(line)):
                    context = []
                    context_seg_id = []
                    for i in range(index):
                        context += line[i] + [vocab.sep_token_id]
                        context_seg_id += ([vocab.p1_token_id if i % 2 == 0 else vocab.p2_token_id] * (len(line[i]) + 1))
                    context = [vocab.cls_token_id] + context[-max_context_len:]
                    context_seg_id = [vocab.p1_token_id] + context_seg_id[-max_context_len:]

                    resp = [vocab.bos_token_id] + line[index][:max_resp_len] + [vocab.eos_token_id]
                    dataset.append([context, context_seg_id, resp, line_index, index])
                    dialog_idx2cap_idx.append(count)

        logger.info('{} data record loaded, {} lines filtered'.format(len(dataset), filtered_count))
        return dataset, dialog_idx2cap_idx

    def __len__(self):
        return len(self.dialog_data)

    def __getitem__(self, idx):
        context, context_seg_id, resp, dialog_index, utt_index = self.dialog_data[idx]
        imgs = self.img_index[self.dialg_idx2session_idx[idx]]
        feat_size = [self.feat_num // len(imgs)] * len(imgs)

        if len(imgs) > 1:
            feat_size[-1] = self.feat_num - sum(feat_size[:-1])

        feat = []
        with self.feat_env.begin(write=False) as txn:
            for size, img in zip(feat_size, imgs):
                item = pickle.loads(txn.get(img))
                feat.append(torch.tensor(item['features'][:size, :]))
            feat = torch.cat(feat, dim=0)
        return {"context": context, "context_seg_id": context_seg_id, "resp": resp, "img_feat": feat, 
                "dialog_index": dialog_index, "utt_index": utt_index}


class PinnedBatch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, k):
        return self.data[k]

    def pin_memory(self):
        for k in self.data.keys():
            self.data[k] = self.data[k].pin_memory()
        return self


class PadBatchSeq:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        res = dict()
        context_max_len = max([len(i['context']) for i in batch])
        resp_max_len = max([len(i['resp']) for i in batch])
        res['context'] = torch.LongTensor([i['context'] + [self.pad_id] * (context_max_len - len(i['context'])) for i in batch])
        res['context_seg_id'] = torch.LongTensor([i['context_seg_id'] + [self.pad_id] * (context_max_len - len(i['context_seg_id'])) for i in batch])
        res['resp'] = torch.LongTensor([i['resp'] + [self.pad_id] * (resp_max_len - len(i['resp'])) for i in batch])

        res['img_feat'] = torch.stack([i['img_feat'] for i in batch], dim=0)
        res['dialog_index'] = torch.LongTensor([i['dialog_index'] for i in batch])
        res['utt_index'] = torch.LongTensor([i['utt_index'] for i in batch])
        return PinnedBatch(res)

