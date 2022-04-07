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


class DialogDataset(Dataset):
    def __init__(self, paths, vocab, logger, max_context_len=2048, max_resp_len=2048):
        self.logger = logger
        self.vocab = vocab
        self.max_context_len = max_context_len
        self.max_resp_len = max_resp_len
        self.data = DialogDataset.make_dataset(paths, vocab, logger, max_context_len, max_resp_len)

    @staticmethod
    def make_dataset(paths, vocab, logger, max_context_len, max_resp_len):
        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:
            lines = []
            with open(path, 'r', encoding='utf8') as f:
                for line in [json.loads(i) for i in f.readlines() if len(i.strip()) != 0]:
                    lines.append([vocab.string2ids(i) for i in line])

            for line_index, line in enumerate(lines):
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

        logger.info('{} data record loaded'.format(len(dataset)))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, context_seg_id, resp, dialog_index, utt_index = self.data[idx]
        return {"context": context, "context_seg_id": context_seg_id, "resp": resp,
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
        res['dialog_index'] = torch.LongTensor([i['dialog_index'] for i in batch])
        res['utt_index'] = torch.LongTensor([i['utt_index'] for i in batch])
        return PinnedBatch(res)

