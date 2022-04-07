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

class Vocab:
    SEP = '[SEP]'
    PAD = '[PAD]'
    EOS = '[EOS]'
    UNK = '[UNK]'

    P1 = '[P1]'
    P2 = '[P2]'
    WEIBO = '[WEIBO]'
    KEYWORD = '[keyword]'
    IMG = '[IMG]'
    BOS = '[BOS]'
    CLS = '[CLS]'

    def __init__(self, vocab_file):
        # Token used in the pre-training process
        self.pretrained_spec_tokens = [Vocab.SEP, Vocab.PAD, Vocab.EOS, Vocab.UNK]
        with open(vocab_file, 'r', encoding='utf8') as fr:
            vocab = [line.strip('\n').split()[0] for line in fr.readlines()]

        added_spec_tokens = [Vocab.P1, Vocab.P2, Vocab.WEIBO, Vocab.KEYWORD, Vocab.IMG, Vocab.BOS, Vocab.CLS]
        vocab = self.pretrained_spec_tokens + vocab + added_spec_tokens
        self.spec_tokens = self.pretrained_spec_tokens + added_spec_tokens
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for i, t in enumerate(vocab)}

    def __len__(self):
        return len(self.token2id)

    @property
    def n_special_tokens(self):
        return len(self.spec_tokens)

    @property
    def special_tokens_ids(self):
        return [self.token2id[t] for t in self.spec_tokens]

    @property
    def pad_token_id(self):
        return self.token2id[Vocab.PAD]

    @property
    def unk_token_id(self):
        return self.token2id[Vocab.UNK]

    @property
    def sep_token_id(self):
        return self.token2id[Vocab.SEP]

    @property
    def p1_token_id(self):
        return self.token2id[Vocab.P1]

    @property
    def p2_token_id(self):
        return self.token2id[Vocab.P2]

    @property
    def bos_token_id(self):
        return self.token2id[Vocab.BOS]

    @property
    def eos_token_id(self):
        return self.token2id[Vocab.EOS]

    @property
    def weibo_token_id(self):
        return self.token2id[Vocab.WEIBO]

    @property
    def keyword_token_id(self):
        return self.token2id[Vocab.KEYWORD]

    @property
    def cls_token_id(self):
        return self.token2id[Vocab.CLS]

    @property
    def img_token_id(self):
        return self.token2id[Vocab.IMG]

    def string2ids(self, string):
        tokens = list(''.join(string.split()))
        ids = [self.token2id[t] if t in self.token2id else self.unk_token_id for t in tokens]
        return ids

    def ids2string(self, ids):
        tokens = [self.id2token[id] for id in ids]
        return ''.join(tokens)

    def ids2string_wo_eos(self, ids):
        res = ''
        for id in ids:
            if id == self.eos_token_id or id == self.pad_token_id:
                return res
            else:
                res += self.id2token[id]
        return res
