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

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_module import TransformerModule
from .filtering import top_k_top_p_filtering


class MultiInputModel(nn.Module):
    def __init__(self, config, vocab):
        super(MultiInputModel, self).__init__()
        self.config = config
        self.vocab = vocab

        self.transformer_module = TransformerModule(config, len(vocab), vocab.pad_token_id)
        self.pre_softmax = nn.Linear(config.embeddings_size, len(vocab), bias=False)
        self.pre_softmax.weight = self.transformer_module.embeddings.weight

    def forward(self, x, contexts=[]):
        enc_contexts = [self.encode(c, c_seg_id) for c, c_seg_id in contexts]
        return self.decode(x, enc_contexts)

    def encode(self, x, x_seg_id=None, masked_selfatt=None):
        return self.transformer_module(x, x_seg_id=x_seg_id, masked_selfatt=masked_selfatt)

    def generate(self, enc_x):
        return self.pre_softmax(enc_x)

    def decode(self, x, enc_contexts=[], img_feat=None):
        x, _ = self.transformer_module(x, enc_contexts, img_feat=img_feat)
        return self.generate(x)

    def predict(self, contexts=[], img_feat=None):
        enc_contexts = [self.encode(c, c_seg_id) for c, c_seg_id in contexts]
        prediction = self.beam_search(enc_contexts, img_feat=img_feat)
        return prediction

    def predict_beam(self, contexts=[], img_feat=None):
        enc_contexts = [self.encode(c, c_seg_id) for c, c_seg_id in contexts]
        prediction = self.beam_search(enc_contexts, return_beams=True, img_feat=img_feat)

        return prediction

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.config.length_penalty / (5 + 1) ** self.config.length_penalty

    def beam_search(self, enc_contexts=[], img_feat=None, return_beams=False):
        with torch.no_grad():
            if len(enc_contexts) == 0:
                return []

            batch_size = enc_contexts[0][0].shape[0]
            device = next(self.parameters()).device

            prevs = torch.full((batch_size * self.config.beam_size, 1), fill_value=self.vocab.bos_token_id, dtype=torch.long,
                               device=device)

            beam_scores = torch.zeros(batch_size, self.config.beam_size, device=device)
            beam_lens = torch.ones(batch_size, self.config.beam_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size, self.config.beam_size, dtype=torch.uint8, device=device)

            beam_enc_contexts = []
            for c, p in enc_contexts:
                c = c.unsqueeze(1).repeat(1, self.config.beam_size, 1, 1)
                c = c.view(-1, c.shape[2], c.shape[3])
                p = p.unsqueeze(1).repeat(1, self.config.beam_size, 1)
                p = p.view(-1, p.shape[2])
                beam_enc_contexts.append((c, p))
            beam_img_feat = img_feat.unsqueeze(1).repeat(1, self.config.beam_size, 1, 1)
            beam_img_feat = beam_img_feat.view(-1, beam_img_feat.shape[2], beam_img_feat.shape[3])

            current_sample_prob = 1
            group_size = self.config.beam_size // self.config.diversity_groups
            diversity_penalty = torch.zeros((batch_size, len(self.vocab)), device=device)

            # zrs:
            repeat = [{} for i in range(batch_size * self.config.beam_size)]
            # **********
            for i in range(self.config.max_decode_len):
                outputs, _ = self.transformer_module(prevs, beam_enc_contexts, img_feat=beam_img_feat)

                logits = self.generate(outputs[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
                # zrs: remove n repeat. prevs: (batch_size*beam_size, 1)
                for idx in range(batch_size * self.config.beam_size):
                    for key in repeat[idx]:
                        for value in repeat[idx][key]:
                            log_probs[idx][value] = -1000
                # **********
                log_probs = log_probs.view(batch_size, self.config.beam_size, -1)

                beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1))
                # zrs, log_probs: batch * beam * dim
                ba, be, dim = beam_scores.shape
                for ba_idx in range(ba):
                    for be_idx in range(be):
                        if int(torch.max(beam_scores[ba_idx][be_idx]) == torch.min(beam_scores[ba_idx][be_idx])):
                            temp = float(beam_scores[ba_idx][be_idx][0])
                            beam_scores[ba_idx][be_idx] = -float('inf')
                            beam_scores[ba_idx][be_idx][0] = temp
                # **********
                penalty = self._length_penalty(beam_lens.float() + 1 - is_end.float())
                penalty = penalty.unsqueeze(-1).repeat(1, 1, len(self.vocab))
                beam_scores = beam_scores / penalty

                if i == 0:
                    penalty = penalty[:, 0, :]
                    beam_scores = beam_scores[:, 0, :]

                    beam_scores, idxs = beam_scores.topk(self.config.beam_size, dim=-1)
                    beam_idxs = torch.zeros((batch_size, self.config.beam_size), dtype=torch.long, device=device)
                else:

                    penalty = penalty.view(batch_size, self.config.diversity_groups, group_size, -1)
                    beam_scores = beam_scores.view(batch_size, self.config.diversity_groups, group_size, -1)

                    all_scores, all_idxs = [], []
                    for g in range(self.config.diversity_groups):
                        g_beam_scores = beam_scores[:, g, :, :]
                        g_penalty = penalty[:, g, :, :]
                        g_beam_scores -= self.config.diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
                        g_beam_scores = g_beam_scores.view(batch_size, -1)

                        if random.random() < current_sample_prob:
                            # print('*********')
                            beam_probas = F.softmax(g_beam_scores/self.config.temperature, dim=-1)
                            if self.config.annealing_topk is not None:
                                beam_probas, sample_idxs = beam_probas.topk(self.config.annealing_topk, dim=-1)
                                g_idxs = torch.multinomial(beam_probas, group_size)
                                g_idxs = torch.gather(sample_idxs, 1, g_idxs)
                            else:
                                g_idxs = torch.multinomial(beam_probas, group_size)
                        else:
                            # print('|||||||||')
                            _, g_idxs = g_beam_scores.topk(group_size, dim=-1)

                        g_scores = torch.gather(beam_scores[:, g, :, :].view(batch_size, -1), 1, g_idxs)
                        g_idxs += g * group_size * len(self.vocab)

                        all_scores.append(g_scores)
                        all_idxs.append(g_idxs)

                        diversity_penalty.scatter_add_(1, torch.fmod(g_idxs, len(self.vocab)),
                                                       torch.ones((batch_size, group_size), device=device))

                    diversity_penalty.fill_(0)
                    penalty = penalty.view(batch_size, -1)
                    beam_scores = torch.cat(all_scores, dim=-1)
                    idxs = torch.cat(all_idxs, dim=-1)

                    beam_idxs = (idxs.float() / len(self.vocab)).long()

                penalty = torch.gather(penalty, 1, idxs)
                sym_idxs = torch.fmod(idxs, log_probs.shape[-1])
                is_end = torch.gather(is_end, 1, beam_idxs).bool()
                beam_lens = torch.gather(beam_lens, 1, beam_idxs)

                sym_idxs[is_end] = self.vocab.pad_token_id
                beam_lens[~is_end] += 1
                is_end[sym_idxs == self.vocab.eos_token_id] = 1

                sym_idxs = sym_idxs.view(batch_size * self.config.beam_size, 1)
                prevs = prevs.view(batch_size, self.config.beam_size, -1)
                prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
                prevs = prevs.view(batch_size * self.config.beam_size, -1)
                prevs = torch.cat([prevs, sym_idxs], dim=1)

                # zrs:
                prevs_list = prevs.tolist()
                for b in range(batch_size * self.config.beam_size):
                    b_list = prevs_list[b]
                    if len(b_list) > 2 and b_list[-1] != self.vocab.pad_token_id and b_list[-1] != self.vocab.eos_token_id:
                        key = (int(b_list[-3]), int(b_list[-2]))
                        if key in repeat[b]:
                            repeat[b][key].append(int(b_list[-1]))
                        else:
                            repeat[b][key] = [int(b_list[-1])]
                # ********

                if all(is_end.view(-1)):
                    break

                beam_scores *= penalty
                current_sample_prob *= self.config.annealing

            predicts = []
            result = prevs.view(batch_size, self.config.beam_size, -1)

            if return_beams:
                bests = torch.argsort(beam_scores, dim=-1, descending=True)
                for i in range(batch_size):
                    temp = []
                    for j in range(self.config.beam_size):
                        best_len = beam_lens[i, bests[i][j]]
                        best_seq = result[i, bests[i][j], 1:best_len - 1]
                        temp.append(best_seq.tolist())
                    predicts.append(temp)
                return predicts

            if self.config.sample:
                probs = F.softmax(beam_scores, dim=-1)
                bests = torch.multinomial(probs, 1).view(-1)
            else:
                bests = beam_scores.argmax(dim=-1)

            for i in range(batch_size):
                best_len = beam_lens[i, bests[i]]
                best_seq = result[i, bests[i], 1:best_len - 1]
                predicts.append(best_seq.tolist())
        return predicts

    def top_k_top_p_search(self, enc_contexts=[], img_feat=None, top_k=0, top_p=0.0, filter_value=-float('Inf'),
                           sample_size=5, temperature=0.8):
        with torch.no_grad():
            batch_size = enc_contexts[0][0].shape[0]
            device = next(self.parameters()).device

            prevs = torch.full((batch_size * sample_size, 1), fill_value=self.vocab.bos_token_id,
                               dtype=torch.long, device=device)

            lens = torch.ones(batch_size * sample_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size * sample_size, device=device).bool()

            beam_enc_contexts = []
            for c, p in enc_contexts:
                c = c.unsqueeze(1).repeat(1, sample_size, 1, 1)
                c = c.view(-1, c.shape[2], c.shape[3])
                p = p.unsqueeze(1).repeat(1, sample_size, 1)
                p = p.view(-1, p.shape[2])
                beam_enc_contexts.append((c, p))
            beam_img_feat = img_feat.unsqueeze(1).repeat(1, sample_size, 1, 1)
            beam_img_feat = beam_img_feat.view(-1, beam_img_feat.shape[2], beam_img_feat.shape[3])

            for i in range(self.config.max_decode_len):
                outputs, _ = self.transformer_module(prevs, beam_enc_contexts, img_feat=beam_img_feat)
                logits = self.generate(outputs[:, -1, :]) / temperature
                filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1).view(-1)

                next_token[is_end] = self.vocab.pad_token_id
                lens[~is_end] += 1
                is_end[next_token == self.vocab.eos_token_id] = 1
                prevs = torch.cat((prevs, next_token.view(-1, 1)), dim=-1)

                if all(is_end.view(-1)):
                    break

        return prevs, lens
