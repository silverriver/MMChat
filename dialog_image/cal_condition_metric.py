from metrics.eval_f1 import eval_f1
from metrics.eval_bleu import eval_bleu, eval_bleu_detail
from metrics.eval_distinct import eval_distinct
from metrics.eval_entropy import eval_entropy
from tqdm import tqdm
import json
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input file', default="dialog_model/eval/infer_out.txt-1")
parser.add_argument('--show_human_res', action="store_true", help='true if show human result')

def print_perform(ref, pred):
    print('BLEU: {:.3f}, F1: {:.2f}, Distinct-1: {:.2f}, Distinct-2: {:.2f}'.format(eval_bleu(ref, pred), eval_f1(ref, pred)*100, eval_distinct(pred, 1), eval_distinct(pred, 2)), end=' ')
    print('BLEU 1, 2, 3, 4: {}'.format(eval_bleu_detail(ref, pred)), end=' ')
    print('Entropy-1: {:.2f}'.format(eval_entropy(pred, 1)), end=' ')
    print('Entropy-2: {:.2f}'.format(eval_entropy(pred, 2)))

def print_latex(ref, pred):
    # {BLEU-1,2,3,4} & Dist-1 & Dist-2 & Ent-1 & Ent-2
    # & - & - & - & - & - & - & - &- \\
    print(f'& {" & ".join(eval_bleu_detail(ref, pred))} & {eval_distinct(pred, 1):.2f} & {eval_distinct(pred, 2):.2f} & {eval_entropy(pred, 1):.2f} & {eval_entropy(pred, 2):.2f} \\\\')

args = parser.parse_args()

print(args.input)

with open(args.input, 'r', encoding='utf-8') as f:
    res = [json.loads(i) for i in f.readlines()]


turns = [1, 2, 3, 4, 5]

pred, ref = defaultdict(list), defaultdict(list)


for d in tqdm(res):
    sent_l = d['context'].count('[SEP]')
    ref[sent_l].append([list(d['response'])])
    pred[sent_l].append(list(d['preds'][1]))

for k in turns:
    print(f'== Context len: {k}', end=' ')
    # print_perform(ref[k], pred[k])
    print_latex(ref[k], pred[k])

if args.show_human_res:
    for k in turns:
        print(f'== Context len: {k} & - & - & - & - & {eval_distinct(pred[k], 1):.2f} & {eval_distinct(pred[k], 2):.2f} & {eval_entropy(pred[k], 1):.2f} & {eval_entropy(pred[k], 2):.2f} \\\\')

