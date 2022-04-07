from metrics.eval_f1 import eval_f1
from metrics.eval_bleu import eval_bleu, eval_bleu_detail
from metrics.eval_distinct import eval_distinct
from metrics.eval_entropy import eval_entropy
from tqdm import tqdm
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input file', default="dialog_model/eval/infer_out.txt-1")

args = parser.parse_args()

print(args.input)

with open(args.input, 'r', encoding='utf-8') as f:
    res = [json.loads(i) for i in f.readlines()]
pred, ref = [], []

for d in tqdm(res):
    ref.append([list(d['response'])])
    pred.append(list(d['preds'][1]))

print('BLEU: {:.3f}, F1: {:.2f}, Distinct-1: {:.2f}, Distinct-2: {:.2f}'.format(eval_bleu(ref, pred), eval_f1(ref, pred)*100, eval_distinct(pred, 1), eval_distinct(pred, 2)))
print('BLEU 1, 2, 3, 4: {}'.format(eval_bleu_detail(ref, pred)))
print('Entropy-1: {:.2f}'.format(eval_entropy(pred, 1)))
print('Entropy-2: {:.2f}'.format(eval_entropy(pred, 2)))

print('---- Response -----')
print('Distinct-1: {:.2f}, Distinct-2: {:.2f}'.format(eval_distinct(ref, 1), eval_distinct(ref, 2)))
print('Entropy-1: {:.2f}'.format(eval_entropy(ref, 1)))
print('Entropy-2: {:.2f}'.format(eval_entropy(ref, 2)))
