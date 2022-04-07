from metrics.eval_f1 import eval_f1
from metrics.eval_bleu import eval_bleu, eval_bleu_detail
from metrics.eval_distinct import eval_distinct
from tqdm import tqdm
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input file', default="dialog_model/eval/infer_out.txt-1")

args = parser.parse_args()

print(args.input)

with open(args.input) as f:
    res = [json.loads(i) for i in f.readlines()]
pred, ref = [], []

for d in tqdm(res):
    ref.append([list(d['response'])])
    pred.append(list(d['preds'][1]))

print('BLEU: {}, F1: {}, Distinct: {}'.format(eval_bleu(ref, pred), eval_f1(ref, pred), eval_distinct(pred)))
print('BLEU 1, 2, 3, 4: {}'.format(eval_bleu_detail(ref, pred)))
