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

print('Entropy-1: {}'.format(eval_entropy(pred, 1)))
print('Entropy-2: {}'.format(eval_entropy(pred, 2)))
