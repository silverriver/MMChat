'''
Skip WeiboID that are already handled
Author: Silver
'''
import argparse
import os
import json
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--exist', help='exist weiboid file, we will skip these weiboid to avoid duplicated download', default='')
parser.add_argument('--input', help='input file', default='weibo_img_expanded_url.json')
parser.add_argument("--output", help='output file (the weiboids that are already downloaded are skipped)', default='weibo_img_expanded_url_unique_weiboid.json')
args = parser.parse_args()

exist_set = set()
for exist in args.exist.split(';'):
    if os.path.isfile(exist):
        print('opening {}'.format(exist))
        with open(exist) as f:
            res_exist = [json.loads(i) for i in f.readlines()]
        for i in res_exist:
            exist_set.add(i['weiboid'])

with open(args.input) as f:
    res = [json.loads(i) for i in f.readlines()]

with open(args.output, 'w') as f:
    for i in tqdm(res):
        if i['weiboid'] in exist_set:
            continue
        exist_set.add(i['weiboid'])
        print(json.dumps(i), file=f)
print('fin.')
