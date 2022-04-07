'''
Filter out images that correspond to a broken url 
Author: Silver
'''
import argparse
import os
import json
from url2imgid import url2imgid

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_index', help='input image index file', default='weibo_img_expanded_url.json')
parser.add_argument('--output_img_index', help='output image index file', default='weibo_img_index.json')
parser.add_argument('--image_dir', help='weibo file', default='weibo_img/image')

args = parser.parse_args()

downloaded_imgs = set(os.listdir(args.image_dir))

print('{} image downloaded'.format(len(downloaded_imgs)))

with open(args.input_img_index) as f:
    res = [json.loads(i) for i in f.readlines()]

valid_img = 0
broken_img = 0
redu_img = 0

used_img = set()
with open(args.output_img_index, 'w') as f:
    for i in res:
        tmp = {}
        tmp['weiboid'] = i['weiboid']
        img = []
        if i['weibo_img'] != '':
            for i in i['weibo_img'].split(';'):
                file_name = url2imgid(i)
                if file_name in downloaded_imgs:
                    img.append(file_name)
                    used_img.add(file_name)
                    valid_img += 1
                else:
                    broken_img += 1
        tmp['weibo_img_path'] = ';'.join(img)
        print(json.dumps(tmp), file=f)

print('valid_img', valid_img)
print('broken_img', broken_img)
print('redu_img', len(downloaded_imgs) - len(used_img))
