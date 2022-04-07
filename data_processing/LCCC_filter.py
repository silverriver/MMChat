'''
Filter dialouges based on the dialogues in LCCC
Author: Silver
'''

import json
import os
import argparse
import pygtrie
from tqdm import trange, tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--raw_dialog_folder', help='folder of raw dialogs', default='')
parser.add_argument('--lccc_path', help='path to lccc dialogs', default='')
args = parser.parse_args()

def leaf(): pass

if __name__ == '__main__':
    folders = [args.raw_dialog_folder]
    trie = pygtrie.StringTrie()
    with open(args.lccc_path) as f:
        lccc = [json.loads(i) for i in f.readlines()]

    for i in trange(len(lccc)):
        lccc[i] = [''.join(j.split()) for j in lccc[i]]
        trie['/'.join(lccc[i])] = leaf
    
    print('trie built')
    for folder in folders:
        print(folder)
        with open(os.path.join(folder, 'dialog.json')) as f:
            dialog = ['/'.join([''.join(j.split()) for j in json.loads(i)]) for i in f.readlines()]

        with open(os.path.join(folder, 'weibo_img_expanded_url.json')) as f:
            weibo_img = [json.loads(i) for i in f.readlines()]
        
        index = []
        for i in trange(len(dialog)):
            if len(weibo_img[i]['weibo_img']) == 0:
                continue

            if trie.has_node(dialog[i]) & (pygtrie.Trie.HAS_VALUE | pygtrie.Trie.HAS_SUBTRIE):
                index.append(i)
            else:
                res = trie.longest_prefix(dialog[i])
                if res.key is not None and sum([1 for j in res.key if j == '/']) > 2:
                    index.append(i)
        print(len(index))
        del dialog
        del weibo_img
        index = set(index)
        for file in ['dialog', 'weibo', 'weibo_img_expanded_url', 'index']:
            with open(os.path.join(folder, f'{file}_lccc_flt.json'), 'w') as f_out:
                with open(os.path.join(folder, f'{file}.json')) as f_in:
                    count = 0
                    for i in f_in.readlines():
                        if count in index:
                            f_out.write(i)
                        count += 1
