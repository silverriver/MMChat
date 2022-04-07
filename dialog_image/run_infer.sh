#!/bin/bash

source ~/.bashrc
conda activate mchat

export LANG="zh_CN.UTF-8"

python3 -m torch.distributed.launch --nproc_per_node=1  \
infer_beam.py \
--config           model/dialog_image/config.json \
--test_dialog_data data/dialog_test.json \
--test_img_index   data/weibo_img_index_test.json \
--img_feat         data/weibo_img/image_feature_lmdb \
--infer_ckpt model-84.ckpt_small \
--out_file         model/dialog_image/eval/infer_out.txt \
--gpu '0' \
2>&1 | tee -i infer.log
