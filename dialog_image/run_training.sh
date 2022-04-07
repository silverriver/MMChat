#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=2  \
train.py \
--config dialog_img/config.json \
--gpu '1,2' \
