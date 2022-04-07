#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=2  \
train.py \
--config config.json \
--gpu '0,1' \
