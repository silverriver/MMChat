#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=2  \
--master_port=8899 \
infer_beam.py \
--config   infer_config.json \
--out_file eval/infer_out.txt \
--gpu '2,7' \
