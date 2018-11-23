#!/bin/bash

python3 run.py \
  DefeatRoaches \
  --map DefeatRoaches \
  --max_windows 1 --gpu 0 --envs 32 --save_iters 1000 \
  --max_to_keep 1000 --step_mul 4 --steps_per_batch 16 \
  --lr 7e-4  --value_loss_weight .5 --ow --vis \
  --save_dir /result/models \
  --summary_dir /result/summary
