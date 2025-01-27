#!/bin/bash

ngpus=4
bs=1
grad_accum=4

torchrun --standalone --nnodes 1 --nproc-per-node $ngpus vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /home/reuss/tensorflow_datasets \
  --dataset_name kit_irl_real_kitchen_lang \
  --batch_size $bs \
  --grad_accumulation_steps $grad_accum \
  --wandb_project openvla_kit_kitchen_lora \
  --wandb_entity omeryagmurlu