#!/bin/sh

CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 run.py \
        --data MMMB \
        --model Parrot