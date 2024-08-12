#!/bin/sh

ROOT=/playgound
MODEL_PATH=${ROOT}/checkpoints/Parrot/Parrot_S2_7B-Qwen15Clip

VT_PATH=${ROOT}/models/clip-vit-large-patch14-336

export ParrotPATH="${MODEL_PATH}"
export ParrotVTPATH="${VT_PATH}"
export LMUData="${ROOT}/LMUData"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 run.py \
        --data MMMB_en MMMB_zh MMMB_pt MMMB_ar MMMB_tr MMMB_ru \
        --model Parrot #--verbose