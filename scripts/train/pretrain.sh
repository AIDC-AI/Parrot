#!/bin/bash

ROOT=/playgound
EXPNAME=Parrot_S1_7B-Qwen15Clip

deepspeed parrot/train/train.py \
        --deepspeed scripts/train/zero2.json \
        --model_name parrot_qwen2 \
        --model_path ${ROOT}/models/Qwen1.5-7B-Chat \
        --num_experts 6 \
        --moe_intermediate_size 4096 \
        --use_moe_residual True \
        --use_moe False \
        --moe_top_k 6 \
        --mm_vision_select_feature cls_patch \
        --data_name 'llava-1.5-pretrain|laion-12k|cc12m-645k' \
        --is_multimodal True \
        --image_aspect_ratio pad \
        --vision_tower ${ROOT}/models/clip-vit-large-patch14-336 \
        --freeze_vision_tower True \
        --mm_projector_type mlp2x_gelu \
        --tune_mm_mlp_adapter True \
        --mm_vision_select_layer -2 \
        --bf16 True \
        --output_dir ${ROOT}/checkpoints/Parrot/${EXPNAME} \
        --group_by_modality_length False \
        --num_train_epochs 1 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 1000 \
        --save_total_limit 2 \
        --learning_rate 1e-3 \
        --max_grad_norm 1.0 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --run_name ${EXPNAME}
