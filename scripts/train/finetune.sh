#!/bin/bash

ROOT=/playgound
EXPNAME=Parrot_S2_7B-Qwen15Clip

deepspeed parrot/train/train.py \
          --deepspeed scripts/train/zero3.json \
          --model_name parrot_qwen2 \
          --model_path ${ROOT}/checkpoints/Parrot/Parrot_S1_7B-Qwen15Clip \
          --num_experts 6 \
          --moe_intermediate_size 4096 \
          --use_moe_residual True \
          --use_moe True \
          --moe_top_k 6 \
          --mm_vision_select_feature cls_patch \
          --data_name 'llava-1.5-finetune|sharegpt4v-sft-zh|sharegpt4v-sft-pt|sharegpt4v-sft-ar|sharegpt4v-sft-tr|sharegpt4v-sft-ru' \
          --is_multimodal True \
          --image_aspect_ratio pad \
          --vision_tower ${ROOT}/models/clip-vit-large-patch14-336 \
          --freeze_vision_tower True \
          --mm_projector_type mlp2x_gelu \
          --tune_mm_mlp_adapter False \
          --mm_vision_select_layer -2 \
          --bf16 True \
          --output_dir ${ROOT}/checkpoints/Parrot/${EXPNAME} \
          --group_by_modality_length False \
          --num_train_epochs 1 \
          --per_device_train_batch_size 4 \
          --gradient_accumulation_steps 4 \
          --evaluation_strategy no \
          --save_strategy steps \
          --save_steps 1000 \
          --save_total_limit 2 \
          --learning_rate 2e-5 \
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
