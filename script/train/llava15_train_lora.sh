#!/bin/bash

echo "----------Start llava15_train----------"

task_name=llava15_7b_DPO
exp_name=llava15_tpo_self
ckpt=${1:-"checkpoint/liuhaotian--llava-v1.5-7b"}
raw_data_path=${2:-"dataset/topic-overwrite"}
data_dir=${raw_data_path}-with-logps

echo "exp_name: "$exp_name
echo "raw_data_path: "$raw_data_path

PYTHONPATH=./:$PYTHONPATH \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
deepspeed muffin/train/train_llava15.py \
    --deepspeed script/zero2.json \
    --model_name_or_path $ckpt \
    --raw_data_path $raw_data_path \
    --data_dir $data_dir \
    --image_folder not_used \
    --vision_tower checkpoint/openai--clip-vit-large-patch14-336 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fully_tune False \
    --train_lora True \
    --image_aspect_ratio pad \
    --bf16 True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --output_dir checkpoint/$task_name-$exp_name \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --data_source_names '' \
    --data_source_weights 1 \
    --max_steps 2500 \
    --learning_rate 2e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --logging_dir checkpoint/$task_name-$exp_name/log \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --task DPO \
    --report_to tensorboard \
    --run_name $exp_name \
    --dataloader_num_workers 16 \
    --dpo_use_average False \
    --dpo_token_weighted False \
    --dpo_token_weight 1.0 \
    --dpo_beta 0.1 \
    # --save_steps 167 \
    # --save_total_limit 1 \