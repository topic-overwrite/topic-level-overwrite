#!/bin/bash

echo "----------Start check_claim_reward----------"

num_gpus=$1
node_num=$2
ckpt=$3
ques_dir=$4
ques_file=$5
ans_dir=$6
ans_file=$7
start_pos=${8:-"0"}
end_pos=${9:-"-1"}

ques_path=${ques_dir}/${ques_file}
ans_path=${ans_dir}/${ans_file}

echo "num_gpus: "$num_gpus
echo "node_num: "$node_num
echo "ckpt: "$ckpt
echo "ques_path: "$ques_path
echo "ans_path: "$ans_path
echo "start_pos "$start_pos" end_pos "$end_pos


PYTHONPATH=./:$PYTHONPATH \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
torchrun --nnodes=1 --nproc_per_node=${num_gpus} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 muffin/llava15_gen_data.py \
    --checkpoint $ckpt \
    --ds_name $ques_path \
    --answer_file $ans_path \
    --max_sample -1 \
    --start_pos $start_pos \
    --end_pos $end_pos \
    --max_tokens 1 \
    --num-workers 5 \
    --batch-size 1 \
    --is_yesno