#!/bin/bash

echo "----------Start generate_diverse_response----------"

num_gpus=$1
ckpt=$2
ques_dir=$3
ques_file=$4
ans_dir=$5
ans_file=$6
repeat_num=${7:-"10"}
start_pos=${8:-"0"}
end_pos=${9:-"-1"}

echo "num_gpus: "$num_gpus
echo "ckpt: "$ckpt
echo "question dir: "$ques_dir
echo "question file: "$ques_file
echo "answer dir: "$ans_dir
echo "answer file: "$ans_file
echo "repeat_num: "$repeat_num
echo "start_pos "$start_pos" end_pos "$end_pos


PYTHONPATH=./:$PYTHONPATH \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
torchrun --nnodes=1 --nproc_per_node=${num_gpus} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 muffin/llava15_gen_data.py \
    --checkpoint $ckpt \
    --ds_name ${ques_dir}/${ques_file} \
    --answer_file ${ans_dir}/${ans_file} \
    --max_sample -1 \
    --start_pos $start_pos \
    --end_pos $end_pos \
    --repeat $repeat_num \
    --max_tokens 512 \
    --num-workers 5 \
    --batch-size 8 \
    --temperature 0.7