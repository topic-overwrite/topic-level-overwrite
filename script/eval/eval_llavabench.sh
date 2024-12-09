#!/bin/bash

ckpt_path=$1
base_path=${2:-"none"}
openai_key=${3:-"123"}
gpu_id=${4:-"0"}
data_dir=${5:-"dataset/llava_bench"}

base_name="${ckpt_path:11}"
answer_file_name=${base_name}
current_time=$(date "+%Y-%m-%d--%H-%M-%S")
log_file=${ckpt_path}/llava_bench_${current_time}.log

echo "log_file: "$log_file
echo "ckpt_path: "$ckpt_path
echo "data_dir: "$data_dir
echo "gpu_id: "$gpu_id


CUDA_VISIBLE_DEVICES=$gpu_id \
PYTHONPATH=./:$PYTHONPATH \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python muffin/eval/inference_llava_bench.py \
    --model-path $ckpt_path \
    --question-file ${data_dir}/data \
    --answers-file ${data_dir}/model_output/${answer_file_name}.jsonl \
    --model-base $base_path \
    --temperature 0.0 \
    --conv-mode llava_v1 \
    --num_beams 3


echo "========>Done generating answers<========"

echo "========>Start evaluating answers<========"


PYTHONPATH=./:$PYTHONPATH \
python eval/eval_llava_bench.py \
    --apikey $openai_key \
    --question ${data_dir}/model_output/${answer_file_name}_ques.jsonl \
    --context ${data_dir}/model_output/${answer_file_name}_context.jsonl \
    --rule dataset/llava_bench/rule.json \
    --answer-list \
        ${data_dir}/model_output/${answer_file_name}_gpt.jsonl \
        ${data_dir}/model_output/${answer_file_name}.jsonl \
    --output ${data_dir}/model_output/${answer_file_name}_score.jsonl


PYTHONPATH=./:$PYTHONPATH \
python eval/summarize_gpt_llava_bench_review.py -f ${data_dir}/model_output/${answer_file_name}_score.jsonl | tee $log_file