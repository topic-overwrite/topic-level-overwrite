#!/bin/bash

ckpt_path=$1
base_path=${2:-"none"}
gpu_id=${3:-"0,1,2,3"}
data_dir=${4:-"dataset/amber"}

IFS=',' read -r -a gpu_list <<< "${gpu_id}"
num_gpus=${#gpu_list[@]}
base_name="${ckpt_path:11}"
answer_file=${data_dir}/model_output/${base_name}_output.json
current_time=$(date "+%Y-%m-%d--%H-%M-%S")
log_file=${ckpt_path}/amber_${current_time}.log

echo "log_file: "$log_file
echo "ckpt_path: "$ckpt_path
echo "data_dir: "$data_dir
echo "gpu_list: "$gpu_id
echo "answer_file: "$answer_file


chunk_counter=-1
for i in "${gpu_list[@]}"; do
    ((chunk_counter++))
    echo "===> gpu_id "$i"    chunk_id "$chunk_counter
    sleep 1
    CUDA_VISIBLE_DEVICES=$i \
    PYTHONPATH=./:$PYTHONPATH \
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python muffin/eval/inference_amber_data.py \
        --num-chunks $num_gpus \
        --chunk-idx $chunk_counter \
        --model-path $ckpt_path \
        --model-base $base_path \
        --question-file ${data_dir}/query/query_discriminative.json \
        --answers-file $answer_file \
        --temperature 0.7 \
        --conv-mode llava_v1 \
        --num_beams 3 &
done
wait

PYTHONPATH=./:$PYTHONPATH \
python utils/merge_json_data.py \
    --answers-file $answer_file \
    --num $num_gpus

PYTHONPATH=./:$PYTHONPATH \
python eval/eval_amber.py \
    --inference_data $answer_file \
    --evaluation_type d \
    | tee $log_file