#!/bin/bash

echo "----------Start generate_wh_question----------"

num_gpus=$1
ckpt=$2
claim_dir=$3
claim_filename=$4
raw_ques_path=$5
image_dir=$6
output_question_dir=$7
output_question_file=$8
batch_size=$9
repeat_num=${10}
wh_type=${11:-"v1"}
start_pos=${12:-"0"}
end_pos=${13:-"-1"}

claim_path=${claim_dir}/${claim_filename}
output_question_path=${output_question_dir}/${output_question_file}

echo "num_gpus: "$num_gpus
echo "ckpt: "$ckpt
echo "claim_path: "$claim_path
echo "raw_ques_path: "$raw_ques_path
echo "image_dir: "$image_dir
echo "output_question_path: "$output_question_path
echo "batch_size: "$batch_size
echo "repeat_num: "$repeat_num
echo "wh_type: "$wh_type
echo "start_pos "$start_pos" end_pos "$end_pos


#array=("8" "9" "0")
#for i in "${array[@]}"; do
for i in $(seq 0 $((num_gpus-1))); do
    echo "===> chuck_num "$i
    sleep 1
    CUDA_VISIBLE_DEVICES=$i \
    PYTHONPATH=./:$PYTHONPATH \
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python utils/llama3_inference.py \
        --checkpoint $ckpt \
        --path ${claim_path} \
        --output_dir ${output_question_dir} \
        --chunk-num $num_gpus \
        --chunk-idx $i \
        --bs $batch_size \
        --prompt_type what \
        --start ${start_pos} \
        --end ${end_pos} &
done
wait

claim_filename="${claim_filename::-6}"

output_file=${output_question_dir}/raw_${output_question_file}
> "$output_file"
for IDX in $(seq 0 $((num_gpus-1))); do
    echo "merge "${output_question_dir}/${claim_filename}.s${start_pos}-e${end_pos}.chunk${num_gpus}-${IDX}.jsonl
    cat ${output_question_dir}/${claim_filename}.s${start_pos}-e${end_pos}.chunk${num_gpus}-${IDX}.jsonl >> "$output_file"
done

python utils/generate_wh_question_for_llava.py \
    --data_path $output_file \
    --raw_ques_path $raw_ques_path \
    --image_dir $image_dir \
    --output_path ${output_question_dir}/${output_question_file} \
    --repeat_num $repeat_num \
    --wh_type $wh_type
