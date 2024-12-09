#!/bin/bash

echo "----------Start generate_yesno_question----------"

num_gpus=$1
ckpt=$2
claim_dir=$3
claim_filename=$4
wh_response_dir=$5
wh_response_file=$6
raw_ques_path=$7
image_dir=$8
merged_inf_file=$9
output_question_dir=${10}
output_question_file=${11}
batch_size=${12}
repeat_num=${13}
only_wh_question=${14:-"no"}
start_pos=${15:-"0"}
end_pos=${16:-"-1"}

claim_path=${claim_dir}/${claim_filename}
wh_response_path=${wh_response_dir}/${wh_response_file}
merged_inf_path=${output_question_dir}/${merged_inf_file}
output_question_path=${output_question_dir}/${output_question_file}

echo "num_gpus: "$num_gpus
echo "ckpt: "$ckpt
echo "claim_path: "$claim_path
echo "wh_response_path: "$wh_response_path
echo "raw_ques_path: "$raw_ques_path
echo "image_dir: "$image_dir
echo "merged_inf_path: "$merged_inf_path
echo "output_question_path: "$output_question_path
echo "batch_size: "$batch_size
echo "repeat_num: "$repeat_num
echo "start_pos "$start_pos" end_pos "$end_pos
echo "only_wh_question "$only_wh_question


srun \
    -p AI4Good_S \
    -J data-preprocess \
    --kill-on-bad-exit \
python utils/collect_all_claim_to_generate_yesno_question.py \
    --claim_path $claim_path \
    --wh_response_path $wh_response_path \
    --output_path ${output_question_dir}/raw_${output_question_file} \
    --only_wh_question $only_wh_question

#array=("0")
#for i in "${array[@]}"; do
for i in $(seq 0 $((num_gpus-1))); do
    echo "===> chuck_num "$i
    sleep 1
    CUDA_VISIBLE_DEVICES=$i \
    PYTHONPATH=./:$PYTHONPATH \
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python utils/llama3_inference.py \
        --checkpoint $ckpt \
        --path ${output_question_dir}/raw_${output_question_file} \
        --output_dir ${output_question_dir} \
        --chunk-num $num_gpus \
        --chunk-idx $i \
        --bs $batch_size \
        --prompt_type yesno \
        --start ${start_pos} \
        --end ${end_pos} &
done
wait

claim_filename="${output_question_file::-6}"

output_file=${output_question_dir}/${merged_inf_file}
> "$output_file"
for IDX in $(seq 0 $((num_gpus-1))); do
    echo "merge "${output_question_dir}/raw_${claim_filename}.s${start_pos}-e${end_pos}.chunk${num_gpus}-${IDX}.jsonl
    cat ${output_question_dir}/raw_${claim_filename}.s${start_pos}-e${end_pos}.chunk${num_gpus}-${IDX}.jsonl >> "$output_file"
done

python utils/get_yesno_question_with_image.py \
    --data_path $output_file \
    --raw_ques_path $raw_ques_path \
    --image_dir $image_dir \
    --output_path ${output_question_dir}/${output_question_file} \
    --repeat_num $repeat_num