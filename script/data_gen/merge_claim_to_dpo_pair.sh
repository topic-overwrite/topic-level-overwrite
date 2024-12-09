#!/bin/bash

echo "----------Start merge_claim_to_dpo_pair----------"

num_gpus=$1
ckpt=$2
image_path=$3
classified_claim_path=$4
claim_reward_path=$5
merged_inf_path=$6
raw_response_path=$7
output_dir=$8
output_file=$9
ans_data_dir=${10}
dpo_pair_generate_method=${11}
prompt_type=${12}
repeat_num=${13}
start_pos=${14:-"0"}
end_pos=${15:-"-1"}
wh_type=${16:-"v1"}

ans_data_path=${ans_data_dir}/${output_file}

echo "num_gpus: "$num_gpus
echo "ckpt: "$ckpt
echo "image_path: "$image_path
echo "classified_claim_path: "$classified_claim_path
echo "claim_reward_path: "$claim_reward_path
echo "merged_inf_path: "$merged_inf_path
echo "raw_response_path: "$raw_response_path
echo "output_dir: "$output_dir
echo "ans_data_path: "$ans_data_path
echo "dpo_pair_generate_method: "$dpo_pair_generate_method
echo "prompt_type: "$prompt_type
echo "repeat_num: "$repeat_num
echo "wh_type: "$wh_type
echo "start_pos: "$start_pos"  end_pos: "$end_pos


python utils/generate_single_dpo_response.py \
    --image_path $image_path \
    --classified_claim_path $classified_claim_path \
    --claim_reward_path $claim_reward_path \
    --merged_inf_path $merged_inf_path \
    --raw_response_path $raw_response_path \
    --output_path ${output_dir}/dpo_single_response_without_merge.jsonl \
    --need_merge_path ${output_dir}/dpo_single_response_need_merge.jsonl \
    --dpo_pair_generate_method $dpo_pair_generate_method \
    --wh_type $wh_type\
    --repeat_num $repeat_num \
    --start_pos $start_pos \
    --end_pos $end_pos

if [[ $dpo_pair_generate_method == *"default"* ]]; then
    if [ ! -f "${output_dir}/dpo_single_response_merged.json" ]; then
        touch "${output_dir}/dpo_single_response_merged.json"
    else
        > "${output_dir}/dpo_single_response_merged.json"
    fi
else
    echo "[Generate Final Response]"
    bash script/data_gen/generate_diverse_response.sh \
        $num_gpus \
        $ckpt \
        $output_dir \
        dpo_single_response_need_merge.jsonl \
        $output_dir \
        dpo_single_response_merged.json \
        1
fi

python utils/merge_dpo_data_pairs.py \
    --image_path $image_path \
    --without_merge_path ${output_dir}/dpo_single_response_without_merge.jsonl \
    --merged_response_path ${output_dir}/dpo_single_response_merged.json \
    --output_path $ans_data_path \
    --prompt_type $prompt_type
