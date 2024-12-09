#!/bin/bash

echo "----------Start data_pipeline----------"

num_gpus=8

generate_response_ckpt="checkpoint/liuhaotian--llava-v1.5-7b"
split_to_claim_ckpt="checkpoint/Meta-Llama-3-8B-Instruct"
classify_claim_ckpt="checkpoint/Meta-Llama-3-8B-Instruct"
generate_wh_question_ckpt="checkpoint/Meta-Llama-3-8B-Instruct"
generate_yesno_question_ckpt="checkpoint/Meta-Llama-3-8B-Instruct"
check_claim_reward_ckpt="checkpoint/liuhaotian--llava-v1.6-34b" # checkpoint/liuhaotian--llava-v1.6-34b  checkpoint/liuhaotian--llava-v1.5-7b
clip_ckpt="checkpoint/openai--clip-vit-large-patch14-336"
reorganize_response_ckpt=$generate_response_ckpt

ques_dir="dataset/raw-question-with-image"
ques_file="question.jsonl"
image_dir="dataset/raw-image-dir"

base_dir="dataset/tpo_pair21000/"

generated_response_dir=$base_dir"generated-response"
generated_response_file="generated_response.json"
splited_claim_dir=$base_dir"decompose-data"
splited_claim_file="splited_claim.jsonl"
classified_claim_dir=$base_dir"classified-claim"
classified_claim_file="classified_claim_louvain_p1_p2_p3_095_llama3_clipvit.json"
wh_question_dir=$base_dir"generated-claim-question"
wh_question_file="generated_v1_claim_wh_question.jsonl"
wh_response_dir=$base_dir"generated-response"
wh_response_file="wh_question_v1_response.json"
yesno_question_dir=$base_dir"generated-claim-question"
yesno_question_file="generated_claim_yesno_question.jsonl"
merged_inf_file="merged_inf.jsonl"
claim_reward_dir=$base_dir"generated-response"
claim_reward_file="claim_reward_llava16.json"
claim_relative_reward_file="claim_relative_reward_llava16.json"
dpo_pairs_dir=$base_dir"max-all-louvain-repeat10-whv1-restructionv1-generated-response"
dpo_pairs_file="dpo_pairs.parquet"
ans_data_dir=$base_dir"max-all-main-postv31-repeat10-round0-traindata"
ans_logps_data_dir=$ans_data_dir"-with-logps"

llama_inference_batch_size=8
repeat_num="10"
start_pos="0"
end_pos="-1"
dpo_pair_generate_method="max_all_claim"  # default_v1, max_one_claim, max_all_claim, all2all
prompt_type="default" # default, all2all
wh_type="v1" # v1, no
cluster_type="louvain" # tarjan, louvain
use_image_classify="yes" # yes, no
image_threshold="0.9" # 0.8, 0.9, 0.95


if [ ! -d "$generated_response_dir" ]; then
  mkdir "$generated_response_dir"
fi
if [ ! -d "$splited_claim_dir" ]; then
  mkdir "$splited_claim_dir"
fi
if [ ! -d "$classified_claim_dir" ]; then
  mkdir "$classified_claim_dir"
fi
if [ ! -d "$wh_question_dir" ]; then
  mkdir "$wh_question_dir"
fi
if [ ! -d "$wh_response_dir" ]; then
  mkdir "$wh_response_dir"
fi
if [ ! -d "$yesno_question_dir" ]; then
  mkdir "$yesno_question_dir"
fi
if [ ! -d "$claim_reward_dir" ]; then
  mkdir "$claim_reward_dir"
fi
if [ ! -d "$dpo_pairs_dir" ]; then
  mkdir "$dpo_pairs_dir"
fi
if [ ! -d "$ans_data_dir" ]; then
  mkdir "$ans_data_dir"
fi
if [ ! -d "$ans_logps_data_dir" ]; then
  mkdir "$ans_logps_data_dir"
fi

if [ "$num_gpus" -gt 8 ]; then
    gpus_per_node=8
    node_num=$(expr $num_gpus / $gpus_per_node)
else
    gpus_per_node=$num_gpus
    node_num=1
fi
echo "num_gpus:"$num_gpus
echo "gpus_per_node:"$gpus_per_node
echo "node_num:"$node_num



bash script/data_gen/generate_diverse_response.sh \
    $gpus_per_node \
    $generate_response_ckpt \
    $ques_dir \
    $ques_file \
    $generated_response_dir \
    $generated_response_file \
    $repeat_num \
    $start_pos \
    $end_pos



bash script/data_gen/split_response_to_claim.sh \
    $num_gpus \
    $split_to_claim_ckpt \
    $generated_response_dir \
    $generated_response_file \
    $splited_claim_dir \
    $splited_claim_file \
    $llama_inference_batch_size



bash script/data_gen/generate_wh_question.sh \
    $num_gpus \
    $generate_wh_question_ckpt \
    $splited_claim_dir \
    $splited_claim_file \
    ${ques_dir}/${ques_file} \
    $image_dir \
    $wh_question_dir \
    $wh_question_file \
    $llama_inference_batch_size \
    $repeat_num \
    $wh_type



bash script/data_gen/generate_diverse_response.sh \
    $gpus_per_node \
    $generate_response_ckpt \
    $wh_question_dir \
    $wh_question_file \
    $wh_response_dir \
    $wh_response_file \
    1



bash script/data_gen/generate_yesno_question.sh \
    $num_gpus \
    $generate_yesno_question_ckpt \
    $splited_claim_dir \
    $splited_claim_file \
    $wh_response_dir \
    $wh_response_file \
    ${ques_dir}/${ques_file} \
    $image_dir \
    $merged_inf_file \
    $yesno_question_dir \
    $yesno_question_file \
    $llama_inference_batch_size \
    $repeat_num



bash script/data_gen/check_claim_reward.sh \
    $gpus_per_node \
    $node_num \
    $check_claim_reward_ckpt \
    $yesno_question_dir \
    $yesno_question_file \
    $claim_reward_dir \
    $claim_reward_file



bash script/data_gen/classify_claim.sh \
    $num_gpus \
    $classify_claim_ckpt \
    $clip_ckpt \
    ${wh_question_dir}/raw_${wh_question_file} \
    ${ques_dir}/${ques_file} \
    $classified_claim_dir \
    $classified_claim_file \
    $llama_inference_batch_size \
    $image_threshold \
    $repeat_num \
    $use_image_classify \
    $cluster_type



bash script/data_gen/merge_claim_to_dpo_pair.sh \
    $gpus_per_node \
    $reorganize_response_ckpt \
    ${ques_dir}/${ques_file} \
    ${classified_claim_dir}/${classified_claim_file} \
    ${claim_reward_dir}/${claim_reward_file} \
    ${yesno_question_dir}/${merged_inf_file} \
    ${generated_response_dir}/${generated_response_file} \
    $dpo_pairs_dir \
    $dpo_pairs_file \
    $ans_data_dir \
    $dpo_pair_generate_method \
    $prompt_type \
    $repeat_num \
    $start_pos \
    $end_pos \
    $wh_type
