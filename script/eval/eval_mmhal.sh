#!/bin/bash

ckpt_path=$1
base_path=${2:-"none"}
openai_key=${3:-"123"}
gpu_id=${4:-"0"}
data_dir=${5:-"dataset/mmhal"}

base_name="${ckpt_path:11}"
answer_file=${data_dir}/model_output/${base_name}.jsonl
q_file=${data_dir}/mmhal-bench_with_image.jsonl
template_file=${data_dir}/mmhal-bench_answer_template.json
current_time=$(date "+%Y-%m-%d--%H-%M-%S")
log_file=${ckpt_path}/mmhal_scores_${current_time}.log
gpt_model="gpt-4-1106-preview"  # gpt-4, gpt-4-turbo, gpt-4-1106-preview

echo "log_file: "$log_file
echo "ckpt_path: "$ckpt_path
echo "data_dir: "$data_dir
echo "gpu_id: "$gpu_id
echo "answer_file: "$answer_file


CUDA_VISIBLE_DEVICES=$gpu_id \
PYTHONPATH=./:$PYTHONPATH \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python ./muffin/eval/muffin_vqa.py \
    --model-path $ckpt_path \
    --question-file $q_file \
    --answers-file $answer_file \
    --model-base $base_path \
    --temperature 0.0 \
    --conv-mode llava_v1 \
    --num_beams 3

echo "========>Done generating answers<========"


echo "========>Start evaluating answers<========"

PYTHONPATH=./:$PYTHONPATH \
python ./eval/change_mmhal_predict_template.py \
    --response-template $template_file \
    --answers-file $answer_file \
    --save-file $answer_file.template.json

PYTHONPATH=./:$PYTHONPATH \
python eval/eval_gpt_mmhal.py \
    --response $answer_file.template.json \
    --evaluation $answer_file.mmhal_test_eval.json \
    --api-key $openai_key \
    --gpt-model $gpt_model

PYTHONPATH=./:$PYTHONPATH \
python ./eval/merge_mmhal_review_with_predict.py \
    --review_path ${answer_file}.mmhal_test_eval.json \
    --predict_path ${answer_file} \
    --save_path ${answer_file}.mmhal_test_all_infos.json

PYTHONPATH=./:$PYTHONPATH \
python ./eval/summarize_gpt_mmhal_review.py ${answer_file}.mmhal_test_eval.json > ${log_file}

echo Scores are:
cat ${log_file}
echo done