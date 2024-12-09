###===> install dependencies
export TORCH_DISTRIBUTED_DEBUG=DETAIL
###<===

ckpt_path=$1
base_path=${2:-"none"}
openai_key=${3:-"123"}
gpu_id=${4:-"1"}
data_dir=${5:-"dataset/objhal"}

base_name="${ckpt_path:11}"
answer_file=${data_dir}/model_output/${base_name}.jsonl
q_file=dataset/objhal/obj_halbench_300_with_image.jsonl # ${data_dir}/obj_halbench_300_with_image.jsonl
synonyms_file=dataset/objhal/synonyms_refine.txt # ${data_dir}/synonyms_refine.txt
current_time=$(date "+%Y-%m-%d--%H-%M-%S")
log_file=${ckpt_path}/objhal_scores_${current_time}.log
gpt_model="gpt-3.5-turbo-0125"

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
    --num_beams 3

echo "========>Done generating answers<========"

echo "========>Start evaluating answers<========"

PYTHONPATH=./:$PYTHONPATH \
python ./eval/eval_gpt_obj_halbench.py \
    --coco_path ${data_dir}/annotations \
    --synonyms_file ${synonyms_file} \
    --cap_folder ${data_dir}/model_output \
    --cap_type ${base_name}.jsonl \
    --org_folder $q_file \
    --use_gpt \
    --openai_key $openai_key \
    --gpt-model $gpt_model

PYTHONPATH=./:$PYTHONPATH \
python ./eval/summarize_gpt_obj_halbench_review.py ${data_dir}/model_output $base_name > ${log_file}

# Print Log
echo Scores are:
cat ${log_file}
echo done