#!/bin/bash

echo "----------Start classify_claim----------"

num_gpus=$1
ckpt=$2
clip_ckpt=$3
wh_question_path=$4
image_path=$5
ans_dir=$6
ans_file=$7
batch_size=$8
image_threshold=$9
repeat_num=${10:-"10"}
use_image_classify=${11:-"no"}
cluster_type=${12:-"tarjan"}
start_pos=${13:-"0"}
end_pos=${14:-"-1"}


echo "num_gpus: "$num_gpus
echo "ckpt: "$ckpt
echo "clip_ckpt: "$clip_ckpt
echo "wh_question_path: "$wh_question_path
echo "image_path: "$image_path
echo "answer dir: "$ans_dir
echo "answer file: "$ans_file
echo "batch_size: "$batch_size
echo "image_threshold: "$image_threshold
echo "repeat_num: "$repeat_num
echo "use_image_classify: "$use_image_classify
echo "start_pos: "$start_pos"  end_pos: "$end_pos


srun \
    -p AI4Good_S \
    -J data-preprocess \
    --kill-on-bad-exit \
python utils/get_claim_pairs_for_classification.py \
    --data_path $wh_question_path \
    --output_path ${ans_dir}/classify_question.jsonl \
    --repeat_num $repeat_num


# array=("0" "1")
# for i in "${array[@]}"; do
for i in $(seq 0 $((num_gpus-1))); do
    echo "===> chuck_num "$i
    sleep 1
    CUDA_VISIBLE_DEVICES=$i \
    PYTHONPATH=./:$PYTHONPATH \
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python utils/llama3_inference.py \
        --checkpoint $ckpt \
        --path ${ans_dir}/classify_question.jsonl \
        --output_dir ${ans_dir} \
        --chunk-num $num_gpus \
        --chunk-idx $i \
        --bs $batch_size \
        --prompt_type classify_claim \
        --start ${start_pos} \
        --end ${end_pos} &
done
wait


output_file=${ans_dir}/claim_relation_with_claim.jsonl
> "$output_file"
for IDX in $(seq 0 $((num_gpus-1))); do
    echo "merge "${ans_dir}/classify_question.s${start_pos}-e${end_pos}.chunk${num_gpus}-${IDX}.jsonl
    cat ${ans_dir}/classify_question.s${start_pos}-e${end_pos}.chunk${num_gpus}-${IDX}.jsonl >> "$output_file"
done

sleep 10

# array=("0" "1")
# for i in "${array[@]}"; do
for i in $(seq 0 $((num_gpus-1))); do
    echo "===> chuck_num "$i
    sleep 1
    CUDA_VISIBLE_DEVICES=$i \
    PYTHONPATH=./:$PYTHONPATH \
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python utils/llama3_inference.py \
        --checkpoint $ckpt \
        --path ${ans_dir}/classify_question.jsonl \
        --output_dir ${ans_dir} \
        --chunk-num $num_gpus \
        --chunk-idx $i \
        --bs $batch_size \
        --prompt_type classify_wh \
        --start ${start_pos} \
        --end ${end_pos} &
done
wait


output_file=${ans_dir}/claim_relation_with_what.jsonl
> "$output_file"
for IDX in $(seq 0 $((num_gpus-1))); do
    echo "merge "${ans_dir}/classify_question.s${start_pos}-e${end_pos}.chunk${num_gpus}-${IDX}.jsonl
    cat ${ans_dir}/classify_question.s${start_pos}-e${end_pos}.chunk${num_gpus}-${IDX}.jsonl >> "$output_file"
done

sleep 10

if [ "$use_image_classify" != "no" ]; then
    echo "[with image]"

    # array=("0" "1")
    # for i in "${array[@]}"; do
    for i in $(seq 0 $((num_gpus-1))); do
        echo "===> chuck_num "$i
        sleep 1
        CUDA_VISIBLE_DEVICES=$i \
        PYTHONPATH=./:$PYTHONPATH \
        HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python utils/image_similarity.py \
            --checkpoint $clip_ckpt \
            --path $wh_question_path \
            --image_path $image_path \
            --output_dir ${ans_dir} \
            --output_name classify_question_image.jsonl \
            --chunk-num $num_gpus \
            --chunk-idx $i \
            --repeat_num $repeat_num \
            --threshold $image_threshold \
            --start ${start_pos} \
            --end ${end_pos} &
    done
    wait

    output_file=${ans_dir}/claim_relation_with_image.jsonl
    > "$output_file"
    for IDX in $(seq 0 $((num_gpus-1))); do
        echo "merge "${ans_dir}/classify_question_image.s${start_pos}-e${end_pos}.chunk${num_gpus}-${IDX}.jsonl
        cat ${ans_dir}/classify_question_image.s${start_pos}-e${end_pos}.chunk${num_gpus}-${IDX}.jsonl >> "$output_file"
    done

    python utils/tarjan_biconnected_graph.py \
        --data_path1 ${ans_dir}/claim_relation_with_claim.jsonl \
        --data_path2 ${ans_dir}/claim_relation_with_what.jsonl \
        --output_path ${ans_dir}/${ans_file} \
        --data_path3 ${ans_dir}/claim_relation_with_image.jsonl \
        --threshold $image_threshold \
        --cluster_type $cluster_type
else
    echo "[without image]"

    python utils/tarjan_biconnected_graph.py \
        --data_path1 ${ans_dir}/claim_relation_with_claim.jsonl \
        --data_path2 ${ans_dir}/claim_relation_with_what.jsonl \
        --output_path ${ans_dir}/${ans_file} \
        --cluster_type $cluster_type
fi
