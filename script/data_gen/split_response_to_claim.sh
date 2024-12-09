#!/bin/bash

echo "----------Start split_response----------"

num_gpus=$1
ckpt=$2
respnse_dir=$3
respnse_filename=$4
output_claim_dir=$5
output_claim_file=$6
batch_size=$7
start_pos=${8:-"0"}
end_pos=${9:-"-1"}

respnse_path=${respnse_dir}/${respnse_filename}

echo "num_gpus: "$num_gpus
echo "ckpt: "$ckpt
echo "respnse_path: "$respnse_path
echo "output_claim_dir: "$output_claim_dir
echo "output_claim_file: "$output_claim_file
echo "batch_size: "$batch_size
echo "start_pos: "$start_pos"  end_pos: "$end_pos

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
        --path ${respnse_path} \
        --output_dir ${output_claim_dir} \
        --chunk-num $num_gpus \
        --chunk-idx $i \
        --bs $batch_size \
        --prompt_type split \
        --start ${start_pos} \
        --end ${end_pos} &
done
wait

respnse_filename="${respnse_filename::-5}"

output_file=${output_claim_dir}/${output_claim_file}
> "$output_file"
for IDX in $(seq 0 $((num_gpus-1))); do
    echo "merge "${output_claim_dir}/${respnse_filename}.s${start_pos}-e${end_pos}.chunk${num_gpus}-${IDX}.jsonl
    cat ${output_claim_dir}/${respnse_filename}.s${start_pos}-e${end_pos}.chunk${num_gpus}-${IDX}.jsonl >> "$output_file"
done