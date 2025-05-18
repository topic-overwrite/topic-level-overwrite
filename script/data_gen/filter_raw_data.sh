#!/bin/bash

echo "----------Start data filter----------"

input_data_folder=$1
output_data_file_path=${2:-"dataset/raw-question-with-image/question.jsonl"}
output_image_file_dir=${3:-"dataset/raw-image-dir"}



python utils/get_raw_data.py \
    --input_data_folder $input_data_folder \
    --output_data_file_path $output_data_file_path \
    --output_image_file_dir $output_image_file_dir 
