import json
import base64
import ast
import io
import PIL
from datasets import load_dataset
from tqdm import tqdm 
import argparse


def get_all_data_format_jsonl(input_data, output_file):
    print("start")
    ds = load_dataset(input_data)
    print("load end.")
    filename = output_file
    data_dict = {}
    error_data_num = 0
    correct_data_num = 0
    with open(filename, 'w') as f:
        for index, example in tqdm(enumerate(ds["train"])):
            if (example['image_path'], example["question"]) in data_dict.keys():
                error_data_num += 1
                continue
            correct_data_num += 1
            data_dict[(example['image_path'], example["question"])] = 1
            image_data = io.BytesIO()
            if type(example["image"]) == PIL.PngImagePlugin.PngImageFile:
                example["image"].save(image_data,format="PNG")
            else:
                try:
                    example["image"].save(image_data,format="JPEG")
                except:
                    try:
                        example["image"].save(image_data,format="WEBP")
                    except:
                        import ipdb; ipdb.set_trace()
                        print(example)
            image_data_bytes = image_data.getvalue()
            encoded_image = base64.b64encode(image_data_bytes).decode('utf-8')
            # item = {"image":encoded_image, "question":ast.literal_eval(example["text"])["question"]}
            item = {
                "image":encoded_image, 
                "question":example["question"],
                "ds_name":example["ds_name"],
                "origin_dataset":example["origin_dataset"],
                "origin_split":example["origin_split"],
                "idx":example["idx"],
                "image_path":example["image_path"]
            }
            json.dump(item, f)
            f.write('\n')
        f.close()
    print("correct_data_num=", correct_data_num, "duplicate_data_num=", error_data_num)
    

if __name__ ==  '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_folder', default="dataset/openbmb___rlaif-v-dataset", type=str)
    parser.add_argument('--output_data_file_path', default="dataset/rlaif-v-question-with-image/question.jsonl", type=str)
    parser.add_argument('--output_image_file_dir', default="dataset/rlaif-v-image-dir", type=str)
    args = parser.parse_args()
    
    input_data = str(args.input_data_folder)
    output_file = str(args.output_data_file_path)
    image_dir = str(args.output_image_file_dir)

    get_all_data_format_jsonl(input_data, output_file)