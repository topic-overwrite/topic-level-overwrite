from collections import defaultdict
import json
import argparse
from tqdm import tqdm
from copy import deepcopy
import base64
import pandas as pd


def input_data(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except:
        data = []
        for line in open(path, 'r'):
            data.append(json.loads(line))
    if type(data) == dict:
        data = [data]
    return data


backup_default_num = 0


def generate_all_dpo_pairs(data1, data2, image_data, prompt_method):
    '''
        {
            "pair_id": int,
            "which_half": str = optional["good", "bad"],
            "raw_response": str,
            "question_id": int = image_id
        }
    '''
    pair_num = -1
    for item in data1:
        pair_num = max(pair_num, item['pair_id'])
    for item in data2:
        pair_num = max(pair_num, item['metainfos']['pair_id'])
    data = []
    pair_num += 1
    empty_item = {"ds_name": None, "question": None, "chosen": None, "rejected": None, "origin_dataset": None, "origin_split": None, "idx": None, "image_path": None, "image": None}
    for _ in range(pair_num):
        data.append(deepcopy(empty_item))
    print(f"pair_num={pair_num}")
    for item in tqdm(data1, desc="without_merge_data"):
        if item['which_half'] == 'good':
            if data[item['pair_id']]['chosen'] is not None:
                print(f"[Error]: {data[item['pair_id']]['chosen']} is not None, but have {item}")
                exit()
            data[item['pair_id']]['chosen'] = item['raw_response']
        else:
            if data[item['pair_id']]['rejected'] is not None:
                print(f"[Error]: {data[item['pair_id']]['rejected']} is not None, but have {item}")
                exit()
            data[item['pair_id']]['rejected'] = item['raw_response']
        if data[item['pair_id']]['image'] is None:
            for item_key in ["image", "question", "ds_name", "origin_dataset", "origin_split", "idx", "image_path"]:
                data[item['pair_id']][item_key] = image_data[item['question_id']][item_key]
            data[item['pair_id']]['image'] = base64.b64decode(data[item['pair_id']]['image'])
            # data[item['pair_id']]['question'] = prompt_method(data[item['pair_id']]['question']) # default
    
    def postprocess(text, metainfos):
        if ('(Corrected answer)' in text 
            or '(Original answer)' in text 
            or '(Question)' in text 
            or '(Tips)' in text 
            or len(text) < 3 
            or len(text) * 4 < len(metainfos['raw_response']) 
            or len(metainfos['raw_response']) * 3 < len(text)):
            global backup_default_num
            backup_default_num += 1
            if 'backup_option' in metainfos.keys():
                return metainfos['backup_option']
            else:
                return metainfos['raw_response']
        return text
    
    for item in tqdm(data2, desc="merged_data"):
        if item['metainfos']['which_half'] == 'good':
            if data[item['metainfos']['pair_id']]['chosen'] is not None:
                print(f"[Error]: {data[item['metainfos']['pair_id']]['chosen']} is not None, but have {item}")
                exit()
            data[item['metainfos']['pair_id']]['chosen'] = postprocess(item['answer'], item['metainfos'])
        else:
            if data[item['metainfos']['pair_id']]['rejected'] is not None:
                print(f"[Error]: {data[item['metainfos']['pair_id']]['rejected']} is not None, but have {item}")
                exit()
            data[item['metainfos']['pair_id']]['rejected'] = postprocess(item['answer'], item['metainfos'])
        if data[item['metainfos']['pair_id']]['image'] is None:
            for item_key in ["image", "question", "ds_name", "origin_dataset", "origin_split", "idx", "image_path"]:
                data[item['metainfos']['pair_id']][item_key] = image_data[item['metainfos']['question_id']][item_key]
            data[item['metainfos']['pair_id']]['image'] = base64.b64decode(data[item['metainfos']['pair_id']]['image'])
            data[item['metainfos']['pair_id']]['question'] = prompt_method(data[item['metainfos']['pair_id']]['question'])
    
    for idx, item in enumerate(data):
        if item['chosen'] is None or item['rejected'] is None or item['image'] is None:
            print(f"[Error]: data[{idx}]={item} is None")
            exit()
    return data

def get_default_prompt(text):
    return text

def get_all2all_prompt(text):
    return 'Please answer the question based on the picture. \n\n(Question): {}'.format(text)

prompt_dict = {
    "default": get_default_prompt,
    "all2all": get_all2all_prompt,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--without_merge_path', type=str)
    parser.add_argument('--merged_response_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--prompt_type', type=str, default="default")
    args = parser.parse_args()

    assert args.prompt_type in prompt_dict.keys()
    prompt_method = prompt_dict[args.prompt_type]
    image_data = input_data(args.image_path)
    without_merge_data = input_data(args.without_merge_path)
    merged_response_data = input_data(args.merged_response_path)

    data = generate_all_dpo_pairs(without_merge_data, merged_response_data, image_data, prompt_method)
    print("backup_default_num: ", backup_default_num)

    df = pd.DataFrame(data)
    df.to_parquet(args.output_path)