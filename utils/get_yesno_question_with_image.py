from collections import defaultdict
import json
import argparse
from tqdm import tqdm
import networkx as nx
import os


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


def prompt_template_v0(ques, response):
    prompt = "You are an expert at determining whether the response to the question matches the image. Answer 'Yes' if the image matches the response, and 'No' if there are illusions or inconsistencies between the image and the response. \n (question): {} \n (response): {} \n Please answer Yes or No."
    return prompt.format(ques, response)


def prompt_template(text):
    prompt = "{} Please answer yes or no."
    return prompt.format(text)


def get_image_path(image_dir, idx, png_files):
    if f'{idx}.png' in png_files:
        image_path = os.path.join(image_dir, f'{idx}.png')
    else:
        image_path = os.path.join(image_dir, f'{idx}.jpg')
    assert os.path.exists(image_path), f'{image_path} is Not Exists!'
    return image_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--raw_ques_path', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--repeat_num', type=int)
    parser.add_argument('--version', default='v1', type=str)
    args = parser.parse_args()

    data = input_data(args.data_path)
    raw_ques = input_data(args.raw_ques_path)
    image_dir = args.image_dir
    png_files = []
    for image_file in os.listdir(image_dir):
        if '.png' in image_file:
            png_files.append(image_file)
    
    if args.version == 'v0':
        
        f = open(args.output_path, 'w')
        for line_id, line in tqdm(enumerate(data)):
            question = prompt_template_v0(line['raw_question'], line['answer'])
            image_path = get_image_path(image_dir, line['question_id']//args.repeat_num, png_files)
            question_item = {
                'question_id': line['question_id'],
                'claim_type': line['claim_type'],
                'question': question,
                'raw_response': line['raw_question'],
                'image_path': image_path,
            }
            json.dump(question_item, f)
            f.write('\n')
        
        exit()
    
    question_num = {'wh_response': 0, 'raw_claim': 0}
    error_data_id = []
    f = open(args.output_path, 'w')
    for line_id, line in tqdm(enumerate(data)):
        if len(line['yesno_question']) == len(line['facts']):
            for fact_id, yesno_question in enumerate(line['yesno_question']):
                question = prompt_template(yesno_question)
                question_num[line['claim_type']] += 1
                image_path = get_image_path(image_dir, line['question_id']//args.repeat_num, png_files)
                question_item = {
                    'question_id': line['question_id'],
                    'fact_id': fact_id,
                    'claim_type': line['claim_type'],
                    'question': question,
                    # 'image': raw_ques[line['question_id']//args.repeat_num]['image'],
                    'image_path': image_path,
                }
                json.dump(question_item, f)
                f.write('\n')
        else:
            error_data_id.append((line['question_id'], line['claim_type']))
    f.close()

    if len(error_data_id) > 0:
        print(f"[Warning] Question_id for ' len(line['yesno_question']) != len(line['facts']) ':{error_data_id}")
    print(f"question_num: {question_num} error_question_id_num: {len(error_data_id)}")