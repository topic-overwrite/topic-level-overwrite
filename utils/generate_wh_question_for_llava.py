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


def prompt_template(text, exmple, wh_type):
    if wh_type == 'v1' or wh_type == 'no':
        prompt = "Question:{}\n\nPlease use a short sentence to indicate the answer, For example: {}\n"
        return prompt.format(text, exmple)
    else:
        assert wh_type == 'v3'
        prompt = "You are an expert in extracting facts from the given image. Your task is to answer the question based on the facts in the given image. You should output your result using a complete declarative sentence without opinions or    subjective statements, containing a subject and predicate. (Question):{} \n (Your Result):"
        return prompt.format(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--raw_ques_path', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--repeat_num', type=int)
    parser.add_argument('--wh_type', type=str)
    args = parser.parse_args()

    data = input_data(args.data_path)
    raw_ques = input_data(args.raw_ques_path)
    image_dir = args.image_dir
    png_files = []
    for image_file in os.listdir(image_dir):
        if '.png' in image_file:
            png_files.append(image_file)

    error_data_id = []
    question_num = 0
    f = open(args.output_path, 'w')
    for line_id, line in tqdm(enumerate(data)):
        if len(line['facts']) == len(line['what_question']):
            for idx, what_question in enumerate(line['what_question']):
                question_num += 1
                new_line = {}
                new_line['question_id'] = line['question_id']
                new_line['fact_id'] = idx
                new_line['question'] = prompt_template(what_question, line['facts'][idx], args.wh_type)
                if f"{line['question_id']//args.repeat_num}.png" in png_files:
                    image_path = os.path.join(image_dir, f"{line['question_id']//args.repeat_num}.png")
                else:
                    image_path = os.path.join(image_dir, f"{line['question_id']//args.repeat_num}.jpg")
                assert os.path.exists(image_path), f'{image_path} is Not Exists!'
                assert raw_ques[line['question_id']//args.repeat_num]['image_path'] == line['metainfos']['image_path']
                #new_line['image'] = raw_ques[line['question_id']//args.repeat_num]['image']
                new_line['image_path'] = image_path
                json.dump(new_line, f)
                f.write('\n')
        else:
            error_data_id.append(line['question_id'])
    f.close()

    if len(error_data_id) > 0:
        print(f"[Warning] Question_id for ' len(data['facts']) != len(data['what_question']) ':{error_data_id}")
    print(f"wh_question_num: {question_num} error_question_id_num: {len(error_data_id)}")