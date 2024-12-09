from collections import defaultdict
import json
import argparse
from tqdm import tqdm
import networkx as nx


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--claim_path', type=str)
    parser.add_argument('--wh_response_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--only_wh_question', type=str, default='no')
    args = parser.parse_args()

    assert args.only_wh_question in ['yes', 'no']
    claim_data = input_data(args.claim_path)
    wh_response_data = input_data(args.wh_response_path)
    question_num = 0
    f = open(args.output_path, 'w')

    new_line = {'question_id': None, 'claim_type': None, 'facts': []}
    for line_id, line in tqdm(enumerate(wh_response_data), desc='wh_response_data'):
        if line['metainfos']['question_id'] != new_line['question_id']:
            if len(new_line['facts']) > 0:
                json.dump(new_line, f)
                f.write('\n')
            question_num += 1
            new_line = {
                'question_id': line['metainfos']['question_id'], 
                'claim_type': 'wh_response', 
                'facts': []
            }
        new_line['facts'].append(line['answer'])
    if len(new_line['facts']) > 0:
        json.dump(new_line, f)
        f.write('\n')
    
    if args.only_wh_question != 'yes':
        print("doing raw_claim yes_no question generate")
        for line_id, line in tqdm(enumerate(claim_data), desc='claim_data'):
            question_num += 1
            new_line = {}
            new_line['question_id'] = line['question_id']
            new_line['claim_type'] = 'raw_claim'
            new_line['facts'] = line['facts']
            json.dump(new_line, f)
            f.write('\n')
        
    f.close()
    print(f"yesno_question_num: {question_num}")