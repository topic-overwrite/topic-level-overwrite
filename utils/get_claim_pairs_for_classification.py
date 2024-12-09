import json
import argparse
from itertools import combinations
from tqdm import tqdm

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

def generate_claim_pairs(data, question_id, output_path1):
    ques_num = 0
    data_len = len(data)
    ans_f = open(output_path1, 'a')
    for i, j in combinations(range(data_len), 2):
        claim_i_len = len(data[i]['facts'])
        claim_j_len = len(data[j]['facts'])
        if claim_i_len != len(data[i]['what_question']):
            continue
        if claim_j_len != len(data[j]['what_question']):
            continue
        for claim_i in range(claim_i_len):
            for claim_j in range(claim_j_len):
                ques_num += 1
                item = {
                    'question_id': question_id,
                    'claim_id': [(data[i]['question_id'], claim_i), (data[j]['question_id'], claim_j)],
                    'options': [data[i]['facts'][claim_i], data[j]['facts'][claim_j]],
                    'response': [data[i]['answer'], data[j]['answer']],
                    'wh_question': [data[i]['what_question'][claim_i], data[j]['what_question'][claim_j]],
                }
                ans_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                ans_f.flush()
    ans_f.close()
    return ques_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--repeat_num', type=int)
    args = parser.parse_args()

    data = input_data(args.data_path)
    question_num = len(data) // args.repeat_num
    with open(args.output_path, 'w') as file:
        pass

    ques_num = 0
    for i in tqdm(range(question_num)):
        item = data[i*args.repeat_num: i*args.repeat_num+args.repeat_num]
        ques_num += generate_claim_pairs(item, i, args.output_path)
    print(f"Classification questions generated ready, total_question_num={ques_num}")


if __name__ == "__main__":
    main()