from collections import defaultdict
import json
import argparse
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations
import random

DefaultMethodPairNum = 0
NotFoundPairNum = 0
CorrectPairNum = 0
TotalPairDataId = -1
NeedChangeResponseNum = 0
NoChangeResonseNum = 0

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


def unify_information_by_question(
        repeat_num, image_data, classified_claim_data, claim_reward_data, 
        merged_inf_data, raw_response_data, start_pos, end_pos, wh_type='v1'
    ):
    print("start unify_information_by_question")
    empty_response =  {'raw_claim':[], 'wh_claim':[], 'raw_response':None, 'flag':True}
    empty_data_item = {"response": [], "classification": [], "question": None, "image": None}
    for _ in range(repeat_num):
        empty_data_item['response'].append(deepcopy(empty_response))
    data = {item['question_id']//repeat_num : deepcopy(empty_data_item) for item in merged_inf_data}
    for idx, item in tqdm(enumerate(merged_inf_data), desc="Unify fact"):
        question_id = item['question_id'] // repeat_num
        response_id = item['question_id'] % repeat_num
        assert item['claim_type'] in ['wh_response', 'raw_claim']
        item_key = 'wh_claim' if item['claim_type'] == 'wh_response' else 'raw_claim'
        if wh_type != 'no':
            for fact in item['facts']:
                data[question_id]['response'][response_id][item_key].append(fact)
        else:
            if item_key == 'raw_claim':
                for fact in item['facts']:
                    data[question_id]['response'][response_id]['raw_claim'].append(fact)
                    data[question_id]['response'][response_id]['wh_claim'].append(deepcopy(fact))
    for idx, item in tqdm(enumerate(raw_response_data), desc="Unify response"):
        question_id = item['question_id'] // repeat_num
        response_id = item['question_id'] % repeat_num
        data[question_id]['response'][response_id]['raw_response'] = item['answer']
    for question_id, item in tqdm(enumerate(image_data), desc="Unify image"): 
        if start_pos <= question_id and (end_pos < 0 or question_id < end_pos):
            data[question_id]['question'] = item['question']
            data[question_id]['image'] = item['image']
    for idx, item in tqdm(enumerate(claim_reward_data), desc="Unify reward"): 
        question_id = item['metainfos']['question_id'] // repeat_num
        response_id = item['question_id'] % repeat_num
        fact_id = item['metainfos']['fact_id']
        claim_type = item['metainfos']['claim_type']
        assert claim_type in ['wh_response', 'raw_claim']
        item_key = 'wh_claim' if claim_type == 'wh_response' else 'raw_claim'
        assert type(data[question_id]['response'][response_id][item_key][fact_id]) == str
        new_item = (data[question_id]['response'][response_id][item_key][fact_id], item['answer'], item['scores'])
        if wh_type != 'no':
            data[question_id]['response'][response_id][item_key][fact_id] = new_item
        else:
            if item_key == 'raw_claim':
                data[question_id]['response'][response_id]['raw_claim'][fact_id] = new_item
                data[question_id]['response'][response_id]['wh_claim'][fact_id] = deepcopy(new_item)
    for idx, item in tqdm(enumerate(classified_claim_data), desc="Unify class"): 
        question_id = item[0][0] // repeat_num
        new_item = item
        for i in range(len(item)):
            new_item[i] = (item[i][0] % repeat_num, item[i][1])
        data[question_id]['classification'].append(new_item)
    tmp_data = {}
    for question_id in tqdm(data.keys(), desc="limit st-ed qustion"): 
        if start_pos <= question_id and (end_pos < 0 or question_id < end_pos):
            tmp_data[question_id] = data[question_id]
    data = tmp_data
    false_num = 0
    for question_id in tqdm(data.keys(), desc="Clear bad data"): 
        for response_id in range(repeat_num):
            raw_claim_len = len(data[question_id]['response'][response_id]['raw_claim'])
            wh_claim_len =len(data[question_id]['response'][response_id]['wh_claim'])
            if raw_claim_len != wh_claim_len:
                data[question_id]['response'][response_id]['flag'] = False
                continue
            for fact_id in range(raw_claim_len):
                if type(data[question_id]['response'][response_id]['raw_claim'][fact_id]) == str:
                    data[question_id]['response'][response_id]['flag'] = False
            for fact_id in range(wh_claim_len):
                if type(data[question_id]['response'][response_id]['wh_claim'][fact_id]) == str:
                    data[question_id]['response'][response_id]['flag'] = False
        flag_true_id = []
        for response_id in range(repeat_num): 
            if data[question_id]['response'][response_id]['flag']:
                flag_true_id.append(response_id)
        false_num += repeat_num - len(flag_true_id)
        new_classification_data=deepcopy(data[question_id]['classification'])
        for i in range(len(data[question_id]['classification']))[::-1]:
            new_item = deepcopy(data[question_id]['classification'][i])
            for j in range(len(data[question_id]['classification'][i]))[::-1]:
                if data[question_id]['classification'][i][j][0] not in flag_true_id:
                    del new_item[j]
            new_classification_data[i] = new_item
            if len(new_item) == 0:
                del new_classification_data[i]
        data[question_id]['classification'] = new_classification_data
    print('bad_response_num:', false_num, '   data_question_num:', len(data))

    return data


def unify_information_by_question_v0(repeat_num, image_data, claim_reward_data, start_pos, end_pos):
    print("start unify_information_by_question_v0")
    empty_response =  {"raw_response": None, "score": True}
    empty_data_item = {"response": [], "question": None, "image": None}
    for _ in range(repeat_num):
        empty_data_item['response'].append(deepcopy(empty_response))
    data_keys = []
    for item in claim_reward_data:
        if (item['question_id']//repeat_num) not in data_keys:
            data_keys.append(item['question_id']//repeat_num)
    data = {idx : deepcopy(empty_data_item) for idx in data_keys}
    for question_id, item in tqdm(enumerate(image_data), desc="Unify image"): 
        if start_pos <= question_id and (end_pos < 0 or question_id < end_pos):
            data[question_id]['question'] = item['question']
            data[question_id]['image'] = item['image']
    for idx, item in tqdm(enumerate(claim_reward_data), desc="Unify reward"): 
        question_id = item['metainfos']['question_id'] // repeat_num
        response_id = item['question_id'] % repeat_num
        data[question_id]['response'][response_id]['raw_response'] = item['metainfos']['raw_response']
        data[question_id]['response'][response_id]['score'] = item['scores']
    return data


used_wh_claim_num, used_raw_claim_num = 0, 0
def get_prompt(data, which_half, raw_question):
    if data['change']['type'] == 'rewrite_claim':
        # v1
        prompt = 'Please correct any errors or omissions in the original text, which is used to answer the question, based on the tips and image. Remember to follow the tips first if tips conflict with the image. You need to make minimal modifications to the original text and maintain its style and format. Your output should only include the corrected text. \n\n(Tips): {} \n\n(Question): {} \n\n(Original text): {} \n\n(Corrected text): '
        assert len(data['change']['fact']) >= 1
        tips_prompt = ""
        tips_id = [0] * 40
        for idx, tip in enumerate(data['change']['fact']):
            if tip[2] == 'wh_claim':
                global used_wh_claim_num
                used_wh_claim_num += 1
            tips_id[tip[1][0]] += 1
            tips_prompt = tips_prompt + f'{idx+1}.{tip[0][0]}; '
        global used_raw_claim_num
        used_raw_claim_num += sum(tips_id) - max(tips_id)
        return prompt.format(tips_prompt, raw_question, data['raw_response'])
    
        # # v2
        # if which_half == 'good':
        #     pass
        # else:
        #     pass

    elif data['change']['type'] == 'generate_new':
        prompt = 'Please answer the question based on the tips and image. Remember to follow the tips first if tips conflict with the image. Your output should include as much knowledge as possible from the tips. Your output should only include answer of question. \n\n(Tips): {} \n\n(Question): {} \n\n(Answer): ' # v4.2
        # prompt = 'Please answer the question based on the pictures and tips. \n\n(Tips): {} \n\n(Question): {}' # v2
        tips_prompt = ""
        for idx, tip in enumerate(data['change']['fact']):
            tips_prompt = tips_prompt + f'{idx+1}.{tip[0][0]}; '
        return prompt.format(tips_prompt, raw_question)


def output_data(half_data, which_half, raw_question, pair_id, question_pair_id, question_id, image, output_path, need_merge_path):
    half_data['pair_id'] = pair_id
    half_data['which_half'] = which_half
    half_data['question_pair_id'] = question_pair_id
    half_data['question_id'] = question_id
    if 'change' in half_data.keys():
        # with llava-1.5   path=need_merge_path   {'question','image'}
        global NeedChangeResponseNum
        NeedChangeResponseNum += 1
        file = open(need_merge_path, 'a')
        prompt = get_prompt(half_data, which_half, raw_question)
        half_data['question'] = prompt
        half_data['image'] = image
    else:
        # without llava-1.5   path=output_path
        global NoChangeResonseNum
        NoChangeResonseNum += 1
        file = open(output_path, 'a')
    json.dump(half_data, file)
    file.write('\n')


def check_need_merge(upper_half, lower_half, raw_question, question_pair_id, question_id, question_data, output_path, need_merge_path):
    global TotalPairDataId
    TotalPairDataId += 1
    image = question_data['image']
    output_data(upper_half, 'good', raw_question, TotalPairDataId, question_pair_id, question_id, image, output_path, need_merge_path)
    output_data(lower_half, 'bad', raw_question, TotalPairDataId, question_pair_id, question_id, image, output_path, need_merge_path)


def calc_score_of_relative_accuracy(claim_list):
    total_score = 0
    for claim in claim_list:
        score_yes = claim[2]['yes'] + claim[2]['Yes']
        score_no = claim[2]['no'] + claim[2]['No']
        # total_score = total_score + 1 if score_yes > score_no else total_score - 1
        total_score += score_yes - score_no
    total_score = total_score / len(claim_list)
    return total_score

def calc_score_of_incorrect_num(claim_list):
    total_score = 0
    for claim in claim_list:
        score_yes = claim[2]['yes'] + claim[2]['Yes']
        score_no = claim[2]['no'] + claim[2]['No']
        if score_no > score_yes:
            total_score = total_score - 1
    return total_score


def check_yesno(claim):
    score_yes = claim[2]['yes'] + claim[2]['Yes']
    score_no = claim[2]['no'] + claim[2]['No']
    if score_yes > score_no:
        return True
    return False


def find_classification(item, type_list):
    ans = None
    for i_type in type_list:
        if item in i_type:
            ans = i_type
            break
    if ans is None:
        return None
    output = deepcopy(ans)
    for ans_item in ans:
        if ans_item != item and ans_item[0] == item[0]:
            output.remove(ans_item)
    return output


def preprocess_data(data, calc_score_method="relative_accuracy"):
    responses = []
    
    for idx, response in enumerate(data['response']):
        if not response['flag']:
            continue
        if calc_score_method == "relative_accuracy":
            score = calc_score_of_relative_accuracy(response['raw_claim'])
        elif calc_score_method == "incorrect_num":
            score = calc_score_of_incorrect_num(response['raw_claim'])
        else:
            print(f"[Warning]: Undefined calc_score_method:{calc_score_method}, use default method 'relative_accuracy' to replace this.")
            score = calc_score_of_relative_accuracy(response['raw_claim'])
        item = {
            'score': score,
            'response_id': idx,
            'response': response
        }
        responses.append(item)
    
    if len(responses) < 2:
        global NotFoundPairNum
        NotFoundPairNum += 1
        return None
    
    def responses_key(item):
        return item['score']
    responses = sorted(responses, key=responses_key)[::-1]
    
    return responses


def max_one_claim_method(data):
    responses = preprocess_data(data)
    if responses is None:
        return None, None
    
    for half_pair in responses:
        for fact_id, claim in enumerate(half_pair['response']['raw_claim']):
            same_claims = find_classification((half_pair['response_id'], fact_id), data['classification'])
            yes_claim = None
            no_claim = None
            for id in same_claims:
                for claim_type in ['raw_claim', 'wh_claim']:
                    item = data['response'][id[0]][claim_type][id[1]]
                    if check_yesno(item):
                        yes_claim = (item, id, claim_type)
                    else:
                        no_claim = (item, id, claim_type)
            if yes_claim is not None and no_claim is not None:
                global CorrectPairNum
                CorrectPairNum += 1
                if check_yesno(claim):
                    lower_half = deepcopy(half_pair['response'])
                    lower_half['change'] = {'type':'rewrite_claim', 'fact': [no_claim]}
                    return [half_pair['response']], [lower_half]
                else:
                    upper_half = deepcopy(half_pair['response'])
                    upper_half['change'] = {'type':'rewrite_claim', 'fact': [yes_claim]}
                    return [upper_half], [half_pair['response']]
    
    global DefaultMethodPairNum
    DefaultMethodPairNum += 1
    return [responses[0]['response']], [responses[-1]['response']]


def max_all_claim_method(data):
    responses = preprocess_data(data)
    if responses is None:
        return None, None
    
    val_response_list = []
    for half_pair in responses: 
        val_claim_list = []
        for fact_id, claim in enumerate(half_pair['response']['raw_claim']):
            same_claims = find_classification((half_pair['response_id'], fact_id), data['classification'])
            # same_claims = [(half_pair['response_id'], fact_id)] # ablation study
            yes_claim = None
            no_claim = None
            for id in same_claims:
                # for claim_type in ['raw_claim']: # ablation study
                for claim_type in ['raw_claim', 'wh_claim']:
                    item = data['response'][id[0]][claim_type][id[1]]
                    if check_yesno(item):
                        yes_claim = (item, id, claim_type)
                    else:
                        no_claim = (item, id, claim_type)
            if yes_claim is not None and no_claim is not None:
                val_claim_list.append((yes_claim, no_claim, fact_id))
        val_response_list.append((half_pair, val_claim_list))

    least_val_claim_num = 2
    val_claim_num = 0
    for item in val_response_list:
        if len(item[1]) > least_val_claim_num:
            val_claim_num += len(item[1]) - least_val_claim_num

    if val_claim_num > 0:
        ans_upper_half = None
        ans_lower_half = None
        ans_num_upper = 100
        ans_num_lower = 0
        for item in val_response_list:
            if len(item[1]) > least_val_claim_num:
                half_pair = item[0]
                upper_half = deepcopy(half_pair['response'])
                upper_half['change'] = {'type':'rewrite_claim', 'fact': []}
                upper_half['backup_option'] = responses[0]['response']['raw_response']
                lower_half = deepcopy(half_pair['response'])
                lower_half['change'] = {'type':'rewrite_claim', 'fact': []}
                lower_half['backup_option'] = responses[-1]['response']['raw_response']
                for yesno in item[1]:
                    yes_claim, no_claim, fact_id = yesno[0], yesno[1], yesno[2]
                    if check_yesno(half_pair['response']['raw_claim'][fact_id]):
                        lower_half['change']['fact'].append(
                            (no_claim[0],no_claim[1],no_claim[2],half_pair['response']['raw_claim'][fact_id])
                            # no_claim
                        )
                    else:
                        upper_half['change']['fact'].append(
                            (yes_claim[0],yes_claim[1],yes_claim[2],half_pair['response']['raw_claim'][fact_id])
                            # yes_claim
                        )
                num_upper = len(upper_half['change']['fact'])
                num_lower = len(lower_half['change']['fact'])

                if num_lower == 0:
                    lower_half = deepcopy(half_pair['response'])
                
                # ablation study
                # lower_half = deepcopy(half_pair['response'])

                if num_upper == 0:
                    upper_half = deepcopy(half_pair['response'])
                if ans_num_upper > num_upper or (ans_num_upper == num_upper and ans_num_lower < num_lower):
                    ans_num_upper = num_upper
                    ans_num_lower = num_lower
                    ans_upper_half = upper_half
                    ans_lower_half = lower_half
        global CorrectPairNum
        CorrectPairNum += 1
        return [ans_upper_half], [ans_lower_half]
    
    global DefaultMethodPairNum
    DefaultMethodPairNum += 1
    return [responses[0]['response']], [responses[-1]['response']]


def all2all_method(data):
    responses = preprocess_data(data)
    if responses is None:
        return None, None
    val_claim_num = 0
    for half_pair in responses: 
        val_claim_dict = {}
        for fact_id, claim in enumerate(half_pair['response']['raw_claim']):
            same_claims = find_classification((half_pair['response_id'], fact_id), data['classification'])
            claim_flag = False
            for same_claim in same_claims:
                if same_claim in val_claim_dict.keys():
                    claim_flag = True
                    break
            if claim_flag:
                continue
            yes_claim = None
            no_claim = None
            for id in same_claims:
                for claim_type in ['raw_claim', 'wh_claim']:
                    item = data['response'][id[0]][claim_type][id[1]]
                    if check_yesno(item):
                        yes_claim = (item, id, claim_type)
                    else:
                        no_claim = (item, id, claim_type)
            if yes_claim is not None and no_claim is not None:
                val_claim_num += 1
                val_claim_dict[same_claims[0]] = (yes_claim, no_claim)
    if val_claim_num >= 2:
        global CorrectPairNum
        CorrectPairNum += 1
        upper_half = {'change': {'type':'generate_new', 'fact': []}}
        lower_half = {'change': {'type':'generate_new', 'fact': []}}
        for idx, val_claim in val_claim_dict.items():
            upper_half['change']['fact'].append(val_claim[0])
            lower_half['change']['fact'].append(val_claim[1])
        return [upper_half], [lower_half]

    global DefaultMethodPairNum
    DefaultMethodPairNum += 1
    # return [responses[0]['response']], [responses[-1]['response']]
    return None, None

def default_v0_simple_method(data):
    def check_yesno_v0(score):
        score_yes = score['yes'] + score['Yes']
        score_no = score['no'] + score['No']
        if score_yes > score_no:
            return True
        return False
    upper_half, lower_half = None, None
    for response in data['response']:
        if check_yesno_v0(response['score']):
            upper_half = {'raw_response': response['raw_response']}
        else:
            lower_half = {'raw_response': response['raw_response']}
    if upper_half is None or lower_half is None:
        global NotFoundPairNum
        NotFoundPairNum += 1
        return None, None
    
    global DefaultMethodPairNum
    DefaultMethodPairNum += 1
    return [upper_half], [lower_half]

def default_v1_simple_method(data):
    responses = preprocess_data(data, "incorrect_num")
    if responses is None:
        return None, None
    
    diff = 1
    pair_candidate = []
    for i, j in combinations(range(len(responses)), 2):
        if responses[i]['score'] - responses[j]['score'] >= diff:
            pair_candidate.append((i, j))
    if len(pair_candidate) == 0:
        global DefaultMethodPairNum
        DefaultMethodPairNum += 1
        return None, None
        responses = preprocess_data(data)
        assert responses is not None
        return [responses[0]['response']], [responses[-1]['response']]

    global CorrectPairNum
    CorrectPairNum += 1

    sample_num = 1
    if len(pair_candidate) >= sample_num:
        sampled_pair = random.sample(pair_candidate, sample_num)
    else:
        sampled_pair = pair_candidate
    upper_half_list = []
    lower_half_list = []
    for item in sampled_pair:
        upper_half_list.append(responses[item[0]]['response'])
        lower_half_list.append(responses[item[1]]['response'])
    return upper_half_list, lower_half_list

def default_v2_simple_method(data):
    responses = preprocess_data(data)
    if responses is None:
        return None, None
    
    global DefaultMethodPairNum
    DefaultMethodPairNum += 1
    return [responses[0]['response']], [responses[-1]['response']]


dpo_pair_generate_func = {
    'default_v0': default_v0_simple_method,
    'default_v1': default_v1_simple_method,
    'default_v2': default_v2_simple_method,
    'max_one_claim': max_one_claim_method,
    'max_all_claim': max_all_claim_method,
    'all2all': all2all_method,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--classified_claim_path', type=str)
    parser.add_argument('--claim_reward_path', type=str)
    parser.add_argument('--merged_inf_path', type=str)
    parser.add_argument('--raw_response_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--need_merge_path', type=str)
    parser.add_argument('--dpo_pair_generate_method', type=str)
    parser.add_argument('--repeat_num', type=int)
    parser.add_argument('--start_pos', type=int)
    parser.add_argument('--end_pos', type=int)
    parser.add_argument('--wh_type', type=str)
    args = parser.parse_args()
    
    assert args.dpo_pair_generate_method in dpo_pair_generate_func.keys()
    construct_method = dpo_pair_generate_func[args.dpo_pair_generate_method]
    image_data = input_data(args.image_path) # [{'image': ...}, {'image': ...}, ...]
    claim_reward_data = input_data(args.claim_reward_path)

    if args.dpo_pair_generate_method == 'default_v0':
        data = unify_information_by_question_v0(
            repeat_num=args.repeat_num,
            image_data=image_data,
            claim_reward_data=claim_reward_data,
            start_pos=args.start_pos,
            end_pos=args.end_pos
        )
    else:
        classified_claim_data = input_data(args.classified_claim_path)
        merged_inf_data = input_data(args.merged_inf_path)
        raw_response_data = input_data(args.raw_response_path)
        data = unify_information_by_question(
            repeat_num=args.repeat_num,
            image_data=image_data,
            classified_claim_data=classified_claim_data,
            claim_reward_data=claim_reward_data,
            merged_inf_data=merged_inf_data,
            raw_response_data=raw_response_data,
            start_pos=args.start_pos,
            end_pos=args.end_pos,
            wh_type=args.wh_type
        )

    with open(args.output_path, 'w') as f:
        pass
    with open(args.need_merge_path, 'w') as f:
        pass
    for question_id, item in tqdm(data.items(), desc="Generate data"):
        upper_half, lower_half = construct_method(item)
        if upper_half is None or lower_half is None:
            continue
        for i in range(len(upper_half)):
            check_need_merge(upper_half[i], lower_half[i], item['question'], i, question_id, item, args.output_path, args.need_merge_path)
    
    print(f'\nused_wh_claim_num={used_wh_claim_num} used_raw_claim_num={used_raw_claim_num}')
    print(f'CorrectPairNum={CorrectPairNum}')
    print(f'DefaultMethodPairNum={DefaultMethodPairNum}')
    print(f'NotFoundPairNum={NotFoundPairNum}\n')
    print(f'NeedChangeResponseNum={NeedChangeResponseNum}')
    print(f'NoChangeResonseNum={NoChangeResonseNum}\n')