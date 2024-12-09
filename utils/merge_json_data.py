import json
import argparse
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


Warning1_num = 0

def yesno_norm_response(raw_text):
    global Warning1_num
    text = raw_text.lower()
    if len(text) < 8:
        if "yes" in text:
            return "Yes"
        elif "no" in text:
            return "No" 
        else: 
            Warning1_num += 1
            return "Yes"
    if "yes" in text[:5]:
        return "Yes"
    elif "no" in text[:5]:
        return "No"
    elif "yes" in text[-5:]:
        Warning1_num += 1
        return "Yes"
    elif "no" in text[-5:]:
        Warning1_num += 1
        return "No"
    elif "yes" in text:
        Warning1_num += 1
        return "Yes"
    elif "no" in text:
        Warning1_num += 1
        return "No" 
    Warning1_num += 1
    print(f"[Warning]Unrecognized response: {raw_text}")
    return raw_text


def abcd_norm_response(raw_text):
    text = raw_text.lower()
    for i in range(len(text)):
        if text[i] in ['a', 'b', 'c', 'd']:
            return text[i].upper()
    print(f"[Warning]Unrecognized response: {raw_text}, will be recognized as A")
    return 'A'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int)
    parser.add_argument("--answers-file", type=str, default="answer.json")
    parser.add_argument("--answers-type", type=str, default="yesno")
    parser.add_argument("--answers-key", type=str, default="response")
    args = parser.parse_args()
    
    num = args.num
    data = []
    answers_file = args.answers_file
    for i in tqdm(range(num), desc="merge_json"):
        data_path = answers_file.replace(".json", f"{i}.json")
        data_i = input_data(data_path)
        for item in data_i:
            if args.answers_type == 'yesno':
                item[args.answers_key] = yesno_norm_response(item[args.answers_key])
            elif args.answers_type == 'none':
                pass
            else:
                item[args.answers_key] = abcd_norm_response(item[args.answers_key])
            data.append(item)
    
    print("Warning1_num:", Warning1_num)

    with open(answers_file, "w") as f:
        json.dump(data, f)