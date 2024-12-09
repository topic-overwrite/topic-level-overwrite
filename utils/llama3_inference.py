import os
import math
import copy
import json
import torch
import argparse
import transformers
from tqdm import tqdm
import time

import torch.utils.data as torch_data


class GenDataset(torch_data.Dataset):
    def __init__(self, data, wrap_func, pipline):
        super().__init__()
        self.data = data
        self.wrap_func = wrap_func
        self.pipeline = pipline

    def __getitem__(self, index):
        item = self.data[index]
        messages = self.wrap_func(item)

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return {
            "batch_data": item,
            "prompt": prompt
        }

    def __len__(self):
        return len(self.data)

def data_collate_fn(data_list):
    batch_data = [x['batch_data'] for x in data_list]
    prompts = [x['prompt'] for x in data_list]

    data = {
        'batch_data': batch_data,
        'prompt': prompts,
    }

    return data

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def read_jsonlines(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data
def write_jsonlines(path, data):
    with open(path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

def get_facts(result):
    result = result.strip().split('\n')

    fact_list = []
    for item in result:
        if item == '':
            continue
        if '###' in item:
            continue

        item = item[1:].strip()
        fact_list.append(item)

    # print(fact_list)
    return fact_list

def init_pipline(ckpt):
    model_id = ckpt
    tokenizer = (model_id, {'padding_side': 'left'})
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        tokenizer=tokenizer
    )
    return tokenizer, pipeline

def batch_inference(
        path, ans_file, tokenizer, pipeline, key, wrap_func, 
        batch_size=8, chunk_num=1, chunk_idx=0, max_tokens=140, 
        start=0, end=-1, do_get_facts=True
    ):
    # load data
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except:
        data = []
        for line in open(path, 'r'):
            data.append(json.loads(line))

    if type(data) == dict:
        data = [data]

    print(f"start={start}, end={end}")
    if end > 0:
        end = min(end, len(data))
    elif end == -1:
        end = len(data)

    data = data[start:end]

    # get current chunk data
    data = get_chunk(data, chunk_num, chunk_idx)
    # load prev inference results
    # if os.path.exists(ans_file):
    #     prev_ans = []
    #     with open(ans_file, 'r') as f:
    #         for line in f.readlines():
    #             temp_ans = json.loads(line)
    #             prev_ans.append(temp_ans)

    #     ans_f = open(ans_file, 'a')
    #     data = data[len(prev_ans):]
    # else:
    prev_ans = []
    os.makedirs(os.path.dirname(ans_file), exist_ok=True)
    ans_f = open(ans_file, 'w')
    time.sleep(10)
    # get batch inputs
    dataset = GenDataset(data, wrap_func, pipeline)
    print(f'Dataset size is {len(dataset)}')
    print(f'Dataset batch size is {batch_size}')

    dataloader = torch_data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=5,
        pin_memory=True,
        drop_last=False,
        collate_fn=data_collate_fn,
    )
    print(f'Dataloader size is {len(dataloader)}')

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id

    # inference
    all_outputs = copy.deepcopy(prev_ans)
    dataloader_len = len(dataloader)
    for i, batch_list in tqdm(enumerate(dataloader), desc=f"[{chunk_idx}chunk_idx] total={dataloader_len} "):
        outputs = pipeline(
            batch_list['prompt'],
            eos_token_id=terminators,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.2,
            # num_beams=3,
            top_p=0.9,
            batch_size=batch_size,
        )

        for item, prompt, output in zip(batch_list['batch_data'], batch_list['prompt'], outputs):
            resp = output[0]["generated_text"][len(prompt):]

            if do_get_facts:
                item[f'raw_{key}'] = resp
                item[key] = get_facts(resp)
            else:
                item[key] = resp

            all_outputs.append(item)

            if (not do_get_facts) and ("relation" in item.keys()):
                # classify_claim classify_wh   to save memory
                ans_f.write(json.dumps({"claim_id": item["claim_id"], "relation": item["relation"]}, ensure_ascii=False) + '\n')
            else:
                ans_f.write(json.dumps(item, ensure_ascii=False) + '\n')
            ans_f.flush()

    ans_f.close()

    return all_outputs

def wrap_prompt_divide_to_list(item):
    if 'raw_question' in item.keys():
        question = item['raw_question']
    elif 'prompt' in item.keys():
        question = item['prompt']
    else:
        question = item['question']
    answer = item['answer'] if 'answer' in item.keys() else item['text']

    content="You are an expert in extracting facts from the given question-answer pair for an image. Your task is to extract and rewrite the facts mentioned in the question-answer pair into self-contained sentences. Exclude opinions or subjective statements.\n\nYou should present your result in the following format:\n### Facts:\n- {Extracted fact 1}\n- {Extracted fact 2}\n- ...\n\n### Question-answer pair:\nQuestion: " + question + "\nAnswer: " + answer
    temp_input = ' '.join(content.split(' ')[:300])

    messages = [{"role": "user", "content": temp_input},]
    return messages

def wrap_prompt_yesno_question_to_list(item):
    facts=item["facts"]
    content="You are an expert at modifying a given declarative sentence into a general question sentence. Your task is to modify the given declarative sentences one by one into a general question form. Do not change tenses or add extra content.\n    If the given declarative sentence contains not, no or negative meaning words, you need to check the modified general interrogative sentence to make sure that the generated general question sentence retains words with not , no or negative meaning words.\n\nYou should present your result in the following format:\n### Modified sentences:\n- {Modified sentence 1}\n- {Modified sentence 2}\n- ...\n\n### Declarative sentences:"
    for fact in facts:
        content+="\n- {}\n".format(fact)

    messages = [
        {"role": "user", "content": content},
    ]
    return messages

def wrap_prompt_clssify_claim_to_list(item):
    optionA = item["options"][0]
    optionB = item["options"][1]
    responseA = item["response"][0]
    responseB = item["response"][1]
    content = "You need to determine if the two sentences are consistent. As long as either criterion is met, it can be judged as 'unrelated': \n1. The subjects described in these two sentences are inconsistent, for example, one describes the entire image while the other describes a certain object in the image. \n\nYou only need to output one word from 'unrelated' or 'consistent'. Please do not output any redundant information!\n\n"
    # content += "(Paragraph A): {}\n(Sentence A): {}\n\n(Paragraph B): {}\n(Sentence B): {}\n".format(responseA, optionA, responseB, optionB)
    content += "(Sentence A): {}\n(Sentence B): {}\n".format(optionA, optionB)
    messages = [
        {"role": "user", "content": content},
    ]
    return messages

def wrap_prompt_clssify_wh_to_list(item):
    optionA = item["wh_question"][0]
    optionB = item["wh_question"][1]
    content = "You need to determine if the two what-questions are equivalent in some sense. If you can confirm that two queries are equivalent in some sense, please output 'consistent'. Otherwise, output 'unrelated'. \n\nYou only need to output one word from 'unrelated' or 'consistent'. Please do not output any redundant information!\n\n"
    #content = "You need to determine if the two what-questions are equivalent in some sense. If you believe that there is at least one answer that can answer both questions simultaneously, then you consider these two questions to be 'consistent'. Otherwise, output 'unrelated'. \n\nYou only need to output one word from 'unrelated' or 'consistent'. Please do not output any redundant information!\n\n"
    # content += "(Paragraph A): {}\n(Sentence A): {}\n\n(Paragraph B): {}\n(Sentence B): {}\n".format(responseA, optionA, responseB, optionB)
    content += "(Question A): {}\n(Question B): {}\n".format(optionA, optionB)
    messages = [
        {"role": "user", "content": content},
    ]
    return messages

def wrap_prompt_what_question_to_list(item):
    facts=item["facts"]

    # v01
    content="You are an expert at modifying a given declarative sentence into a wh-question sentence. Your task is to modify the given declarative sentences one by one into a wh-question form. Do not change tenses or add extra content.\n\nYou should present your result in the following format:\n### Modified sentences:\n- {Modified sentence 1}\n- {Modified sentence 2}\n- ...\n\n\nPlease be careful not to output any unnecessary content. \nHere are the Declarative sentences you need to rewrite:"
    
    # v02
    #content="You are an expert at modifying a given declarative sentence into a wh-question sentence (i.e. questions starting with 'what/how/whose/which/when/where/how/why'). Your task is to modify the given declarative sentences one by one into a wh-question form. Do not change tenses or add extra content, but you can delete unnecessary content. \n\n You should present your result in the following format:\n### Modified sentences:\n- {Modified sentence 1}\n- {Modified sentence 2}\n- ...\n\n\nPlease be careful not to output any unnecessary content. \nHere are the Declarative sentences you need to rewrite."
    
    for fact in facts:
        content+="\n- {}\n".format(fact)

    messages = [
        {"role": "user", "content": content},
    ]
    return messages


prompt_func_dict = {
    "split": wrap_prompt_divide_to_list,
    "yesno": wrap_prompt_yesno_question_to_list,
    "classify_claim": wrap_prompt_clssify_claim_to_list,
    "classify_wh": wrap_prompt_clssify_wh_to_list,
    "what": wrap_prompt_what_question_to_list
}

key_dict = {
    "split": "facts",
    "yesno": "yesno_question",
    "classify_claim": "relation",
    "classify_wh": "relation",
    "what": "what_question"
}

do_get_facts_dict = {
    "split": True,
    "yesno": True,
    "classify_claim": False,
    "classify_wh": False,
    "what": True
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--chunk-num', type=int, default=1)
    parser.add_argument('--chunk-idx', type=int, default=0)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--prompt_type', type=str, default='split')
    args = parser.parse_args()
    # import ipdb; ipdb.set_trace()
    print(f"chunk_num={args.chunk_num}, chunk_idx={args.chunk_idx}")
    tokenizer, pipeline = init_pipline(args.checkpoint)
    transformers.logging.set_verbosity_error()

    path = args.path
    output_path = os.path.join(args.output_dir, os.path.basename(path))
    if ".jsonl" in os.path.basename(path):
        save_divide_path = output_path.replace('.jsonl', f'.s{args.start}-e{args.end}.chunk{args.chunk_num}-{args.chunk_idx}.jsonl')
    else:
        save_divide_path = output_path.replace('.json', f'.s{args.start}-e{args.end}.chunk{args.chunk_num}-{args.chunk_idx}.jsonl')
    print(f"==> save path = {save_divide_path}")

    assert args.prompt_type in prompt_func_dict.keys()
    wrap_func = prompt_func_dict[args.prompt_type]
    chosen_key = key_dict[args.prompt_type]
    do_get_facts = do_get_facts_dict[args.prompt_type]
    all_outputs = batch_inference(path, save_divide_path, tokenizer, pipeline, 
                                  key=chosen_key, wrap_func=wrap_func,
                                  batch_size=args.bs, chunk_num=args.chunk_num, 
                                  chunk_idx=args.chunk_idx, max_tokens=256,
                                  start=args.start, end=args.end,
                                  do_get_facts=do_get_facts)


if __name__ == "__main__":
    main()