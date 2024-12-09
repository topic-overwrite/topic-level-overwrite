import os
import io
import json
import math
import base64
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from itertools import combinations
import numpy as np
from transformers import CLIPProcessor, CLIPModel


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

def split_list(lst, n, repeat_num):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil((len(lst) / repeat_num) / n) * repeat_num  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k, repeat_num):
    chunks = split_list(lst, n, repeat_num)
    return chunks[k]

def sliding_window(image, step_ratio, window_ratio):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    window_size = (int(w * window_ratio[0]), int(h * window_ratio[1]))
    step_size = (int(w * step_ratio[0]), int(h * step_ratio[1]))
    windows = []
    y_len = len(range(0, h - window_size[1] + 1, step_size[1]))
    x_len = len(range(0, w - window_size[0] + 1, step_size[0]))
    for y in range(0, h - window_size[1] + 1, step_size[1]):
        for x in range(0, w - window_size[0] + 1, step_size[0]):
            window = img_array[y:y + window_size[1], x:x + window_size[0]]
            windows.append(window)
    return windows, y_len, x_len

def min_max_normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if min_val == max_val:
        return np.zeros(matrix.shape)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--output_name', type=str)
    parser.add_argument('--chunk-num', type=int, default=1)
    parser.add_argument('--chunk-idx', type=int, default=0)
    parser.add_argument('--repeat_num', type=int)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()

    print(f"chunk_num={args.chunk_num}, chunk_idx={args.chunk_idx}")
    path = args.path
    output_path = os.path.join(args.output_dir, args.output_name)
    if ".jsonl" in os.path.basename(path):
        save_divide_path = output_path.replace('.jsonl', f'.s{args.start}-e{args.end}.chunk{args.chunk_num}-{args.chunk_idx}.jsonl')
    else:
        save_divide_path = output_path.replace('.json', f'.s{args.start}-e{args.end}.chunk{args.chunk_num}-{args.chunk_idx}.jsonl')
    print(f"==> save path = {save_divide_path}")

    data_list = input_data(path)
    print("total data_list len:", len(data_list))
    start_question_id = int(data_list[0]['question_id'] // args.repeat_num)
    chunk_question_size = int(math.ceil((len(data_list) / args.repeat_num) / args.chunk_num))
    data_list = get_chunk(data_list, args.chunk_num, args.chunk_idx, args.repeat_num)
    image_data = input_data(args.image_path)
    
    model = CLIPModel.from_pretrained(args.checkpoint)
    process = CLIPProcessor.from_pretrained(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(save_divide_path), exist_ok=True)
    ans_f = open(save_divide_path, 'w')
    
    idx_bias = chunk_question_size * args.chunk_idx + start_question_id
    response_bias = idx_bias * args.repeat_num
    print(f"[{args.chunk_idx}chunk_idx] data len {len(data_list) // args.repeat_num}, start question {idx_bias}")
    assert len(data_list) % args.repeat_num == 0
    edge_num = 0
    for question_i in tqdm(range(len(data_list) // args.repeat_num), desc=f"[{args.chunk_idx}chunk_idx]"):
        start_id = question_i * args.repeat_num
        end_id = start_id + args.repeat_num
        data = data_list[start_id: end_id]
        image_id = question_i + idx_bias
        
        similarity_list = []
        for item in data:
            assert item['question_id'] // args.repeat_num == image_id, f"{item['question_id']}/repeat_num!={image_id}"
            img_b64 = image_data[image_id]['image']
            if len(img_b64) > 100:
                image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')
            else:
                image = Image.open(img_b64).convert('RGB')
            step_ratio = (0.125, 0.125)
            window_ratio = (0.5, 0.5)
            patch_images, y_len, x_len = sliding_window(image, step_ratio, window_ratio)
            description_list = item['facts']
            
            inputs = process(text=description_list, images=patch_images, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            except:
                import ipdb; ipdb.set_trace()
                print(inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

            similarity = torch.matmul(image_features, text_features.T)
            similarity = [min_max_normalize(similarity[:,i].reshape(y_len, x_len).cpu().detach().numpy()) for i in range(len(description_list))]
            similarity_list.append(similarity)
        
        for i, j in combinations(range(args.repeat_num), 2):
            for idx_i, item_i in enumerate(similarity_list[i]):
                for idx_j, item_j in enumerate(similarity_list[j]):
                    corrcoef = np.corrcoef(item_i.flatten(), item_j.flatten())[0, 1]
                    if corrcoef > args.threshold:
                        edge_num += 1
                        line = {
                            "claim_id": [(response_bias+start_id+i, idx_i), (response_bias+start_id+j, idx_j)],
                            "relation": "consistent",
                            "corrcoef": corrcoef
                        }
                        ans_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                        ans_f.flush()
                    else:
                        line = {
                            "claim_id": [(response_bias+start_id+i, idx_i), (response_bias+start_id+j, idx_j)],
                            "relation": "unrelated",
                            "corrcoef": corrcoef
                        }
                        ans_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                        ans_f.flush()
    print("edge_num:", edge_num)
    ans_f.close()

if __name__ == "__main__":
    main()
