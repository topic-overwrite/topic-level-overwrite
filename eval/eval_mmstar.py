from copy import deepcopy
import logging
from tqdm import tqdm
import argparse
import pickle
import json
import csv
import pandas as pd
import numpy as np


logger_initialized = {}

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
    except ImportError:
        rank = 0

    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    return logger


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)): 
            return None
        return json.JSONEncoder.default(self, obj)

def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)

def load(f):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](f) 


def MMStar_eval(eval_file, log_file):
    MMStar_score_l2 = {
        'coarse perception': {
            'image scene and topic': 0,
            'image style & quality': 0,
            'image emotion': 0
        },
        'fine-grained perception': {
            'object counting': 0,
            'recognition': 0,
            'localization': 0
        },
        'instance reasoning': {
            'single-instance reasoning': 0,
            'cross-instance attribute reasoning': 0,
            'cross-instance relation reasoning': 0
        },
        'logical reasoning': {
            'code & sequence reasoning': 0,
            'diagram reasoning': 0,
            'common reasoning': 0
        },
        'science & technology': {
            'biology & chemistry & physics': 0,
            'electronics & energy & mechanical eng.': 0,
            'geography & earth science & agriculture': 0
        },
        'math': {
            'geometry': 0,
            'numeric commonsense and calculation': 0,
            'statistical reasoning': 0
        },
    }
    MMStar_counter = deepcopy(MMStar_score_l2)
    logger = get_logger('Evaluation', log_file)

    lines = load(eval_file)
    for i in tqdm(range(len(lines))):
        line = lines[i]
        predict = str(line['prediction'])
        answers = str(line['answer'])
        ori_bench = str(line['bench'])
        category = str(line['category'])
        l2_category = str(line['l2_category'])
        MMStar_counter[category][l2_category] += 1

        answer = answers.lower().strip().replace('\n', ' ')
        predict = predict.lower().strip().replace('\n', ' ')
        # if ori_bench == 'MathVista' and answer not in ['a', 'b', 'c', 'd']:
        #     if answer in predict:
        #         MMStar_score_l2[category][l2_category] += 1
        # else:
        try:
            if answer == predict[0]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0] == '(' and answer == predict[1]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0:7] == 'option ' and answer == predict[7]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0:14] == 'the answer is ' and answer == predict[14]:
                MMStar_score_l2[category][l2_category] += 1
        except Exception as e:
            pass

    MMStar_score = {}
    MMStar_score['final score'] = 0
    for k, v in MMStar_score_l2.items():
        MMStar_score[k] = 0
        for l2_k, l2_v in v.items():
            MMStar_score[f'{k}({l2_k})'] = float(l2_v) / \
                float(MMStar_counter[k][l2_k])
            MMStar_score[k] += l2_v
        MMStar_score['final score'] += MMStar_score[k]
        MMStar_score[k] = float(MMStar_score[k]) / 250.0
    MMStar_score['final score'] = float(MMStar_score['final score']) / 1500.0

    score_pth = eval_file.replace('.json', '_score.json')
    dump(MMStar_score, score_pth)
    logger.info(
        f'MMStar_eval successfully finished evaluating {eval_file}, results saved in {score_pth}')
    logger.info('Score: ')
    for key, value in MMStar_score.items():
        logger.info('{}:{}'.format(key, value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="answer.json")
    parser.add_argument("--log_file", type=str, default="mmstar.log")
    args = parser.parse_args()
    
    MMStar_eval(args.answers_file, args.log_file)