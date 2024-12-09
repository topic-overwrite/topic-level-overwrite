import io
import gc
import os
import json
import random
import numpy
import torch
import base64
import pandas as pd
import os.path as op
import torch.utils.data as torch_data

from PIL import Image
from typing import List, Iterator
from muffin.data.tsv_file import TSVFile
from torch.utils.data.sampler import Sampler
from muffin.eval.muffin_inference_logp import inference_logp
import datasets as hf_datasets

def bytes_to_PIL_image(img_buffer):
    img_io = io.BytesIO(img_buffer)
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image

class TPODataset(torch_data.Dataset):
    def __init__(self, data_dir: str, reference_model=None,
                 tokenizer=None, image_token_len=None, img_processor=None, use_im_start_end=True, is_llava15=False, raw_data_path=None):
        super().__init__()

        if not op.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        data_path = [file for file in os.listdir(data_dir) if file.endswith('.parquet') and 'logp' in file]
        self.data_path = data_dir
        
        if len(data_path) == 0:
            assert reference_model is not None, "`reference_model` is mandatory when logps do not exist."
            assert raw_data_path is not None
            if not op.exists(raw_data_path):
                os.mkdir(raw_data_path)
            hf_data = hf_datasets.load_dataset(raw_data_path)['train'].cast_column("image", hf_datasets.Image(decode=False))

            inference_logp(reference_model, tokenizer, hf_data, self.data_path,
                            image_token_len, img_processor, use_im_start_end, is_llava15=is_llava15)

            torch.distributed.barrier()

            self.data = hf_datasets.load_dataset(data_dir)['train'].cast_column("image", hf_datasets.Image(decode=False))
        else:
            self.data = hf_datasets.load_dataset(data_dir)['train'].cast_column("image", hf_datasets.Image(decode=False))

        self.line_idx = list(range(len(self.data)))
        random.shuffle(self.line_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[self.line_idx[index]]
        question = {'from': 'human', 'value': f"<image>\n{sample['question']}"}
        chosen = {'from': 'gpt', 'value': sample['chosen']}
        rejected = {'from': 'gpt', 'value': sample['rejected']}

        image = bytes_to_PIL_image(sample['image']['bytes'])

        metainfo = {
            "origin_dataset": sample['origin_dataset'],
            "origin_split": sample['origin_split'],
            "origin_idx": sample['idx'],
            "image_id": sample['image_path'],
        }

        data_dict = {
            'image': image,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "metainfo": metainfo
        }
        logps=json.loads(sample['logps'])

        if type(logps) == type([]):
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
            data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = logps
        else:
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
            data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = logps['logps']

        return data_dict


class ChunckedRandomSampler(Sampler[int]):
    def __init__(self, data_source, chunk_size=5000) -> None:
        self.data_source = data_source
        self.chunk_size = chunk_size

    def __iter__(self):
        n = len(self.data_source)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        print(f'Chuncked Random Sampler seed is {seed}')
        generator = torch.Generator()
        generator.manual_seed(seed)

        for st in torch.randperm(n // self.chunk_size, generator=generator).tolist():
            base = st * self.chunk_size
            for i in torch.randperm(self.chunk_size, generator=generator).tolist():
                yield base + i

        base = (n // self.chunk_size) * self.chunk_size
        for i in torch.randperm(n % self.chunk_size, generator=generator).tolist():
            yield base + i

    def __len__(self) -> int:
        return len(self.data_source)
