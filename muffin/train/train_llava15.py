import os
import sys
from llava.model import *
import gc
import torch
import random
import logging
import copy
import pathlib
import getpass
import transformers
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from torch.utils.data import Dataset

from muffin.train.args import ModelArguments, DataArguments, TrainingArguments
from utils.utils import is_main_process, get_rank
from muffin.train.trainers import LLaVA15DPOTrainer
from muffin.data.datasets import TPODataset
from muffin.train.train_utils import encode_multimodal_preference_sample, preprocess_v1

from muffin.train.train_muffin import DataCollatorForDPODataset
from functools import partial
import muffin.conversation as conversation_lib

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str,
                                   train_lora = False):
    """Collects the state dict and dump to disk."""
    if train_lora:
        torch.cuda.synchronize()
        state_dict = get_peft_state_maybe_zero_3(
            trainer.model.named_parameters(), "none"
        )
        trainer.model.save_pretrained(output_dir)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            trainer.model.named_parameters()
        )
        trainer.model.config.save_pretrained(output_dir)
        trainer.model.save_pretrained(output_dir, state_dict=state_dict)
        torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
    else:
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {
                key: value.cpu()
                for key, value in state_dict.items()
            }
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
            # if train_lora:
            #     if hasattr(trainer.model, 'lora_model'):
            #         lora_model = trainer.model.lora_model 
            #         lora_model.save_pretrained(output_dir + "/lora_weights")
            #     non_lora_trainables = {k: v for k, v in cpu_state_dict.items() if 'lora' not in k} 
            #     torch.save(non_lora_trainables, output_dir + "/non_lora_trainables.bin")
            #     config = trainer.model.config 
            #     with open(output_dir + "/config.json", "w") as f: 
            #         f.write(config.to_json_string())


class DPODataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_dir: str,
                 multimodal_cfg: dict,
                 reference_model = None,
                 raw_data_path = None):
        super(DPODataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = TPODataset(data_dir, reference_model, tokenizer,multimodal_cfg['image_token_len'], multimodal_cfg['image_processor'], multimodal_cfg['use_im_start_end'], is_llava15=True, raw_data_path=raw_data_path)
        self.multimodal_cfg = multimodal_cfg
        self.multimodal_cfg['keep_image_tag'] = True

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        source: dict = self.list_data_dict[i]
        preprocess_func = partial(preprocess_v1, has_image=True)
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(
            source, self.tokenizer, self.multimodal_cfg, preprocess_func=preprocess_func)
        return rej_data_dict, win_data_dict


def make_dpo_data_module(tokenizer, data_args, reference_model):
    train_dataset = DPODataset(tokenizer=tokenizer,
                               data_dir=data_args.data_dir,
                               multimodal_cfg=dict(
                                   is_multimodal=data_args.is_multimodal,
                                   image_token_len=data_args.image_token_len,
                                   image_folder=data_args.image_folder,
                                   image_aspect_ratio=data_args.image_aspect_ratio,
                                   use_im_start_end=getattr(
                                       data_args, 'mm_use_im_start_end', False),
                                   image_processor=getattr(
                                       data_args, 'image_processor', None),
                                   data_source_names=getattr(
                                       data_args, 'data_source_names'),
                                   data_source_weights=getattr(data_args, 'data_source_weights'),
                                   shuffle_data=data_args.shuffle_data
                                   ),
                               reference_model=reference_model,
                               raw_data_path=data_args.raw_data_path)
    print(f'Train data size is {len(train_dataset)}', flush=True)
    data_collator = DataCollatorForDPODataset(
        tokenizer=tokenizer, beta=data_args.dpo_beta, mod_token_weight=data_args.dpo_token_weight)

    if data_args.eval_data_source_names is not None:
        eval_datasets = {}
        for name in data_args.eval_data_source_names:
            eval_dataset = DPODataset(tokenizer=tokenizer,
                                      data_dir=data_args.data_dir,
                                      multimodal_cfg=dict(
                                          is_multimodal=data_args.is_multimodal,
                                          image_token_len=data_args.image_token_len,
                                          image_folder=data_args.image_folder,
                                          image_aspect_ratio=data_args.image_aspect_ratio,
                                          use_im_start_end=getattr(
                                              data_args, 'mm_use_im_start_end', False),
                                          image_processor=getattr(
                                              data_args, 'image_processor', None),
                                          data_source_names=[name],
                                          data_source_weights=[1],
                                           shuffle_data=False
                                          ),
                                      reference_model=reference_model)
            eval_datasets[name] = eval_dataset
    else:
        eval_datasets = None

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_datasets,
                data_collator=data_collator)


def init_model(model_args, data_args, training_args, attn_implementation):
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        # attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None)
    )
    model.config.use_cache = False
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        truncation_side='right',
    )
    # for llava 1.5
    tokenizer.pad_token = tokenizer.unk_token
    assert model_args.version == 'llava_v1'
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        print("conv template:", conversation_lib.default_conversation)
    else:
        raise NotImplementedError

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = lambda x: vision_tower.image_processor(x)['pixel_values'][0]
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.fully_tune:
        model.requires_grad_(True)

    if training_args.train_lora:
        # Initialize LoRA
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=128,  # Rank of the low-rank matrices
            lora_alpha=256,  # Scaling factor
            target_modules=["o_proj","gate_proj","down_proj","v_proj","q_proj","up_proj","k_proj"],  # Targeted modules
            lora_dropout=0.05,  # Dropout rate
            bias="none"
        )
        model = get_peft_model(model, lora_config)
    else:
        params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
        if is_main_process():
            print(f'No grad params are : {params_no_grad}', flush=True)

    if training_args.task == 'LM':
        raise NotImplementedError
    elif training_args.task == 'DPO':
        data_module = make_dpo_data_module(tokenizer, data_args=data_args, reference_model=copy.deepcopy(model).cuda())

    return model.cuda(), data_module, tokenizer


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"


def train(attn_implementation=None):
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.data_source_names = data_args.data_source_names.split('#')
    data_args.data_source_weights = [
        int(x) for x in data_args.data_source_weights.split('#')]

    data_args.eval_data_source_names = data_args.eval_data_source_names.split(
        '#') if data_args.eval_data_source_names is not None else None

    data_args.kto_win_data_source_names = data_args.kto_win_data_source_names.split('#')
    data_args.kto_win_data_source_weights = list(map(int, data_args.kto_win_data_source_weights.split('#')))
    data_args.kto_rej_data_source_names = data_args.kto_rej_data_source_names.split('#')
    data_args.kto_rej_data_source_weights = list(map(int, data_args.kto_rej_data_source_weights.split('#')))
    # import ipdb; ipdb.set_trace()
    model, data_module, tokenizer = init_model(
        model_args, data_args, training_args, attn_implementation=attn_implementation)

    if training_args.task == 'DPO':
        trainer = LLaVA15DPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module
        )
    else:
        # TODO
        raise NotImplementedError
    
    print(f'Training args: {training_args}')
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print(f'Resume from checkpoint.')
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f'Train from start.')
        if training_args.train_lora:
            params_with_grad = [n for n, p in model.named_parameters() if p.requires_grad]
            if is_main_process():
                print(f'With grad params are : {params_with_grad}', flush=True)
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir,
                                   train_lora=training_args.train_lora)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
