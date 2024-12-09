from dataclasses import dataclass, field
from typing import Dict, Optional
import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="llava_v1") # llava 1.5
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    parquet: bool = False
    data_source_names: str = 'unimm-chat'
    data_source_weights: str = '100'
    eval_data_source_names: Optional[str] = field(default=None)
    data_dir: str = './TPO-Dataset/'
    raw_data_path: str = './TPO-Dataset/'
    kto_win_data_source_names: str = '100'
    kto_win_data_source_weights: str = '100'
    kto_rej_data_source_names : str = '100'
    kto_rej_data_source_weights: str = '100'

    dpo_beta: float = 0.5
    dpo_token_weight: float = 3.0

    shuffle_data: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_steps: int = field(default=1_000)
    no_randaug: bool = False

    task: str = field(
        default='LM',
        metadata={
            'help': 'LM for language modeling. DPO for direct preference optimization'
        }
    )
    dpo_use_average: bool = False
    dpo_token_weighted: bool = False

    train_lora: bool = False
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    fully_tune: bool = False