import os
from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class DataArguments:
    data_name: str  # a|b|c
    data_processor: Optional[str] = field(default=None)
    data_mode: Optional[str] = field(default=None)
    processed_data_dir: Optional[str] = field(default=None)
    is_multimodal: bool = field(default=True)
    image_dir: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default='pad')
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class ModelArguments:
    model_name: str
    model_path: str
    vision_tower: str
    freeze_vision_tower: bool = field(default=True)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    num_experts: int = field(default=4)
    use_moe_residual: bool = field(default=True)
    use_moe: bool = field(default=True)
    moe_intermediate_size: int = field(default=4096)
    moe_top_k: int = field(default=2)
    moe_weight: float = field(default=0.1)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    logging_dir: str = field(default=os.getenv('SUMMARY_DIR'))
    monitor_step: int = field(default=200)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    def unfreeze(self):
        self._frozen = False

    def freeze(self):
        self._frozen = True
