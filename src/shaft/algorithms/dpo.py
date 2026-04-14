from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.config import DPOConfig, SFTTrainConfig
from shaft.training import ShaftDPOTrainer

from .base import AlgorithmContext
from .rlhf_utils import build_reference_model, build_trl_dpo_config
from .registry import register_algorithm


@dataclass
@register_algorithm("dpo")
class DPOAlgorithm:
    name: str = "dpo"

    def build_trainer(self, *, context: AlgorithmContext, **kwargs: Any) -> ShaftDPOTrainer:
        _ = context
        train_config: SFTTrainConfig = kwargs.pop("train_config")
        _ = train_config
        rlhf_config: DPOConfig = kwargs.pop("rlhf_config")
        training_args = kwargs.pop("args")
        finetune_mode: str = kwargs.pop("finetune_mode")
        model = kwargs.pop("model")
        ref_model = build_reference_model(
            model=model,
            finetune_mode=finetune_mode,
        )
        dpo_args = build_trl_dpo_config(
            train_args=training_args,
            rlhf_config=rlhf_config,
        )
        return ShaftDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_args,
            **kwargs,
        )
