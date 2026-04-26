from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.config import DPOConfig, TrainConfig
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
        train_config: TrainConfig = kwargs.pop("train_config")
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
            optimizer_name=train_config.optimizer_name,
            scheduler_name=train_config.scheduler_name,
            scheduler_num_cycles=train_config.scheduler_num_cycles,
            scheduler_power=train_config.scheduler_power,
            adam_beta1=train_config.adam_beta1,
            adam_beta2=train_config.adam_beta2,
            adam_epsilon=train_config.adam_epsilon,
            model_adapter=kwargs.pop("model_adapter"),
            finetune_plan=kwargs.pop("finetune_plan"),
            param_group_lrs=dict(train_config.param_group_lrs),
            no_decay_name_patterns=list(train_config.no_decay_name_patterns),
            **kwargs,
        )
