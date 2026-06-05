from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.config import GRPOConfig, TrainConfig
from shaft.training.trl_trainers import ShaftGRPOTrainer

from .base import AlgorithmContext
from .grpo_rewards import build_grpo_reward_functions
from .registry import register_algorithm
from .rlhf_utils import build_trl_grpo_config


@dataclass
@register_algorithm("grpo")
class GRPOAlgorithm:
    name: str = "grpo"

    def build_trainer(self, *, context: AlgorithmContext, **kwargs: Any) -> ShaftGRPOTrainer:
        _ = context
        train_config: TrainConfig = kwargs.pop("train_config")
        rlhf_config: GRPOConfig = kwargs.pop("rlhf_config")
        training_args = kwargs.pop("args")
        model = kwargs.pop("model")
        kwargs.pop("finetune_mode", None)
        grpo_args = build_trl_grpo_config(
            train_args=training_args,
            rlhf_config=rlhf_config,
        )
        reward_funcs = build_grpo_reward_functions(rlhf_config.reward_functions)
        kwargs.pop("train_sampler", None)
        return ShaftGRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=grpo_args,
            processing_class=kwargs.pop("processing_class"),
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
