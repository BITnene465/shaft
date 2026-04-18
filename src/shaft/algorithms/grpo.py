from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.config import GRPOConfig, TrainConfig
from shaft.training import ShaftGRPOTrainer

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
        _ = train_config
        rlhf_config: GRPOConfig = kwargs.pop("rlhf_config")
        training_args = kwargs.pop("args")
        model = kwargs.pop("model")
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
            **kwargs,
        )
