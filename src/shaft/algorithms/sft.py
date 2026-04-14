from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.config import SFTTrainConfig
from shaft.training import ShaftSFTTrainer

from .base import AlgorithmContext
from .registry import register_algorithm


@dataclass
@register_algorithm("sft")
class SFTAlgorithm:
    name: str = "sft"

    def build_trainer(self, *, context: AlgorithmContext, **kwargs: Any) -> ShaftSFTTrainer:
        _ = context
        train_config: SFTTrainConfig = kwargs.pop("train_config")
        return ShaftSFTTrainer(
            loss_name=train_config.loss_name,
            optimizer_name=train_config.optimizer_name,
            scheduler_name=train_config.scheduler_name,
            scheduler_num_cycles=train_config.scheduler_num_cycles,
            scheduler_power=train_config.scheduler_power,
            adam_beta1=train_config.adam_beta1,
            adam_beta2=train_config.adam_beta2,
            adam_epsilon=train_config.adam_epsilon,
            ignore_index=-100,
            **kwargs,
        )
