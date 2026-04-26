from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.config import PPOConfig, TrainConfig
from shaft.training import ShaftPPOTrainer

from .base import AlgorithmContext
from .rlhf_utils import (
    build_ppo_value_and_reward_models,
    build_reference_model,
    build_trl_ppo_config,
    validate_ppo_runtime_requirements,
)
from .registry import register_algorithm


@dataclass
@register_algorithm("ppo")
class PPOAlgorithm:
    name: str = "ppo"

    def build_trainer(self, *, context: AlgorithmContext, **kwargs: Any) -> ShaftPPOTrainer:
        _ = context
        train_config: TrainConfig = kwargs.pop("train_config")
        rlhf_config: PPOConfig = kwargs.pop("rlhf_config")
        training_args = kwargs.pop("args")
        finetune_mode: str = kwargs.pop("finetune_mode")
        model = kwargs.pop("model")
        model_meta = kwargs.pop("model_meta")
        validate_ppo_runtime_requirements(
            model_meta=model_meta,
            model=model,
            finetune_mode=finetune_mode,
            rlhf_config=rlhf_config,
        )
        ref_model = build_reference_model(
            model=model,
            finetune_mode=finetune_mode,
        )
        value_model, reward_model = build_ppo_value_and_reward_models(
            model=model,
            train_value_backbone=bool(rlhf_config.train_value_backbone),
            value_model_mode=str(rlhf_config.value_model_mode),
            reward_model_mode=str(rlhf_config.reward_model_mode),
            allow_untrained_reward_model=bool(rlhf_config.allow_untrained_reward_model),
        )
        ppo_args = build_trl_ppo_config(
            train_args=training_args,
            rlhf_config=rlhf_config,
        )
        return ShaftPPOTrainer(
            args=ppo_args,
            processing_class=kwargs.pop("processing_class"),
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            value_model=value_model,
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
            **kwargs,  # train_dataset/eval_dataset/data_collator/callbacks
        )
