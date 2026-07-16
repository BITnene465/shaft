from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.config import PPOConfig, TrainConfig
from shaft.training.trl_trainers import ShaftPPOTrainer

from .base import AlgorithmContext, ShaftTrainerSpec, trainer_spec_contract
from .rlhf_utils import (
    build_ppo_value_and_reward_models,
    build_reference_model,
    build_trl_ppo_config,
    validate_ppo_runtime_requirements,
)
from .registry import register_algorithm

_PPO_ARG_FIELDS = (
    "batch_size",
    "cliprange",
    "cliprange_value",
    "gamma",
    "kl_coef",
    "kl_estimator",
    "lam",
    "local_batch_size",
    "local_mini_batch_size",
    "local_rollout_forward_batch_size",
    "micro_batch_size",
    "mini_batch_size",
    "missing_eos_penalty",
    "num_mini_batches",
    "num_ppo_epochs",
    "num_sample_generations",
    "num_total_batches",
    "response_length",
    "stop_token",
    "stop_token_id",
    "temperature",
    "total_episodes",
    "vf_coef",
    "whiten_rewards",
    "world_size",
)


@dataclass
@register_algorithm("ppo")
class PPOAlgorithm:
    name: str = "ppo"

    def prepare_trainer(
        self,
        *,
        context: AlgorithmContext,
        **kwargs: Any,
    ) -> ShaftTrainerSpec[ShaftPPOTrainer]:
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
        trainer_kwargs = {
            "args": ppo_args,
            "processing_class": kwargs.pop("processing_class"),
            "model": model,
            "ref_model": ref_model,
            "reward_model": reward_model,
            "value_model": value_model,
            "optimizer_name": train_config.optimizer_name,
            "scheduler_name": train_config.scheduler_name,
            "scheduler_num_cycles": train_config.scheduler_num_cycles,
            "scheduler_power": train_config.scheduler_power,
            "adam_beta1": train_config.adam_beta1,
            "adam_beta2": train_config.adam_beta2,
            "adam_epsilon": train_config.adam_epsilon,
            "model_adapter": kwargs.pop("model_adapter"),
            "finetune_plan": kwargs.pop("finetune_plan"),
            "resolved_optimizer_plan": kwargs.pop("resolved_optimizer_plan"),
            "param_group_lrs": dict(train_config.param_group_lrs),
            "no_decay_name_patterns": list(train_config.no_decay_name_patterns),
            **kwargs,  # train_dataset/eval_dataset/data_collator/callbacks
        }
        return ShaftTrainerSpec(
            trainer_cls=ShaftPPOTrainer,
            kwargs=trainer_kwargs,
            contract=trainer_spec_contract(
                algorithm=self.name,
                args=ppo_args,
                train_config=train_config,
                arg_fields=_PPO_ARG_FIELDS,
                extra={
                    "finetune_mode": str(finetune_mode).strip().lower(),
                    "reference_model": "adapter_disabled" if ref_model is None else "frozen_copy",
                    "value_model_mode": str(rlhf_config.value_model_mode).strip().lower(),
                    "reward_model_mode": str(rlhf_config.reward_model_mode).strip().lower(),
                    "train_value_backbone": bool(rlhf_config.train_value_backbone),
                    "allow_untrained_reward_model": bool(
                        rlhf_config.allow_untrained_reward_model
                    ),
                    "value_score_in_features": int(value_model.score.in_features),
                    "reward_score_in_features": int(reward_model.score.in_features),
                },
            ),
        )

    def build_trainer(self, *, context: AlgorithmContext, **kwargs: Any) -> ShaftPPOTrainer:
        return self.prepare_trainer(context=context, **kwargs).build()
