from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.config import GRPOConfig, TrainConfig
from shaft.training.trl_trainers import ShaftGRPOTrainer

from .base import AlgorithmContext, ShaftTrainerSpec, trainer_spec_contract
from .grpo_rewards import build_grpo_reward_functions
from .registry import register_algorithm
from .rlhf_utils import build_trl_grpo_config

_GRPO_ARG_FIELDS = (
    "beta",
    "cache_implementation",
    "chat_template_kwargs",
    "delta",
    "disable_dropout",
    "ds3_gather_for_generation",
    "epsilon",
    "epsilon_high",
    "generation_batch_size",
    "generation_kwargs",
    "importance_sampling_level",
    "mask_truncated_completions",
    "max_completion_length",
    "max_tool_calling_iterations",
    "min_p",
    "multi_objective_aggregation",
    "num_generations",
    "num_iterations",
    "off_policy_mask_threshold",
    "ref_model_mixup_alpha",
    "ref_model_sync_steps",
    "repetition_penalty",
    "reward_weights",
    "sapo_temperature_neg",
    "sapo_temperature_pos",
    "scale_rewards",
    "shuffle_dataset",
    "steps_per_generation",
    "sync_ref_model",
    "temperature",
    "top_entropy_quantile",
    "top_k",
    "top_p",
    "use_bias_correction_kl",
    "use_transformers_paged",
    "use_vllm",
    "vllm_importance_sampling_cap",
    "vllm_importance_sampling_correction",
    "vllm_importance_sampling_mode",
    "vllm_mode",
    "vllm_model_impl",
    "vllm_structured_outputs_regex",
)


@dataclass
@register_algorithm("grpo")
class GRPOAlgorithm:
    name: str = "grpo"

    def prepare_trainer(
        self,
        *,
        context: AlgorithmContext,
        **kwargs: Any,
    ) -> ShaftTrainerSpec[ShaftGRPOTrainer]:
        _ = context
        train_config: TrainConfig = kwargs.pop("train_config")
        rlhf_config: GRPOConfig = kwargs.pop("rlhf_config")
        training_args = kwargs.pop("args")
        model = kwargs.pop("model")
        kwargs.pop("finetune_mode", None)
        grpo_args = kwargs.pop("resolved_grpo_args", None)
        if grpo_args is None:
            grpo_args = build_trl_grpo_config(
                train_args=training_args,
                rlhf_config=rlhf_config,
            )
        reward_funcs = build_grpo_reward_functions(rlhf_config.reward_functions)
        trainer_kwargs = {
            "model": model,
            "reward_funcs": reward_funcs,
            "args": grpo_args,
            "processing_class": kwargs.pop("processing_class"),
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
            **kwargs,
        }
        return ShaftTrainerSpec(
            trainer_cls=ShaftGRPOTrainer,
            kwargs=trainer_kwargs,
            contract=trainer_spec_contract(
                algorithm=self.name,
                args=grpo_args,
                train_config=train_config,
                arg_fields=_GRPO_ARG_FIELDS,
                extra={
                    "reward_functions": [
                        {
                            "name": reward.name,
                            "codec": reward.codec,
                            "weight": reward.weight,
                            "params": dict(reward.params),
                        }
                        for reward in rlhf_config.reward_functions
                    ],
                },
            ),
        )

    def build_trainer(self, *, context: AlgorithmContext, **kwargs: Any) -> ShaftGRPOTrainer:
        return self.prepare_trainer(context=context, **kwargs).build()
