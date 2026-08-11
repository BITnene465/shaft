from __future__ import annotations

from collections.abc import Mapping
import inspect
import math
from typing import Any

from shaft.training.input_contract import callable_semantic_signature
from shaft.training.resume_contract import (
    canonical_training_resume_value,
    training_module_implementation_signature,
)


def _require_context(context: Mapping[str, Any], key: str, *, algorithm: str) -> Any:
    value = context.get(key)
    if value is None:
        raise ValueError(f"{algorithm.upper()} training resume contract requires {key}.")
    return value


def _owning_module_signature(value: Any) -> str:
    module = inspect.getmodule(value)
    if module is None:
        raise ValueError(f"Cannot resolve owning module for {value!r}.")
    return training_module_implementation_signature(module)


def _dpo_reference_semantics(finetune_mode: object) -> str:
    mode = str(finetune_mode).strip().lower()
    if mode in {"lora", "dora", "qlora"}:
        return "policy_with_adapter_disabled"
    return "frozen_policy_copy"


def dpo_resume_objective(
    config: Any,
    training_args: Any,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    _ = training_args
    resolved_args = _require_context(
        context,
        "resolved_trainer_args",
        algorithm="dpo",
    )
    fields = (
        "disable_dropout",
        "pad_token",
        "max_length",
        "truncation_mode",
        "padding_free",
        "pad_to_multiple_of",
        "precompute_ref_log_probs",
        "precompute_ref_batch_size",
        "loss_type",
        "loss_weights",
        "ld_alpha",
        "f_divergence_type",
        "f_alpha_divergence_coef",
        "label_smoothing",
        "beta",
        "use_weighting",
        "discopop_tau",
        "activation_offloading",
        "sync_ref_model",
        "ref_model_mixup_alpha",
        "ref_model_sync_steps",
    )
    objective = {
        name: canonical_training_resume_value(getattr(resolved_args, name, None)) for name in fields
    }
    finetune_mode = str(config.model.finetune.mode).strip().lower()
    objective.update(
        {
            "finetune_mode": finetune_mode,
            "reference_model": _dpo_reference_semantics(finetune_mode),
        }
    )
    return objective


def grpo_resume_objective(
    config: Any,
    training_args: Any,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    from shaft.algorithms.grpo_rewards import GRPO_REWARD_REGISTRY
    from shaft.codec import CODEC_REGISTRY

    resolved_args = _require_context(
        context,
        "resolved_trainer_args",
        algorithm="grpo",
    )
    rewards = [
        {
            "name": str(reward.name),
            "codec": str(reward.codec),
            "weight": float(reward.weight),
            "params": canonical_training_resume_value(reward.params),
            "implementation_signature": callable_semantic_signature(
                GRPO_REWARD_REGISTRY.get(str(reward.name)),
                role=f"grpo_reward:{reward.name}",
            ),
            "implementation_module_signature": _owning_module_signature(
                GRPO_REWARD_REGISTRY.get(str(reward.name))
            ),
            "codec_implementation_signature": callable_semantic_signature(
                CODEC_REGISTRY.get(str(reward.codec)),
                role=f"grpo_codec:{reward.codec}",
            ),
            "codec_module_signature": _owning_module_signature(
                CODEC_REGISTRY.get(str(reward.codec))
            ),
        }
        for reward in config.rlhf.grpo.reward_functions
    ]
    fields = (
        "disable_dropout",
        "beta",
        "num_generations",
        "max_completion_length",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "generation_kwargs",
        "chat_template_kwargs",
        "cache_implementation",
        "use_transformers_paged",
        "ds3_gather_for_generation",
        "steps_per_generation",
        "num_iterations",
        "generation_batch_size",
        "shuffle_dataset",
        "use_vllm",
        "vllm_mode",
        "vllm_model_impl",
        "vllm_structured_outputs_regex",
        "epsilon",
        "delta",
        "epsilon_high",
        "sapo_temperature_neg",
        "sapo_temperature_pos",
        "importance_sampling_level",
        "reward_weights",
        "multi_objective_aggregation",
        "scale_rewards",
        "loss_type",
        "mask_truncated_completions",
        "sync_ref_model",
        "ref_model_mixup_alpha",
        "ref_model_sync_steps",
        "top_entropy_quantile",
        "max_tool_calling_iterations",
        "vllm_importance_sampling_correction",
        "vllm_importance_sampling_mode",
        "vllm_importance_sampling_cap",
        "off_policy_mask_threshold",
        "use_bias_correction_kl",
    )
    objective = {
        name: canonical_training_resume_value(getattr(resolved_args, name, None)) for name in fields
    }
    gradient_accumulation = int(training_args.gradient_accumulation_steps)
    generation_reuse_microsteps = int(resolved_args.steps_per_generation) * int(
        resolved_args.num_iterations
    )
    if gradient_accumulation <= 0 or generation_reuse_microsteps <= 0:
        raise ValueError("Resolved GRPO update cadence values must be > 0.")
    max_steps = int(training_args.max_steps)
    unique_prompts_per_group = int(resolved_args.generation_batch_size) // int(
        resolved_args.num_generations
    )
    objective.update(
        {
            "checkpoint_optimizer_step_cadence": (
                generation_reuse_microsteps
                // math.gcd(gradient_accumulation, generation_reuse_microsteps)
            ),
            "generation_reuse_microsteps": generation_reuse_microsteps,
            "step_horizon_unique_prompt_budget": (
                None
                if max_steps < 0
                else math.ceil(max_steps * gradient_accumulation / generation_reuse_microsteps)
                * unique_prompts_per_group
            ),
            "unique_prompts_per_generation_group": unique_prompts_per_group,
            "reward_functions": rewards,
        }
    )
    return objective


def build_trl_resume_implementation(
    *,
    algorithm: str,
    config_builder: Any,
    trainer_impl: type,
    extra_config_policy: tuple[str, ...] = (),
) -> dict[str, Any]:
    import shaft.algorithms.rlhf_utils as rlhf_utils_module
    from shaft.algorithms.registry import ALGORITHM_REGISTRY

    config_policy = [
        callable_semantic_signature(
            rlhf_utils_module._normalize_training_args_payload,
            role="trl_config:normalize_training_args",
        ),
        callable_semantic_signature(
            config_builder,
            role=f"trl_config:{algorithm}",
        ),
        *list(extra_config_policy),
    ]
    return {
        "algorithm_impl": ALGORITHM_REGISTRY.get(algorithm),
        "trainer_impl": trainer_impl,
        "objective_impl": {"trl_config_policy": config_policy},
        "package_names": ("trl",),
    }
