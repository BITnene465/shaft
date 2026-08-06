from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


SFT_AUXILIARY_LOSS_WEIGHTS_PARAM = "auxiliary_loss_weights"


def normalize_auxiliary_loss_weights(
    value: Any,
    *,
    field_name: str = "algorithm.params.auxiliary_loss_weights",
) -> dict[str, float]:
    """Normalize the generic SFT auxiliary-objective coefficient overrides."""

    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping.")
    normalized: dict[str, float] = {}
    for raw_name, raw_weight in value.items():
        if not isinstance(raw_name, str):
            raise TypeError(f"{field_name} term names must be strings.")
        name = raw_name.strip().lower()
        if not name:
            raise ValueError(f"{field_name} contains an empty term name.")
        if name in normalized:
            raise ValueError(
                f"{field_name} contains duplicate normalized term name {name!r}."
            )
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError(f"{field_name}.{name} must be a number.")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{field_name}.{name} must be finite and >= 0.")
        if weight == 0.0:
            weight = 0.0
        normalized[name] = weight
    return dict(sorted(normalized.items()))


def normalize_sft_algorithm_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("algorithm.params must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("algorithm.params keys must be strings.")
    unknown = sorted(set(value) - {SFT_AUXILIARY_LOSS_WEIGHTS_PARAM})
    if unknown:
        raise ValueError(f"Unknown algorithm.params keys for built-in SFT: {unknown}.")
    weights = normalize_auxiliary_loss_weights(
        value.get(SFT_AUXILIARY_LOSS_WEIGHTS_PARAM, {})
    )
    if not weights:
        return {}
    return {SFT_AUXILIARY_LOSS_WEIGHTS_PARAM: weights}


@dataclass
class AlgorithmConfig:
    name: str = "sft"  # sft | dpo | ppo | grpo
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DPOConfig:
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    precompute_ref_log_probs: bool = False
    use_weighting: bool = False


@dataclass
class PPOConfig:
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    kl_coef: float = 0.02
    vf_coef: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    whiten_rewards: bool = False
    response_length: int = 128
    temperature: float = 1.0
    num_ppo_epochs: int = 4
    num_mini_batches: int = 1
    local_rollout_forward_batch_size: int = 16
    num_sample_generations: int = 0
    stop_token: str | None = "eos"
    value_model_mode: str = "shared_backbone"  # shared_backbone | copy_backbone
    reward_model_mode: str = "adapter_disabled_policy"  # adapter_disabled_policy | copy_backbone
    train_value_backbone: bool = False
    allow_untrained_reward_model: bool = False
    allow_text_only_multimodal_ppo: bool = False


@dataclass
class GRPORewardConfig:
    name: str = "exact_match"
    codec: str = "json_any"
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GRPORolloutConfig:
    num_generations: int = 8
    num_generations_eval: int | None = 1
    max_completion_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float | None = None
    repetition_penalty: float = 1.0
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    cache_implementation: str | None = None
    use_transformers_paged: bool = False


@dataclass
class GRPOVLLMConfig:
    enabled: bool = False
    mode: str = "server"  # server | colocate
    model_impl: str = "vllm"  # vllm | transformers
    enable_sleep_mode: bool = False
    structured_outputs_regex: str | None = None
    server_base_url: str | None = None
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_timeout: float = 240.0
    group_port: int = 51216
    gpu_memory_utilization: float = 0.3
    max_model_length: int | None = None
    tensor_parallel_size: int = 1


@dataclass
class GRPOConfig:
    beta: float = 0.0
    rollout: GRPORolloutConfig = field(default_factory=GRPORolloutConfig)
    vllm: GRPOVLLMConfig = field(default_factory=GRPOVLLMConfig)
    reward_functions: list[GRPORewardConfig] = field(
        default_factory=lambda: [GRPORewardConfig()]
    )
    # Backward-compatible flat aliases. New configs should use rollout/vllm.
    num_generations: int | None = None
    num_generations_eval: int | None = None
    max_completion_length: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    use_vllm: bool | None = None


@dataclass
class RLHFConfig:
    enabled: bool = False
    dpo: DPOConfig = field(default_factory=DPOConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
