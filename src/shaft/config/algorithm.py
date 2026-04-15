from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AlgorithmConfig:
    name: str = "sft"  # sft | dpo | ppo
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
class RLHFConfig:
    enabled: bool = False
    dpo: DPOConfig = field(default_factory=DPOConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
