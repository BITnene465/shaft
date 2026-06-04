from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainFSDPConfig:
    sharding_strategy: str = "full_shard"
    auto_wrap_policy: str = "transformer"
    transformer_layer_cls_to_wrap: list[str] = field(default_factory=lambda: ["auto"])
    min_num_params: int = 0
    activation_checkpointing: bool = True
    cpu_offload: bool = False
    use_orig_params: bool = True
    backward_prefetch: str | None = None
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    state_dict_type: str = "full_state_dict"
    sync_module_states: bool = False


@dataclass
class TrainDeepSpeedConfig:
    config_path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainDistributedConfig:
    strategy: str = "ddp"  # ddp | fsdp | deepspeed
    fsdp: TrainFSDPConfig = field(default_factory=TrainFSDPConfig)
    deepspeed: TrainDeepSpeedConfig = field(default_factory=TrainDeepSpeedConfig)


@dataclass
class TrainConfig:
    epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-5
    param_group_lrs: dict[str, float] = field(default_factory=dict)
    no_decay_name_patterns: list[str] = field(default_factory=list)
    optimizer_name: str = "adamw_torch"
    scheduler_name: str = "auto"
    scheduler_num_cycles: float = 0.5
    scheduler_power: float = 1.0
    loss_name: str = "auto"
    loss_scale: str = "default"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    bf16: bool = True
    use_cpu: bool = False
    logging_steps: int = 10
    save_strategy: str = "epoch"  # no | steps | epoch
    save_epoch_interval: int = 1
    save_steps: int = 200
    save_total_limit: int = 3
    ddp_find_unused_parameters: bool = False
    report_to: list[str] = field(default_factory=lambda: ["none"])
    load_best_model_at_end: bool = True
    save_final_model: bool = True
    save_final_state: bool = True
    init_from_checkpoint: str | None = None
    resume_from_checkpoint: str | None = None
    distributed: TrainDistributedConfig = field(default_factory=TrainDistributedConfig)


@dataclass
class EvalMetricConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalNormalizerConfig:
    type: str = "identity"
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class EvalDatasetPolicyConfig:
    prediction_codec: str = "text"
    target_adapter: str = "target_text"
    target_adapter_params: dict[str, Any] = field(default_factory=dict)
    metrics: list[EvalMetricConfig] = field(default_factory=list)
    primary_metric: str = ""
    normalizer: EvalNormalizerConfig = field(default_factory=EvalNormalizerConfig)
    weight: float = 1.0


@dataclass
class EvalConfig:
    enabled: bool = True
    per_device_eval_batch_size: int = 1
    eval_strategy: str = "epoch"  # no | steps | epoch
    epoch_interval: int = 1
    eval_steps: int = 200
    loss_metrics_enabled: bool = True
    do_sample: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 512
    online_metrics_enabled: bool = False
    datasets: dict[str, EvalDatasetPolicyConfig] = field(default_factory=dict)
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


def resolve_effective_gradient_checkpointing(config: Any) -> bool:
    train_cfg = config.train
    distributed = train_cfg.distributed
    if (
        distributed.strategy == "fsdp"
        and bool(distributed.fsdp.activation_checkpointing)
        and bool(train_cfg.gradient_checkpointing)
    ):
        return False
    return bool(train_cfg.gradient_checkpointing)
