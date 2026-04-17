from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainConfig:
    epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
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
    save_steps: int = 200
    save_total_limit: int = 3
    ddp_find_unused_parameters: bool = False
    report_to: list[str] = field(default_factory=lambda: ["none"])
    load_best_model_at_end: bool = True
    save_final_model: bool = True
    save_final_state: bool = True
    init_from_checkpoint: str | None = None
    resume_from_checkpoint: str | None = None


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
    eval_steps: int = 200
    do_sample: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 512
    online_metrics_enabled: bool = False
    datasets: dict[str, EvalDatasetPolicyConfig] = field(default_factory=dict)
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
