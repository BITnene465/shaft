from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    name: str = "shaft"
    seed: int = 42
    output_dir: str = "outputs/default"
    run_id: str | None = None


@dataclass
class ModelConfig:
    model_type: str = "qwen3vl"
    model_name_or_path: str = "models/Qwen3-VL-4B-Instruct"
    template: str | None = None
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    torch_dtype: str = "bfloat16"
    finetune: "FinetuneConfig" = field(default_factory=lambda: FinetuneConfig())


@dataclass
class FinetuneConfig:
    mode: str = "full"  # full | lora | dora | qlora
    target_modules: list[str] = field(default_factory=lambda: ["auto"])
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    use_rslora: bool = False
    qlora_load_in_4bit: bool = True
    qlora_use_double_quant: bool = True
    qlora_quant_type: str = "nf4"
    qlora_compute_dtype: str = "bfloat16"


@dataclass
class DatasetSourceConfig:
    dataset_name: str
    source_type: str = "jsonl_sft"
    train_path: str | None = None
    val_path: str | None = None
    train_paths: list[str] = field(default_factory=list)
    val_paths: list[str] = field(default_factory=list)
    weight: float = 1.0
    enabled: bool = True
    offline_transforms: list[str] = field(default_factory=list)
    online_transforms: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    catalog_path: str | None = None
    catalog_names: list[str] = field(default_factory=list)
    datasets: list[DatasetSourceConfig] = field(default_factory=list)
    mix_strategy: str = "interleave_under"
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    min_pixels: int | None = 200704
    max_pixels: int | None = 1048576
    add_eos_token: bool = True
    shuffle: bool = True


@dataclass
class AlgorithmConfig:
    name: str = "sft"  # sft | dpo | ppo
    params: dict[str, Any] = field(default_factory=dict)


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
class EvalConfig:
    enabled: bool = True
    per_device_eval_batch_size: int = 1
    eval_strategy: str = "epoch"  # no | steps | epoch
    eval_steps: int = 200
    do_sample: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 512
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class PluginsConfig:
    hooks: list[str] = field(default_factory=list)
    interceptors: list[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    fmt: str = "text"  # text | json
    file_path: str | None = None
    rank_zero_only: bool = True


@dataclass
class ProgressConfig:
    enabled: bool = True
    leave: bool = False
    mininterval: float = 0.2


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


@dataclass
class RuntimeConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    rlhf: RLHFConfig = field(default_factory=RLHFConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
