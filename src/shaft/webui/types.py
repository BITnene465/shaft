from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


ShaftRunStatus = Literal["idle", "validated", "running", "succeeded", "failed", "stopped"]


@dataclass
class ShaftSFTWebUIOverrides:
    run_id: str | None = None
    seed: int | None = None
    epochs: int | None = None
    max_steps: int | None = None
    learning_rate: float | None = None
    train_batch_size: int | None = None
    eval_batch_size: int | None = None
    mix_strategy: str | None = None
    optimizer_name: str | None = None
    scheduler_name: str | None = None
    scheduler_num_cycles: float | None = None
    scheduler_power: float | None = None
    loss_name: str | None = None
    loss_scale: str | None = None
    finetune_mode: str | None = None
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    qlora_load_in_4bit: bool | None = None
    freeze_groups: str | None = None
    freeze_prefixes: str | None = None
    freeze_regex: str | None = None
    trainable_prefixes: str | None = None
    trainable_regex: str | None = None
    use_cpu: bool | None = None
    init_from: str | None = None
    resume_from: str | None = None


@dataclass
class ShaftRunRecord:
    run_id: str
    algorithm: str
    status: ShaftRunStatus
    command: list[str]
    config_source_path: str
    resolved_config_path: str
    log_path: str
    output_dir: str
    pid: int | None = None
    return_code: int | None = None
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShaftRunRecord":
        return cls(**payload)

    @property
    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed", "stopped"}
