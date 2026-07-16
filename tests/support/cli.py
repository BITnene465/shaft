from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from shaft.config import RuntimeConfig

def build_common_train_args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "config": "dummy.yaml",
        "run_id": None,
        "seed": None,
        "epochs": None,
        "max_steps": None,
        "gradient_checkpointing": None,
        "learning_rate": None,
        "train_batch_size": None,
        "eval_batch_size": None,
        "mix_strategy": None,
        "optimizer_name": None,
        "scheduler_name": None,
        "scheduler_num_cycles": None,
        "scheduler_power": None,
        "loss_name": None,
        "loss_scale": None,
        "finetune_mode": None,
        "lora_r": None,
        "lora_alpha": None,
        "lora_dropout": None,
        "qlora_load_in_4bit": None,
        "use_cpu": None,
        "init_from": None,
        "resume_from": None,
        "algorithm": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def capture_algorithm_runner(captured: dict[str, str]) -> Callable[[RuntimeConfig], dict[str, int]]:
    def _runner(config: RuntimeConfig) -> dict[str, int]:
        captured["algorithm"] = config.algorithm.name
        return {"ok": 1}

    return _runner
