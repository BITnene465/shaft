from __future__ import annotations

from typing import Any

import torch
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from vlm_structgen.core.config import ExperimentRuntimeConfig


def _is_lora_param(name: str) -> bool:
    return "lora_" in name or ".lora_" in name


def build_optimizer(model: torch.nn.Module, config: ExperimentRuntimeConfig) -> torch.optim.Optimizer:
    lora_lr = config.train.lora_learning_rate or config.train.learning_rate

    groups = {
        "lora_params": {"params": [], "lr": lora_lr, "weight_decay": 0.0},
        "other": {"params": [], "lr": config.train.learning_rate, "weight_decay": config.train.weight_decay},
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_lora_param(name):
            groups["lora_params"]["params"].append(parameter)
        else:
            groups["other"]["params"].append(parameter)

    if config.finetune.mode == "lora":
        non_lora_groups = {
            group_name: len(payload["params"])
            for group_name, payload in groups.items()
            if group_name != "lora_params" and payload["params"]
        }
        if non_lora_groups:
            raise ValueError(
                "LoRA-only training expects only LoRA parameters to be trainable. "
                f"Unexpected trainable parameter groups: {sorted(non_lora_groups.items())}."
            )
        if not groups["lora_params"]["params"]:
            raise ValueError(
                "LoRA-only training found no trainable LoRA parameters. "
                "Check LoRA target module configuration."
            )

    param_groups = []
    for group_name, payload in groups.items():
        if not payload["params"]:
            continue
        param_groups.append(
            {
                "name": group_name,
                "params": payload["params"],
                "lr": payload["lr"],
                "weight_decay": payload["weight_decay"],
            }
        )
    return torch.optim.AdamW(param_groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentRuntimeConfig,
    total_training_steps: int,
):
    warmup_steps = int(total_training_steps * config.train.warmup_ratio)
    if config.train.scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
