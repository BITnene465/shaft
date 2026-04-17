from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from shaft.config import FinetuneConfig

from .freeze import (
    apply_full_freeze,
    build_freeze_plan,
    resolve_adapter_modules_to_save,
    resolve_adapter_target_modules,
)
from .types import ShaftModelAdapter


@dataclass
class FinetuneSummary:
    mode: str
    total_params: int
    trainable_params: int


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return total, trainable


def summarize_finetune(model: torch.nn.Module, mode: str) -> FinetuneSummary:
    total, trainable = _count_parameters(model)
    return FinetuneSummary(mode=str(mode), total_params=total, trainable_params=trainable)


def apply_finetune_strategy(
    model: torch.nn.Module,
    finetune: FinetuneConfig,
    *,
    model_adapter: ShaftModelAdapter,
) -> torch.nn.Module:
    mode = str(finetune.mode).strip().lower()
    freeze_plan = build_freeze_plan(model_adapter=model_adapter, finetune=finetune)
    if mode == "full":
        apply_full_freeze(model, freeze_plan)
        return model

    if mode not in {"lora", "dora", "qlora"}:
        raise ValueError(f"Unsupported finetune mode: {mode!r}")

    if mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    resolved_target_modules = resolve_adapter_target_modules(
        model,
        list(finetune.target_modules),
        plan=freeze_plan,
    )
    modules_to_save = resolve_adapter_modules_to_save(
        model,
        plan=freeze_plan,
        target_modules=resolved_target_modules,
    )

    peft_config = LoraConfig(
        r=int(finetune.lora_r),
        lora_alpha=int(finetune.lora_alpha),
        lora_dropout=float(finetune.lora_dropout),
        bias=str(finetune.lora_bias),
        target_modules=resolved_target_modules,
        modules_to_save=modules_to_save,
        task_type=TaskType.CAUSAL_LM,
        use_dora=(mode == "dora"),
        use_rslora=bool(finetune.use_rslora),
    )
    wrapped = get_peft_model(model, peft_config)
    return wrapped


def make_bnb_4bit_config(finetune: FinetuneConfig, *, dtype: torch.dtype | str) -> Any:
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # noqa: BLE001
        raise ImportError("BitsAndBytesConfig is unavailable in current transformers version.") from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=bool(finetune.qlora_use_double_quant),
        bnb_4bit_quant_type=str(finetune.qlora_quant_type),
        bnb_4bit_compute_dtype=dtype if isinstance(dtype, torch.dtype) else torch.bfloat16,
    )
