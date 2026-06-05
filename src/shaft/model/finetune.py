from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from shaft.config import FinetuneConfig

from .finetune_plan import ShaftResolvedFinetunePlan, build_resolved_finetune_plan
from .generation import set_model_use_cache
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
        count = _parameter_numel(parameter)
        total += count
        if parameter.requires_grad:
            trainable += count
    return total, trainable


def _parameter_numel(parameter: torch.nn.Parameter) -> int:
    deepspeed_numel = getattr(parameter, "ds_numel", None)
    if deepspeed_numel is not None:
        return int(deepspeed_numel)
    deepspeed_shape = getattr(parameter, "ds_shape", None)
    if deepspeed_shape is not None:
        total = 1
        for dim in deepspeed_shape:
            total *= int(dim)
        return int(total)
    return int(parameter.numel())


def summarize_finetune(model: torch.nn.Module, mode: str) -> FinetuneSummary:
    total, trainable = _count_parameters(model)
    return FinetuneSummary(mode=str(mode), total_params=total, trainable_params=trainable)


def apply_resolved_finetune_plan(
    model: torch.nn.Module,
    plan: ShaftResolvedFinetunePlan,
    *,
    finetune: FinetuneConfig,
    gradient_checkpointing: bool = False,
) -> torch.nn.Module:
    if gradient_checkpointing:
        _ = set_model_use_cache(model, enabled=False)

    if plan.mode == "full":
        trainable_names = set(plan.parameter_plan.trainable_parameter_names)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in trainable_names)
        return model

    if plan.mode not in {"lora", "dora", "qlora"}:
        raise ValueError(f"Unsupported finetune mode: {plan.mode!r}")

    if plan.mode == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(gradient_checkpointing),
        )

    if plan.adapter_plan is None:
        raise ValueError(f"Missing adapter finetune plan for mode={plan.mode!r}.")

    peft_config = LoraConfig(
        r=plan.adapter_plan.peft_signature.r,
        lora_alpha=plan.adapter_plan.peft_signature.lora_alpha,
        lora_dropout=float(finetune.lora_dropout),
        bias=plan.adapter_plan.peft_signature.lora_bias,
        target_modules=list(plan.adapter_plan.resolved_target_modules),
        modules_to_save=list(plan.adapter_plan.modules_to_save),
        task_type=TaskType.CAUSAL_LM,
        use_dora=plan.adapter_plan.peft_signature.use_dora,
        use_rslora=plan.adapter_plan.peft_signature.use_rslora,
    )
    wrapped = get_peft_model(model, peft_config)
    if gradient_checkpointing and hasattr(wrapped, "enable_input_require_grads"):
        wrapped.enable_input_require_grads()
    return wrapped


def apply_finetune_strategy(
    model: torch.nn.Module,
    finetune: FinetuneConfig,
    *,
    model_adapter: ShaftModelAdapter,
    gradient_checkpointing: bool = False,
) -> torch.nn.Module:
    plan = build_resolved_finetune_plan(model, finetune, model_adapter=model_adapter)
    wrapped = apply_resolved_finetune_plan(
        model,
        plan,
        finetune=finetune,
        gradient_checkpointing=gradient_checkpointing,
    )
    setattr(wrapped, "_shaft_finetune_plan", plan)
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
