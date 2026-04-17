from __future__ import annotations

from dataclasses import dataclass, field

import torch

from shaft.config import FinetuneConfig

from .freeze import (
    ShaftFreezePlan,
    build_freeze_plan,
    resolve_adapter_modules_to_save,
    resolve_adapter_target_modules,
)
from .types import ShaftModelAdapter


def _tuple_from_names(values: list[str] | tuple[str, ...] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = [values]
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


@dataclass(frozen=True)
class ShaftParameterSelectionPlan:
    trainable_parameter_names: tuple[str, ...] = ()
    frozen_parameter_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShaftPeftSignature:
    target_modules: tuple[str, ...] = ()
    modules_to_save: tuple[str, ...] = ()
    r: int = 0
    lora_alpha: int = 0
    lora_bias: str = "none"
    use_rslora: bool = False
    use_dora: bool = False


@dataclass(frozen=True)
class ShaftAdapterFinetunePlan:
    resolved_target_modules: tuple[str, ...] = ()
    modules_to_save: tuple[str, ...] = ()
    peft_signature: ShaftPeftSignature = field(default_factory=ShaftPeftSignature)


@dataclass(frozen=True)
class ShaftResolvedFinetunePlan:
    mode: str
    freeze_plan: ShaftFreezePlan
    parameter_plan: ShaftParameterSelectionPlan
    adapter_plan: ShaftAdapterFinetunePlan | None = None


def _build_parameter_selection_plan(
    model: torch.nn.Module,
    *,
    freeze_plan: ShaftFreezePlan,
) -> ShaftParameterSelectionPlan:
    trainable: list[str] = []
    frozen: list[str] = []
    for name, _parameter in model.named_parameters():
        if freeze_plan.should_train_name(name):
            trainable.append(name)
        else:
            frozen.append(name)
    return ShaftParameterSelectionPlan(
        trainable_parameter_names=tuple(trainable),
        frozen_parameter_names=tuple(frozen),
    )


def _build_adapter_plan(
    model: torch.nn.Module,
    *,
    finetune: FinetuneConfig,
    model_adapter: ShaftModelAdapter,
    freeze_plan: ShaftFreezePlan,
) -> ShaftAdapterFinetunePlan:
    resolved_target_modules = model_adapter.resolve_target_modules(list(finetune.target_modules))
    filtered_target_modules = resolve_adapter_target_modules(
        model,
        resolved_target_modules,
        plan=freeze_plan,
    )
    modules_to_save = resolve_adapter_modules_to_save(
        model,
        plan=freeze_plan,
        target_modules=filtered_target_modules,
    )
    target_tuple = _tuple_from_names(filtered_target_modules)
    modules_to_save_tuple = _tuple_from_names(modules_to_save)
    return ShaftAdapterFinetunePlan(
        resolved_target_modules=target_tuple,
        modules_to_save=modules_to_save_tuple,
        peft_signature=ShaftPeftSignature(
            target_modules=target_tuple,
            modules_to_save=modules_to_save_tuple,
            r=int(finetune.lora_r),
            lora_alpha=int(finetune.lora_alpha),
            lora_bias=str(finetune.lora_bias).strip().lower(),
            use_rslora=bool(finetune.use_rslora),
            use_dora=bool(str(finetune.mode).strip().lower() == "dora"),
        ),
    )


def build_resolved_finetune_plan(
    model: torch.nn.Module,
    finetune: FinetuneConfig,
    *,
    model_adapter: ShaftModelAdapter,
) -> ShaftResolvedFinetunePlan:
    mode = str(finetune.mode).strip().lower()
    freeze_plan = build_freeze_plan(model_adapter=model_adapter, finetune=finetune)
    parameter_plan = _build_parameter_selection_plan(model, freeze_plan=freeze_plan)
    adapter_plan = None
    if mode in {"lora", "dora", "qlora"}:
        adapter_plan = _build_adapter_plan(
            model,
            finetune=finetune,
            model_adapter=model_adapter,
            freeze_plan=freeze_plan,
        )
    return ShaftResolvedFinetunePlan(
        mode=mode,
        freeze_plan=freeze_plan,
        parameter_plan=parameter_plan,
        adapter_plan=adapter_plan,
    )
