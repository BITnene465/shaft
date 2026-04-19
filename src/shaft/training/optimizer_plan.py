from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import torch
from transformers import TrainingArguments

from shaft.model.finetune_plan import ShaftResolvedFinetunePlan
from shaft.model.types import ShaftModelAdapter

_ADAPTER_MODES = {"lora", "dora", "qlora"}
_LORA_PARAM_MARKERS = (
    ".lora_a.",
    ".lora_b.",
    ".lora_embedding_a.",
    ".lora_embedding_b.",
    ".lora_magnitude_vector.",
)
_MODULES_TO_SAVE_PATTERN = re.compile(r"\.modules_to_save(?:\.[^.]+)?")
_BASE_MODEL_PREFIX = "base_model.model."


@dataclass(frozen=True)
class ShaftOptimizerParamGroup:
    logical_group: str
    decay: bool
    lr: float
    weight_decay: float
    parameter_names: tuple[str, ...] = ()
    parameters: tuple[torch.nn.Parameter, ...] = field(default_factory=tuple, repr=False)

    def to_optimizer_group(self) -> dict[str, Any]:
        return {
            "params": list(self.parameters),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
        }


@dataclass(frozen=True)
class ShaftResolvedOptimizerPlan:
    groups: tuple[ShaftOptimizerParamGroup, ...]

    def to_optimizer_groups(self) -> list[dict[str, Any]]:
        return [group.to_optimizer_group() for group in self.groups]


def _normalize_runtime_parameter_name(name: str) -> str:
    normalized = str(name).strip()
    while normalized.startswith(_BASE_MODEL_PREFIX):
        normalized = normalized[len(_BASE_MODEL_PREFIX) :]
    normalized = _MODULES_TO_SAVE_PATTERN.sub("", normalized)
    normalized = normalized.replace("..", ".")
    return normalized.lstrip(".")


def _is_lora_parameter_name(name: str) -> bool:
    normalized = str(name).strip().lower()
    return any(marker in normalized for marker in _LORA_PARAM_MARKERS)


def _is_modules_to_save_parameter_name(name: str) -> bool:
    return ".modules_to_save." in str(name)


def _is_no_decay_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    return parameter.ndim <= 1 or str(name).endswith(".bias")


def _resolve_logical_group(
    name: str,
    *,
    finetune_plan: ShaftResolvedFinetunePlan | None,
    model_adapter: ShaftModelAdapter | None,
) -> str:
    mode = str(finetune_plan.mode).strip().lower() if finetune_plan is not None else ""
    if mode in _ADAPTER_MODES:
        if _is_lora_parameter_name(name):
            return "lora_params"
        if _is_modules_to_save_parameter_name(name):
            return "modules_to_save"
    if model_adapter is None:
        return "default"
    normalized_name = _normalize_runtime_parameter_name(name)
    resolved = model_adapter.module_groups.resolve_group_for_name(normalized_name)
    return resolved or "default"


def build_resolved_optimizer_plan(
    *,
    model: torch.nn.Module,
    args: TrainingArguments,
    finetune_plan: ShaftResolvedFinetunePlan | None = None,
    model_adapter: ShaftModelAdapter | None = None,
    param_group_lrs: dict[str, float] | None = None,
) -> ShaftResolvedOptimizerPlan:
    configured_lrs = {str(key).strip().lower(): float(value) for key, value in dict(param_group_lrs or {}).items()}
    grouped_parameters: dict[tuple[str, bool], list[tuple[str, torch.nn.Parameter]]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        logical_group = _resolve_logical_group(
            name,
            finetune_plan=finetune_plan,
            model_adapter=model_adapter,
        )
        decay = not _is_no_decay_parameter(name, parameter)
        grouped_parameters.setdefault((logical_group, decay), []).append((name, parameter))

    if not grouped_parameters:
        raise ValueError("No trainable parameters found for optimizer creation.")

    resolved_groups: list[ShaftOptimizerParamGroup] = []
    default_lr = float(args.learning_rate)
    default_weight_decay = float(args.weight_decay)
    for (logical_group, decay), items in grouped_parameters.items():
        lr = configured_lrs.get(logical_group, default_lr)
        weight_decay = default_weight_decay if decay else 0.0
        resolved_groups.append(
            ShaftOptimizerParamGroup(
                logical_group=logical_group,
                decay=decay,
                lr=float(lr),
                weight_decay=float(weight_decay),
                parameter_names=tuple(name for name, _ in items),
                parameters=tuple(parameter for _, parameter in items),
            )
        )
    return ShaftResolvedOptimizerPlan(groups=tuple(resolved_groups))
