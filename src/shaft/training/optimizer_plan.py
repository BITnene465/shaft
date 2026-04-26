from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
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


@dataclass(frozen=True)
class ShaftResolvedOptimizerGroupSummary:
    logical_group: str
    decay: bool
    lr: float
    weight_decay: float
    num_parameters: int
    num_tensors: int
    sample_parameter_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShaftResolvedOptimizerSummary:
    total_trainable_params: int
    group_count: int
    groups: tuple[ShaftResolvedOptimizerGroupSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "total_trainable_params": int(self.total_trainable_params),
            "group_count": int(self.group_count),
            "groups": [
                {
                    "logical_group": item.logical_group,
                    "decay": item.decay,
                    "lr": item.lr,
                    "weight_decay": item.weight_decay,
                    "num_parameters": item.num_parameters,
                    "num_tensors": item.num_tensors,
                    "sample_parameter_names": list(item.sample_parameter_names),
                }
                for item in self.groups
            ],
        }


OPTIMIZER_SUMMARY_FILENAME = "shaft_optimizer_summary.json"


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


def _matches_no_decay_name_pattern(name: str, no_decay_name_patterns: list[str] | None) -> bool:
    if not no_decay_name_patterns:
        return False
    normalized_name = _normalize_runtime_parameter_name(name).lower()
    return any(normalized_name.endswith(str(pattern).strip().lower()) for pattern in no_decay_name_patterns)


def _is_no_decay_parameter(
    name: str,
    parameter: torch.nn.Parameter,
    *,
    no_decay_name_patterns: list[str] | None = None,
) -> bool:
    return (
        parameter.ndim <= 1
        or str(name).endswith(".bias")
        or _matches_no_decay_name_pattern(name, no_decay_name_patterns)
    )


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
    no_decay_name_patterns: list[str] | None = None,
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
        decay = not _is_no_decay_parameter(
            name,
            parameter,
            no_decay_name_patterns=no_decay_name_patterns,
        )
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


def summarize_resolved_optimizer_plan(
    plan: ShaftResolvedOptimizerPlan,
    *,
    sample_limit: int = 5,
) -> ShaftResolvedOptimizerSummary:
    group_summaries: list[ShaftResolvedOptimizerGroupSummary] = []
    total_trainable_params = 0
    for group in plan.groups:
        num_parameters = sum(int(parameter.numel()) for parameter in group.parameters)
        total_trainable_params += num_parameters
        group_summaries.append(
            ShaftResolvedOptimizerGroupSummary(
                logical_group=group.logical_group,
                decay=group.decay,
                lr=float(group.lr),
                weight_decay=float(group.weight_decay),
                num_parameters=int(num_parameters),
                num_tensors=len(group.parameters),
                sample_parameter_names=tuple(group.parameter_names[:sample_limit]),
            )
        )
    return ShaftResolvedOptimizerSummary(
        total_trainable_params=int(total_trainable_params),
        group_count=len(group_summaries),
        groups=tuple(group_summaries),
    )


def resolved_optimizer_summary_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / OPTIMIZER_SUMMARY_FILENAME


def write_resolved_optimizer_summary(
    output_dir: str | Path,
    summary: ShaftResolvedOptimizerSummary,
) -> Path:
    path = resolved_optimizer_summary_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path
