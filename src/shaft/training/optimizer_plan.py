from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import TrainingArguments

from shaft.model.parameters import parameter_numel
from shaft.model.types import ShaftModelAdapter

_FSDP_WRAPPER_SEGMENT = "_fsdp_wrapped_module"
_PEFT_BASE_PREFIX = ("base_model", "model")
_MODULES_TO_SAVE_SEGMENT = "modules_to_save"
_LORA_LINEAR_WEIGHT_SEGMENTS = {
    "lora_A",
    "lora_B",
}
_LORA_EMBEDDING_SEGMENTS = {
    "lora_embedding_A",
    "lora_embedding_B",
}
_DORA_MAGNITUDE_SEGMENT = "lora_magnitude_vector"


@dataclass(frozen=True)
class ShaftOptimizerParamGroup:
    module_group: str
    decay: bool
    lr: float
    weight_decay: float
    raw_parameter_names: tuple[str, ...] = ()
    canonical_parameter_names: tuple[str, ...] = ()
    parameters: tuple[torch.nn.Parameter, ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        if len(self.raw_parameter_names) != len(self.canonical_parameter_names):
            raise ValueError("Optimizer raw and canonical parameter names must align.")
        if len(self.raw_parameter_names) != len(self.parameters):
            raise ValueError("Optimizer parameter names and tensors must align.")

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

    @property
    def fingerprint(self) -> str:
        payload = {
            "version": "shaft-resolved-optimizer-plan-v2",
            "groups": [
                {
                    "module_group": group.module_group,
                    "decay": bool(group.decay),
                    "lr": float(group.lr),
                    "weight_decay": float(group.weight_decay),
                    # Raw names differ across PEFT/FSDP wrappers. Canonical names
                    # bind the semantic parameters while preserving A/B/DoRA roles.
                    "canonical_parameter_names": list(group.canonical_parameter_names),
                }
                for group in self.groups
            ],
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def summary(self, *, sample_limit: int = 5) -> ShaftResolvedOptimizerSummary:
        return _summarize_plan(self, sample_limit=sample_limit)

    def to_log_dict(self, *, sample_limit: int = 5) -> dict[str, Any]:
        return self.summary(sample_limit=sample_limit).to_log_dict()


@dataclass(frozen=True)
class ShaftResolvedOptimizerGroupSummary:
    module_group: str
    decay: bool
    lr: float
    weight_decay: float
    num_parameters: int
    num_tensors: int
    sample_raw_parameter_names: tuple[str, ...] = ()
    sample_canonical_parameter_names: tuple[str, ...] = ()

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
                    "module_group": item.module_group,
                    "decay": item.decay,
                    "lr": item.lr,
                    "weight_decay": item.weight_decay,
                    "num_parameters": item.num_parameters,
                    "num_tensors": item.num_tensors,
                    "sample_raw_parameter_names": list(item.sample_raw_parameter_names),
                    "sample_canonical_parameter_names": list(
                        item.sample_canonical_parameter_names
                    ),
                }
                for item in self.groups
            ],
        }


OPTIMIZER_SUMMARY_FILENAME = "shaft_optimizer_summary.json"


def canonicalize_optimizer_parameter_name(name: str) -> str:
    """Normalize only confirmed FSDP/PEFT wrapper grammar.

    The result preserves the real model hierarchy and PEFT tensor role. It only
    removes FSDP's reserved wrapper segment, PEFT's leading base-model wrapper,
    runtime adapter namespaces, and the exact ``modules_to_save`` wrapper.
    """

    raw_name = str(name).strip()
    if not raw_name:
        raise ValueError("Optimizer parameter name must not be empty.")
    parts = raw_name.split(".")
    if any(not part for part in parts):
        raise ValueError(f"Optimizer parameter name contains an empty segment: {name!r}.")

    parts = [part for part in parts if part != _FSDP_WRAPPER_SEGMENT]
    while tuple(parts[:2]) == _PEFT_BASE_PREFIX:
        parts = parts[2:]
    if not parts:
        raise ValueError(f"Optimizer parameter name contains only wrapper segments: {name!r}.")

    module_wrapper_indexes = [
        index for index, part in enumerate(parts) if part == _MODULES_TO_SAVE_SEGMENT
    ]
    if len(module_wrapper_indexes) > 1:
        raise ValueError(f"Optimizer parameter name has nested modules_to_save wrappers: {name!r}.")
    if module_wrapper_indexes:
        wrapper_index = module_wrapper_indexes[0]
        if wrapper_index == 0 or wrapper_index + 2 >= len(parts):
            raise ValueError(f"Incomplete PEFT wrapper in optimizer parameter name: {name!r}.")
        # Remove both the wrapper marker and its single adapter namespace.
        parts = parts[:wrapper_index] + parts[wrapper_index + 2 :]

    peft_role_indexes = [
        index
        for index, part in enumerate(parts)
        if part in _LORA_LINEAR_WEIGHT_SEGMENTS
        or part in _LORA_EMBEDDING_SEGMENTS
        or part == _DORA_MAGNITUDE_SEGMENT
    ]
    if len(peft_role_indexes) > 1:
        raise ValueError(f"Optimizer parameter name has multiple PEFT roles: {name!r}.")
    if peft_role_indexes:
        role_index = peft_role_indexes[0]
        role = parts[role_index]
        tail = parts[role_index + 1 :]
        if role in _LORA_LINEAR_WEIGHT_SEGMENTS:
            if len(tail) == 2 and tail[-1] == "weight":
                parts = parts[: role_index + 1] + ["weight"]
            elif tail != ["weight"]:
                raise ValueError(f"Incomplete PEFT wrapper in optimizer parameter name: {name!r}.")
        elif role in _LORA_EMBEDDING_SEGMENTS:
            if len(tail) == 1:
                parts = parts[: role_index + 1]
            elif tail:
                raise ValueError(f"Incomplete PEFT wrapper in optimizer parameter name: {name!r}.")
        elif not tail:
            # PEFT serializes DoRA magnitude keys without the terminal ``weight``.
            parts.append("weight")
        elif len(tail) == 2 and tail[-1] == "weight":
            parts = parts[: role_index + 1] + ["weight"]
        elif tail != ["weight"]:
            raise ValueError(f"Incomplete PEFT wrapper in optimizer parameter name: {name!r}.")

    canonical = ".".join(parts)
    if not canonical:
        raise ValueError(f"Optimizer parameter name resolved to an empty path: {name!r}.")
    return canonical


def _matches_no_decay_name_pattern(
    canonical_name: str,
    no_decay_name_patterns: list[str] | None,
) -> bool:
    if not no_decay_name_patterns:
        return False
    normalized_name = canonical_name.lower()
    return any(
        normalized_name.endswith(str(pattern).strip().lower())
        for pattern in no_decay_name_patterns
    )


def _is_no_decay_parameter(
    canonical_name: str,
    parameter: torch.nn.Parameter,
    *,
    no_decay_name_patterns: list[str] | None = None,
) -> bool:
    return (
        _parameter_ndim(parameter) <= 1
        or canonical_name.endswith(".bias")
        or _matches_no_decay_name_pattern(canonical_name, no_decay_name_patterns)
    )


def _known_prefixes(model_adapter: ShaftModelAdapter) -> dict[str, tuple[str, ...]]:
    groups = model_adapter.module_groups
    return {name: groups.prefixes_for_group(name) for name in groups.group_names()}


def _diagnostic_finetune_mode(model: torch.nn.Module) -> str:
    peft_configs = getattr(model, "peft_config", None)
    if not peft_configs:
        return "full"
    configs = peft_configs.values() if isinstance(peft_configs, dict) else (peft_configs,)
    if any(bool(getattr(config, "use_dora", False)) for config in configs):
        return "dora"
    candidates = (model, getattr(model, "base_model", None), getattr(model, "model", None))
    if any(bool(getattr(candidate, "is_loaded_in_4bit", False)) for candidate in candidates):
        return "qlora"
    return "lora"


def _format_group_parameter_counts(
    grouped_parameters: dict[
        tuple[str, bool],
        list[tuple[str, str, torch.nn.Parameter]],
    ],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for (module_group, _decay), items in grouped_parameters.items():
        counts[module_group] = counts.get(module_group, 0) + sum(
            parameter_numel(parameter) for _raw, _canonical, parameter in items
        )
    return counts


def build_resolved_optimizer_plan(
    *,
    model: torch.nn.Module,
    args: TrainingArguments,
    model_adapter: ShaftModelAdapter | None = None,
    param_group_lrs: dict[str, float] | None = None,
    no_decay_name_patterns: list[str] | None = None,
) -> ShaftResolvedOptimizerPlan:
    configured_lrs = {
        str(key).strip().lower(): float(value)
        for key, value in dict(param_group_lrs or {}).items()
    }
    invalid_lr_values = {
        name: value
        for name, value in configured_lrs.items()
        if not math.isfinite(value) or value <= 0
    }
    if invalid_lr_values:
        raise ValueError(
            "Optimizer structural group learning rates must be finite and positive; "
            f"invalid_lrs={invalid_lr_values!r}."
        )
    structural_group_names = (
        model_adapter.module_groups.group_names()
        if model_adapter is not None
        else ()
    )
    invalid_configured_groups = sorted(set(configured_lrs) - set(structural_group_names))
    if invalid_configured_groups:
        raise ValueError(
            "Differential optimizer LR requires model module-group metadata and only "
            "structural group keys; "
            f"configured_groups={invalid_configured_groups!r}, "
            f"known_groups={structural_group_names!r}."
        )

    structured = bool(
        model_adapter is not None and model_adapter.module_groups.has_structural_metadata
    )
    if configured_lrs and not structured:
        raise ValueError(
            "Differential optimizer LR requires model module-group metadata; "
            f"configured_lrs={configured_lrs!r}."
        )

    grouped_parameters: dict[
        tuple[str, bool],
        list[tuple[str, str, torch.nn.Parameter]],
    ] = {}
    for raw_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        canonical_name = canonicalize_optimizer_parameter_name(raw_name)
        module_group = None
        if structured:
            assert model_adapter is not None
            module_group = model_adapter.module_groups.resolve_group_for_name(canonical_name)
            if module_group is None:
                raise ValueError(
                    "Trainable parameter cannot be assigned to a structural optimizer group; "
                    f"raw_name={raw_name!r}, canonical_name={canonical_name!r}, "
                    f"model_type={model_adapter.model_type!r}, "
                    f"finetune_mode={_diagnostic_finetune_mode(model)!r}, "
                    f"known_prefixes={_known_prefixes(model_adapter)!r}."
                )
        else:
            module_group = "default"
        decay = not _is_no_decay_parameter(
            canonical_name,
            parameter,
            no_decay_name_patterns=no_decay_name_patterns,
        )
        grouped_parameters.setdefault((module_group, decay), []).append(
            (raw_name, canonical_name, parameter)
        )

    if not grouped_parameters:
        raise ValueError("No trainable parameters found for optimizer creation.")

    group_parameter_counts = _format_group_parameter_counts(grouped_parameters)
    for configured_group, configured_lr in configured_lrs.items():
        if configured_group in group_parameter_counts:
            continue
        model_type = model_adapter.model_type if model_adapter is not None else type(model).__name__
        raise ValueError(
            "Configured optimizer LR group has no trainable parameters; "
            f"configured_group={configured_group!r}, configured_lr={configured_lr!r}, "
            f"trainable_groups={group_parameter_counts!r}, model_type={model_type!r}, "
            f"finetune_mode={_diagnostic_finetune_mode(model)!r}."
        )

    resolved_groups: list[ShaftOptimizerParamGroup] = []
    default_lr = float(args.learning_rate)
    default_weight_decay = float(args.weight_decay)
    for (module_group, decay), items in grouped_parameters.items():
        lr = configured_lrs.get(module_group, default_lr)
        weight_decay = default_weight_decay if decay else 0.0
        resolved_groups.append(
            ShaftOptimizerParamGroup(
                module_group=module_group,
                decay=decay,
                lr=float(lr),
                weight_decay=float(weight_decay),
                raw_parameter_names=tuple(raw_name for raw_name, _canonical, _parameter in items),
                canonical_parameter_names=tuple(
                    canonical for _raw_name, canonical, _parameter in items
                ),
                parameters=tuple(parameter for _raw, _canonical, parameter in items),
            )
        )
    return ShaftResolvedOptimizerPlan(groups=tuple(resolved_groups))


def _summarize_plan(
    plan: ShaftResolvedOptimizerPlan,
    *,
    sample_limit: int,
) -> ShaftResolvedOptimizerSummary:
    group_summaries: list[ShaftResolvedOptimizerGroupSummary] = []
    total_trainable_params = 0
    for group in plan.groups:
        num_parameters = sum(parameter_numel(parameter) for parameter in group.parameters)
        total_trainable_params += num_parameters
        group_summaries.append(
            ShaftResolvedOptimizerGroupSummary(
                module_group=group.module_group,
                decay=group.decay,
                lr=float(group.lr),
                weight_decay=float(group.weight_decay),
                num_parameters=int(num_parameters),
                num_tensors=len(group.parameters),
                sample_raw_parameter_names=tuple(group.raw_parameter_names[:sample_limit]),
                sample_canonical_parameter_names=tuple(
                    group.canonical_parameter_names[:sample_limit]
                ),
            )
        )
    return ShaftResolvedOptimizerSummary(
        total_trainable_params=int(total_trainable_params),
        group_count=len(group_summaries),
        groups=tuple(group_summaries),
    )


def summarize_resolved_optimizer_plan(
    plan: ShaftResolvedOptimizerPlan,
    *,
    sample_limit: int = 5,
) -> ShaftResolvedOptimizerSummary:
    return plan.summary(sample_limit=sample_limit)


def _parameter_ndim(parameter: torch.nn.Parameter) -> int:
    deepspeed_shape = getattr(parameter, "ds_shape", None)
    if deepspeed_shape is not None:
        return len(tuple(deepspeed_shape))
    return int(parameter.ndim)


def resolved_optimizer_summary_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / OPTIMIZER_SUMMARY_FILENAME


def write_resolved_optimizer_summary(
    output_dir: str | Path,
    plan: ShaftResolvedOptimizerPlan,
) -> Path:
    path = resolved_optimizer_summary_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(plan.summary().to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)
    return path
