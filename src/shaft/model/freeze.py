from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from shaft.config import FinetuneConfig

from .types import ModelModuleGroups, ShaftModelAdapter


def _dedupe(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _matches_prefix(name: str, prefix: str) -> bool:
    normalized_name = str(name).strip()
    normalized_prefix = str(prefix).strip()
    return bool(normalized_prefix) and (
        normalized_name == normalized_prefix or normalized_name.startswith(f"{normalized_prefix}.")
    )


def _matches_prefixes(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(_matches_prefix(name, prefix) for prefix in prefixes)


def _matches_regex(name: str, pattern: re.Pattern[str] | None) -> bool:
    return bool(pattern is not None and pattern.search(name))


@dataclass(frozen=True)
class ShaftFreezeSpec:
    groups: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    regex: str | None = None
    trainable_prefixes: tuple[str, ...] = ()
    trainable_regex: str | None = None


@dataclass(frozen=True)
class ShaftFreezePlan:
    module_groups: ModelModuleGroups
    frozen_groups: tuple[str, ...] = ()
    frozen_prefixes: tuple[str, ...] = ()
    frozen_regex: str | None = None
    trainable_prefixes: tuple[str, ...] = ()
    trainable_regex: str | None = None

    def compile_frozen_regex(self) -> re.Pattern[str] | None:
        return re.compile(self.frozen_regex) if self.frozen_regex else None

    def compile_trainable_regex(self) -> re.Pattern[str] | None:
        return re.compile(self.trainable_regex) if self.trainable_regex else None

    def matches_frozen_rule(self, name: str) -> bool:
        resolved_group = self.module_groups.resolve_group_for_name(name)
        return (
            (resolved_group is not None and resolved_group in self.frozen_groups)
            or _matches_prefixes(name, self.frozen_prefixes)
            or _matches_regex(name, self.compile_frozen_regex())
        )

    def matches_trainable_override(self, name: str) -> bool:
        return _matches_prefixes(name, self.trainable_prefixes) or _matches_regex(
            name,
            self.compile_trainable_regex(),
        )

    def should_train_name(self, name: str) -> bool:
        trainable = True
        if self.matches_frozen_rule(name):
            trainable = False
        if self.matches_trainable_override(name):
            trainable = True
        return trainable

    def filter_module_names(self, module_names: list[str]) -> list[str]:
        filtered: list[str] = []
        for name in module_names:
            keep = True
            if self.matches_frozen_rule(name):
                keep = False
            if self.matches_trainable_override(name):
                keep = True
            if keep:
                filtered.append(name)
        return list(dict.fromkeys(filtered))


def build_freeze_spec(finetune: FinetuneConfig) -> ShaftFreezeSpec:
    freeze = finetune.freeze
    return ShaftFreezeSpec(
        groups=_dedupe(list(freeze.groups)),
        prefixes=_dedupe(list(freeze.prefixes)),
        regex=freeze.regex,
        trainable_prefixes=_dedupe(list(freeze.trainable_prefixes)),
        trainable_regex=freeze.trainable_regex,
    )


def build_freeze_plan(*, model_adapter: ShaftModelAdapter, finetune: FinetuneConfig) -> ShaftFreezePlan:
    spec = build_freeze_spec(finetune)
    frozen_prefixes = _dedupe(list(spec.prefixes))
    return ShaftFreezePlan(
        module_groups=model_adapter.module_groups,
        frozen_groups=spec.groups,
        frozen_prefixes=frozen_prefixes,
        frozen_regex=spec.regex,
        trainable_prefixes=spec.trainable_prefixes,
        trainable_regex=spec.trainable_regex,
    )


def apply_full_freeze(model: torch.nn.Module, plan: ShaftFreezePlan) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(plan.should_train_name(name))


def _is_linear_module(module: torch.nn.Module) -> bool:
    return isinstance(module, torch.nn.Linear)


def _find_all_linear_module_names(model: torch.nn.Module) -> list[str]:
    ignore_suffixes = ("lm_head", "score", "v_head", "classifier", "lora_A", "lora_B", "base_layer")
    names: list[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if any(part in name for part in ignore_suffixes):
            continue
        if _is_linear_module(module):
            names.append(name)
    return names


def _parameter_global_shape(parameter: torch.nn.Parameter) -> tuple[int, ...]:
    """Return the logical shape when ZeRO-3 has materialized an empty shard."""

    deepspeed_shape = getattr(parameter, "ds_shape", None)
    if deepspeed_shape is not None:
        return tuple(int(dimension) for dimension in deepspeed_shape)
    return tuple(int(dimension) for dimension in parameter.shape)


def resolve_adapter_target_modules(
    model: torch.nn.Module,
    target_modules: list[str],
    *,
    plan: ShaftFreezePlan,
) -> list[str] | str:
    normalized = list(target_modules)
    if not normalized:
        return []
    resolved: list[str] = []
    for target in normalized:
        if target == "all-linear":
            resolved.extend(
                plan.filter_module_names(_find_all_linear_module_names(model))
            )
        else:
            # Explicit module targets remain authoritative even when they overlap
            # a freeze group. Only the model-owned all-linear expansion is filtered.
            resolved.append(target)
    resolved = list(dict.fromkeys(resolved))
    if not resolved:
        raise ValueError("No adapter target modules remain after applying freeze filters.")
    return resolved


def resolve_adapter_target_parameters(
    model: torch.nn.Module,
    target_parameters: list[str],
    *,
    plan: ShaftFreezePlan,
) -> list[str]:
    resolved: list[str] = []
    named_parameters = tuple(model.named_parameters())
    for requested in target_parameters:
        normalized = str(requested).strip()
        if not normalized:
            continue
        matches = [
            (name, parameter)
            for name, parameter in named_parameters
            if name == normalized or name.endswith(f".{normalized}")
        ]
        if not matches:
            raise ValueError(
                f"Adapter target parameter {normalized!r} does not match the model."
            )
        for name, parameter in matches:
            if not plan.should_train_name(name):
                continue
            parameter_shape = _parameter_global_shape(parameter)
            if len(parameter_shape) not in {2, 3}:
                raise ValueError(
                    f"Adapter target parameter {name!r} must be 2-D or 3-D; "
                    f"got shape={parameter_shape}."
                )
            resolved.append(name)
    return list(dict.fromkeys(resolved))


def resolve_adapter_modules_to_save(
    model: torch.nn.Module,
    *,
    plan: ShaftFreezePlan,
    target_modules: list[str] | str,
) -> list[str]:
    target_names = {str(name) for name in (target_modules if isinstance(target_modules, list) else [target_modules])}
    modules_to_save: list[str] = []
    for name, _module in model.named_modules():
        if not name or name in target_names:
            continue
        if not plan.matches_trainable_override(name):
            continue
        modules_to_save.append(name)
    return list(dict.fromkeys(modules_to_save))


def summarize_trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
