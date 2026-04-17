from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import torch

from shaft.config import FinetuneConfig

from .types import ModelModuleGroups, ShaftModelAdapter


def _dedupe(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _matches_prefixes(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


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
    frozen_prefixes: tuple[str, ...] = ()
    frozen_regex: str | None = None
    trainable_prefixes: tuple[str, ...] = ()
    trainable_regex: str | None = None

    def compile_frozen_regex(self) -> re.Pattern[str] | None:
        return re.compile(self.frozen_regex) if self.frozen_regex else None

    def compile_trainable_regex(self) -> re.Pattern[str] | None:
        return re.compile(self.trainable_regex) if self.trainable_regex else None

    def matches_frozen_rule(self, name: str) -> bool:
        return _matches_prefixes(name, self.frozen_prefixes) or _matches_regex(name, self.compile_frozen_regex())

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
    group_prefixes: list[str] = []
    for group_name in spec.groups:
        group_prefixes.extend(model_adapter.module_groups.prefixes_for_group(group_name))
    frozen_prefixes = _dedupe(group_prefixes + list(spec.prefixes))
    return ShaftFreezePlan(
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


def resolve_adapter_target_modules(
    model: torch.nn.Module,
    target_modules: list[str],
    *,
    plan: ShaftFreezePlan,
) -> list[str] | str:
    normalized = list(target_modules)
    if normalized != ["all-linear"]:
        return normalized
    normalized = _find_all_linear_module_names(model)
    filtered = plan.filter_module_names(normalized)
    if not filtered:
        raise ValueError("No adapter target modules remain after applying freeze filters.")
    return filtered


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
