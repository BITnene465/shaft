from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

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


def _is_explicit_target_modules(values: tuple[str, ...]) -> bool:
    return bool(values and values != ("auto",))


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

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            asdict(self),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ShaftFreezePreview:
    mode: str
    frozen_groups: tuple[str, ...] = ()
    frozen_prefixes: tuple[str, ...] = ()
    frozen_regex: str | None = None
    trainable_prefixes: tuple[str, ...] = ()
    trainable_regex: str | None = None
    target_modules_input: tuple[str, ...] = ()
    explicit_target_modules: bool = False
    policy_target_modules: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ShaftResolvedFreezeSummary:
    mode: str
    total_params: int
    trainable_params: int
    frozen_params: int
    trainable_ratio: float
    frozen_groups: tuple[str, ...] = ()
    frozen_prefixes: tuple[str, ...] = ()
    frozen_regex: str | None = None
    trainable_prefixes: tuple[str, ...] = ()
    trainable_regex: str | None = None
    target_modules_input: tuple[str, ...] = ()
    explicit_target_modules: bool = False
    policy_target_modules: tuple[str, ...] = ()
    resolved_target_modules: tuple[str, ...] = ()
    modules_to_save: tuple[str, ...] = ()
    sample_trainable_parameters: tuple[str, ...] = ()
    sample_frozen_parameters: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_log_dict(self) -> dict[str, object]:
        payload = self.to_dict()
        payload["trainable_ratio"] = round(self.trainable_ratio, 4)
        resolved_targets = tuple(payload.pop("resolved_target_modules"))
        payload["resolved_target_module_count"] = len(resolved_targets)
        payload["sample_resolved_target_modules"] = resolved_targets[:8]
        return payload


FINETUNE_SUMMARY_FILENAME = "shaft_finetune_summary.json"


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


def build_freeze_preview(
    finetune: FinetuneConfig,
    *,
    model_adapter: ShaftModelAdapter,
) -> ShaftFreezePreview:
    freeze_plan = build_freeze_plan(model_adapter=model_adapter, finetune=finetune)
    target_modules_input = _tuple_from_names(list(finetune.target_modules))
    policy_target_modules = _tuple_from_names(
        model_adapter.resolve_target_modules(list(finetune.target_modules))
    )
    return ShaftFreezePreview(
        mode=str(finetune.mode).strip().lower(),
        frozen_groups=freeze_plan.frozen_groups,
        frozen_prefixes=freeze_plan.frozen_prefixes,
        frozen_regex=freeze_plan.frozen_regex,
        trainable_prefixes=freeze_plan.trainable_prefixes,
        trainable_regex=freeze_plan.trainable_regex,
        target_modules_input=target_modules_input,
        explicit_target_modules=_is_explicit_target_modules(target_modules_input),
        policy_target_modules=policy_target_modules,
    )


def summarize_resolved_finetune_plan(
    model: torch.nn.Module,
    *,
    finetune: FinetuneConfig,
    plan: ShaftResolvedFinetunePlan,
    model_adapter: ShaftModelAdapter,
    sample_limit: int = 8,
) -> ShaftResolvedFreezeSummary:
    total_params, trainable_params = _count_parameters(model)
    frozen_params = total_params - trainable_params
    target_modules_input = _tuple_from_names(list(finetune.target_modules))
    policy_target_modules = _tuple_from_names(
        model_adapter.resolve_target_modules(list(finetune.target_modules))
    )
    actual_trainable_names: list[str] = []
    actual_frozen_names: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            actual_trainable_names.append(name)
        else:
            actual_frozen_names.append(name)
    return ShaftResolvedFreezeSummary(
        mode=plan.mode,
        total_params=total_params,
        trainable_params=trainable_params,
        frozen_params=frozen_params,
        trainable_ratio=(float(trainable_params) / float(total_params)) if total_params else 0.0,
        frozen_groups=plan.freeze_plan.frozen_groups,
        frozen_prefixes=plan.freeze_plan.frozen_prefixes,
        frozen_regex=plan.freeze_plan.frozen_regex,
        trainable_prefixes=plan.freeze_plan.trainable_prefixes,
        trainable_regex=plan.freeze_plan.trainable_regex,
        target_modules_input=target_modules_input,
        explicit_target_modules=_is_explicit_target_modules(target_modules_input),
        policy_target_modules=policy_target_modules,
        resolved_target_modules=(
            plan.adapter_plan.resolved_target_modules if plan.adapter_plan is not None else ()
        ),
        modules_to_save=(
            plan.adapter_plan.modules_to_save if plan.adapter_plan is not None else ()
        ),
        sample_trainable_parameters=tuple(actual_trainable_names[:sample_limit]),
        sample_frozen_parameters=tuple(actual_frozen_names[:sample_limit]),
    )


def resolved_finetune_summary_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / FINETUNE_SUMMARY_FILENAME


def write_resolved_finetune_summary(
    output_dir: str | Path,
    summary: ShaftResolvedFreezeSummary,
) -> Path:
    path = resolved_finetune_summary_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)
    return path


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
