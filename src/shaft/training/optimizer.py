from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers import TrainingArguments
from transformers.optimization import Adafactor

from shaft.model.finetune_plan import ShaftResolvedFinetunePlan
from shaft.model.types import ShaftModelAdapter
from shaft.plugins import Registry

from .optimizer_plan import build_resolved_optimizer_plan
from .muon import Muon

OptimizerBuilder = Callable[..., torch.optim.Optimizer]
OPTIMIZER_REGISTRY: Registry[OptimizerBuilder] = Registry("optimizer")


def register_optimizer(name: str):
    return OPTIMIZER_REGISTRY.register(name)


@register_optimizer("adamw")
@register_optimizer("adamw_torch")
def build_adamw(
    *,
    grouped_params: list[dict[str, Any]],
    args: TrainingArguments,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    **_: Any,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        grouped_params,
        lr=float(args.learning_rate),
        betas=(float(adam_beta1), float(adam_beta2)),
        eps=float(adam_epsilon),
    )


@register_optimizer("sgd")
def build_sgd(
    *,
    grouped_params: list[dict[str, Any]],
    args: TrainingArguments,
    **_: Any,
) -> torch.optim.Optimizer:
    return torch.optim.SGD(grouped_params, lr=float(args.learning_rate), momentum=0.9)


@register_optimizer("adafactor")
def build_adafactor(
    *,
    grouped_params: list[dict[str, Any]],
    args: TrainingArguments,
    **_: Any,
) -> torch.optim.Optimizer:
    return Adafactor(
        grouped_params,
        lr=float(args.learning_rate),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )


@register_optimizer("muon")
def build_muon(
    *,
    grouped_params: list[dict[str, Any]],
    args: TrainingArguments,
    adam_epsilon: float,
    **_: Any,
) -> torch.optim.Optimizer:
    return Muon(
        grouped_params,
        lr=float(args.learning_rate),
        momentum=0.95,
        nesterov=True,
        eps=float(adam_epsilon),
    )


@register_optimizer("adam8bit")
@register_optimizer("paged_adamw_8bit")
def build_adam8bit(
    *,
    grouped_params: list[dict[str, Any]],
    args: TrainingArguments,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    **_: Any,
) -> torch.optim.Optimizer:
    try:
        import bitsandbytes as bnb  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency-gated branch
        raise ImportError(
            "optimizer 'adam8bit/paged_adamw_8bit' requires bitsandbytes. "
            "Install with `uv pip install -e \".[gpu]\"`."
        ) from exc
    optim_cls = bnb.optim.Adam8bit
    return optim_cls(
        grouped_params,
        lr=float(args.learning_rate),
        betas=(float(adam_beta1), float(adam_beta2)),
        eps=float(adam_epsilon),
    )


def build_optimizer(
    *,
    model: torch.nn.Module,
    args: TrainingArguments,
    optimizer_name: str,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    finetune_plan: ShaftResolvedFinetunePlan | None = None,
    model_adapter: ShaftModelAdapter | None = None,
    param_group_lrs: dict[str, float] | None = None,
) -> torch.optim.Optimizer:
    normalized = str(optimizer_name).strip().lower()
    grouped_params = build_resolved_optimizer_plan(
        model=model,
        args=args,
        finetune_plan=finetune_plan,
        model_adapter=model_adapter,
        param_group_lrs=param_group_lrs,
    ).to_optimizer_groups()
    builder = OPTIMIZER_REGISTRY.get(normalized)
    return builder(
        grouped_params=grouped_params,
        args=args,
        adam_beta1=float(adam_beta1),
        adam_beta2=float(adam_beta2),
        adam_epsilon=float(adam_epsilon),
    )
