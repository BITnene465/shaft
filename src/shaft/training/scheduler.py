from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_inverse_sqrt_schedule,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from shaft.plugins import Registry

SchedulerBuilder = Callable[..., torch.optim.lr_scheduler.LRScheduler]
SCHEDULER_REGISTRY: Registry[SchedulerBuilder] = Registry("scheduler")


def register_scheduler(name: str):
    return SCHEDULER_REGISTRY.register(name)


@register_scheduler("cosine")
def build_cosine_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    **_: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_warmup_steps),
        num_training_steps=int(num_training_steps),
        num_cycles=float(num_cycles),
    )


@register_scheduler("linear")
def build_linear_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    **_: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_warmup_steps),
        num_training_steps=int(num_training_steps),
    )


@register_scheduler("cosine_with_restarts")
@register_scheduler("cosine_restarts")
def build_cosine_restarts_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 1.0,
    **_: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    return get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_warmup_steps),
        num_training_steps=int(num_training_steps),
        num_cycles=int(max(1, round(float(num_cycles)))),
    )


@register_scheduler("polynomial")
@register_scheduler("poly")
def build_polynomial_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    power: float = 1.0,
    **_: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    return get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_warmup_steps),
        num_training_steps=int(num_training_steps),
        power=float(power),
    )


@register_scheduler("constant")
def build_constant_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    **_: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    return get_constant_schedule(optimizer=optimizer)


@register_scheduler("constant_with_warmup")
def build_constant_with_warmup_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    **_: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    return get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_warmup_steps),
    )


@register_scheduler("inverse_sqrt")
def build_inverse_sqrt_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    **_: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    return get_inverse_sqrt_schedule(
        optimizer=optimizer,
        num_warmup_steps=int(num_warmup_steps),
    )


def build_scheduler(
    *,
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    power: float = 1.0,
) -> torch.optim.lr_scheduler.LRScheduler:
    normalized = str(scheduler_name).strip().lower()
    if normalized in {"auto", ""}:
        normalized = "cosine"
    builder = SCHEDULER_REGISTRY.get(normalized)
    return builder(
        optimizer=optimizer,
        num_warmup_steps=int(num_warmup_steps),
        num_training_steps=int(num_training_steps),
        num_cycles=float(num_cycles),
        power=float(power),
    )
