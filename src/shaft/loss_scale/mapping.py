from __future__ import annotations

from shaft.plugins import Registry

from .base import ALL_BASE_STRATEGY, ShaftLossScale

LOSS_SCALE_REGISTRY: Registry[type[ShaftLossScale]] = Registry("loss_scale")


def register_loss_scale(name: str):
    return LOSS_SCALE_REGISTRY.register(name)


@register_loss_scale("base")
class BaseLossScale(ShaftLossScale):
    pass


def build_loss_scale(loss_scale: str) -> ShaftLossScale:
    normalized = str(loss_scale).strip().lower()
    if not normalized:
        normalized = "default"
    parts = normalized.split("+", 1)
    if len(parts) == 1:
        if parts[0] in ALL_BASE_STRATEGY:
            base_strategy, scale_name = parts[0], "base"
        else:
            base_strategy, scale_name = "default", parts[0]
    else:
        base_strategy, scale_name = parts
    scale_cls = LOSS_SCALE_REGISTRY.get(scale_name)
    return scale_cls(base_strategy=base_strategy)
