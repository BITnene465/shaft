from .base import ALL_BASE_STRATEGY, ShaftLossScale, ShaftLossScaleSpec
from .mapping import LOSS_SCALE_REGISTRY, build_loss_scale, register_loss_scale

__all__ = [
    "ALL_BASE_STRATEGY",
    "LOSS_SCALE_REGISTRY",
    "ShaftLossScale",
    "ShaftLossScaleSpec",
    "build_loss_scale",
    "register_loss_scale",
]
