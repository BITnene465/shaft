from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ALL_BASE_STRATEGY = ["default", "last_round", "all"]


@dataclass(frozen=True)
class ShaftLossScaleSpec:
    base_strategy: Literal["default", "last_round", "all"] = "default"
    prefix_scale: float = 1.0
    target_scale: float = 1.0

    @property
    def is_binary(self) -> bool:
        return self.prefix_scale in {0.0, 1.0} and self.target_scale in {0.0, 1.0}


class ShaftLossScale:
    is_binary = True

    def __init__(self, base_strategy: Literal["default", "last_round", "all"] = "default") -> None:
        if base_strategy not in ALL_BASE_STRATEGY:
            raise ValueError(f"ALL_BASE_STRATEGY: {ALL_BASE_STRATEGY}, base_strategy: {base_strategy}")
        self.base_strategy = base_strategy

    def get_loss_scale(self, item: dict[str, Any]) -> ShaftLossScaleSpec:
        _ = item
        return ShaftLossScaleSpec(
            base_strategy=self.base_strategy,
            prefix_scale=1.0,
            target_scale=1.0,
        )

    def __call__(self, item: dict[str, Any]) -> ShaftLossScaleSpec:
        return self.get_loss_scale(item)

    @property
    def is_loss_scale_binary(self) -> bool:
        return bool(self.is_binary)
