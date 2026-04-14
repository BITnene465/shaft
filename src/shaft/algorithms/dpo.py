from __future__ import annotations

from dataclasses import dataclass

from .registry import register_algorithm


@dataclass
@register_algorithm("dpo")
class DPOAlgorithm:
    name: str = "dpo"

    def build_trainer(self, **kwargs):
        raise NotImplementedError("DPOAlgorithm is reserved for phase-2 integration.")
