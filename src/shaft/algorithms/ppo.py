from __future__ import annotations

from dataclasses import dataclass

from .registry import register_algorithm


@dataclass
@register_algorithm("ppo")
class PPOAlgorithm:
    name: str = "ppo"

    def build_trainer(self, **kwargs):
        raise NotImplementedError("PPOAlgorithm is reserved for phase-3 integration.")
