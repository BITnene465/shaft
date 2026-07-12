from __future__ import annotations

from dataclasses import dataclass

from .cost import ShaftSampleCost
from .mixing import ShaftSampleRef


@dataclass(frozen=True, slots=True)
class ShaftLocalMicroBatchPlan:
    """Small immutable value object shared by the bounded planner and diagnostics."""

    sample_refs: tuple[ShaftSampleRef, ...]
    sample_costs: tuple[ShaftSampleCost, ...]

    def __post_init__(self) -> None:
        if not self.sample_refs:
            raise ValueError("A local microbatch plan cannot be empty.")
        if len(self.sample_refs) != len(self.sample_costs):
            raise ValueError("Local microbatch refs and costs must have the same length.")

    @property
    def useful_llm_tokens(self) -> int:
        return sum(cost.llm_tokens for cost in self.sample_costs)

    @property
    def max_llm_tokens(self) -> int:
        return max(cost.llm_tokens for cost in self.sample_costs)

    @property
    def padded_llm_tokens(self) -> int:
        return len(self.sample_costs) * self.max_llm_tokens

    @property
    def supervised_tokens(self) -> int:
        return sum(cost.supervised_tokens for cost in self.sample_costs)

    @property
    def vision_patches(self) -> int:
        return sum(cost.vision_patches for cost in self.sample_costs)
