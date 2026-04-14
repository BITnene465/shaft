from __future__ import annotations

import math
import random
from collections import defaultdict

from .dataset import SFTRecord
from .registry import MIX_STRATEGY_REGISTRY, register_mix_strategy


class MixedDatasetBuilder:
    def __init__(self, *, seed: int = 42) -> None:
        self.seed = int(seed)

    def build_indices(
        self,
        records_by_dataset: dict[str, list[SFTRecord]],
        dataset_weights: dict[str, float],
        *,
        strategy: str = "interleave_under",
        shuffle: bool = True,
    ) -> list[tuple[str, int]]:
        strategy = str(strategy).strip().lower()
        if strategy not in {"concat", "interleave_under", "interleave_over"}:
            raise ValueError(f"Unsupported mix strategy: {strategy!r}.")

        active: dict[str, list[int]] = {}
        for dataset_id, records in records_by_dataset.items():
            weight = float(dataset_weights.get(dataset_id, 1.0))
            if weight <= 0.0 or not records:
                continue
            active[dataset_id] = list(range(len(records)))
        if not active:
            raise ValueError("No active datasets for mixing.")

        rng = random.Random(self.seed)
        if shuffle:
            for values in active.values():
                rng.shuffle(values)

        normalized_weights = self._normalize_weights(active.keys(), dataset_weights)
        strategy_fn = MIX_STRATEGY_REGISTRY.get(strategy)
        return strategy_fn(self, active, normalized_weights, shuffle=shuffle, rng=rng)

    def _normalize_weights(self, keys: list[str] | set[str], weights: dict[str, float]) -> dict[str, float]:
        result = {k: max(float(weights.get(k, 1.0)), 0.0) for k in keys}
        total = sum(result.values())
        if total <= 0.0:
            uniform = 1.0 / float(len(result))
            return {k: uniform for k in result}
        return {k: v / total for k, v in result.items()}

    def _build_quotas(
        self,
        active: dict[str, list[int]],
        normalized_weights: dict[str, float],
        *,
        strategy: str,
    ) -> dict[str, int]:
        sizes = {k: len(v) for k, v in active.items()}
        if strategy == "interleave_under":
            base = min(sizes[k] / max(normalized_weights[k], 1e-12) for k in sizes)
            quotas = {k: min(sizes[k], int(math.floor(base * normalized_weights[k]))) for k in sizes}
        else:
            base = max(sizes[k] / max(normalized_weights[k], 1e-12) for k in sizes)
            quotas = {k: max(sizes[k], int(math.ceil(base * normalized_weights[k]))) for k in sizes}
        if sum(quotas.values()) <= 0:
            return sizes
        return quotas

    def _interleave(
        self,
        active: dict[str, list[int]],
        quotas: dict[str, int],
        normalized_weights: dict[str, float],
        *,
        shuffle: bool,
        rng: random.Random,
    ) -> list[tuple[str, int]]:
        resolution = 100
        cycle: list[str] = []
        for dataset_id in sorted(quotas):
            repeat = max(int(round(normalized_weights[dataset_id] * resolution)), 1)
            cycle.extend([dataset_id] * repeat)
        if shuffle:
            rng.shuffle(cycle)

        cursor = defaultdict(int)
        output: list[tuple[str, int]] = []
        remaining = dict(quotas)
        while sum(remaining.values()) > 0:
            progressed = False
            for dataset_id in cycle:
                if remaining.get(dataset_id, 0) <= 0:
                    continue
                idxs = active[dataset_id]
                if not idxs:
                    continue
                pos = cursor[dataset_id] % len(idxs)
                output.append((dataset_id, idxs[pos]))
                cursor[dataset_id] += 1
                remaining[dataset_id] -= 1
                progressed = True
                if sum(remaining.values()) <= 0:
                    break
            if not progressed:
                break
        return output


@register_mix_strategy("concat")
def mix_concat(
    builder: MixedDatasetBuilder,
    active: dict[str, list[int]],
    normalized_weights: dict[str, float],
    *,
    shuffle: bool,
    rng: random.Random,
) -> list[tuple[str, int]]:
    del builder, normalized_weights
    merged = []
    for dataset_id, indices in sorted(active.items(), key=lambda x: x[0]):
        merged.extend((dataset_id, i) for i in indices)
    if shuffle:
        rng.shuffle(merged)
    return merged


@register_mix_strategy("interleave_under")
def mix_interleave_under(
    builder: MixedDatasetBuilder,
    active: dict[str, list[int]],
    normalized_weights: dict[str, float],
    *,
    shuffle: bool,
    rng: random.Random,
) -> list[tuple[str, int]]:
    quotas = builder._build_quotas(active, normalized_weights, strategy="interleave_under")
    return builder._interleave(active, quotas, normalized_weights, shuffle=shuffle, rng=rng)


@register_mix_strategy("interleave_over")
def mix_interleave_over(
    builder: MixedDatasetBuilder,
    active: dict[str, list[int]],
    normalized_weights: dict[str, float],
    *,
    shuffle: bool,
    rng: random.Random,
) -> list[tuple[str, int]]:
    quotas = builder._build_quotas(active, normalized_weights, strategy="interleave_over")
    return builder._interleave(active, quotas, normalized_weights, shuffle=shuffle, rng=rng)
