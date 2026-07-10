from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import math

_MASK_64 = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & _MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return (value ^ (value >> 31)) & _MASK_64


def _affine_permute(position: int, size: int, *, seed: int) -> int:
    if size <= 1:
        return 0
    multiplier = int(_splitmix64(seed) % size) | 1
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 2) % size
        if multiplier == 0:
            multiplier = 1
    offset = int(_splitmix64(seed ^ 0xD1B54A32D192ED03) % size)
    return (multiplier * int(position) + offset) % size


@dataclass(frozen=True, slots=True)
class ShaftSampleContext:
    draw_id: int
    plan_cycle: int
    transform_seed: int

    def to_dict(self) -> dict[str, int]:
        return {
            "draw_id": self.draw_id,
            "plan_cycle": self.plan_cycle,
            "transform_seed": self.transform_seed,
        }


@dataclass(frozen=True, slots=True)
class ShaftSampleRef:
    dataset_name: str
    row_index: int
    context: ShaftSampleContext


class ShaftSamplePlan:
    """Stateless position-to-sample plan shared by Trainer samplers and direct indexing."""

    def __init__(
        self,
        source_sizes: dict[str, int],
        source_weights: dict[str, float],
        *,
        strategy: str = "weighted",
        num_samples: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.strategy = str(strategy).strip().lower()
        if self.strategy not in {"concat", "weighted"}:
            raise ValueError(f"Unsupported mix strategy: {self.strategy!r}.")
        active = [
            (str(name), int(size), float(source_weights.get(name, 1.0)))
            for name, size in sorted(source_sizes.items())
            if int(size) > 0 and float(source_weights.get(name, 1.0)) > 0
        ]
        if not active:
            raise ValueError("No active datasets for sample planning.")
        self.source_names = tuple(item[0] for item in active)
        self.source_sizes = tuple(item[1] for item in active)
        raw_weights = tuple(item[2] for item in active)
        max_weight = max(raw_weights)
        scaled_weights = tuple(weight / max_weight for weight in raw_weights)
        total_weight = sum(scaled_weights)
        self.source_weights = tuple(weight / total_weight for weight in scaled_weights)
        self.shuffle = bool(shuffle)
        self.seed = int(seed) & _MASK_64
        self.base_size = sum(self.source_sizes)
        self.num_samples = self.base_size if num_samples is None else int(num_samples)
        if self.num_samples <= 0:
            raise ValueError("Sample plan num_samples must be > 0.")

        total = 0
        self._source_ends: list[int] = []
        for size in self.source_sizes:
            total += size
            self._source_ends.append(total)
        cumulative = 0.0
        self._weight_ends: list[float] = []
        for weight in self.source_weights:
            cumulative += weight
            self._weight_ends.append(cumulative)
        self._weight_ends[-1] = 1.0
        fingerprint_payload = (
            self.strategy,
            self.source_names,
            self.source_sizes,
            self.source_weights,
            self.num_samples,
            self.shuffle,
            self.seed,
        )
        self.fingerprint = hashlib.sha256(repr(fingerprint_payload).encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return self.num_samples

    def ref_at(self, position: int, *, plan_cycle: int = 0) -> ShaftSampleRef:
        position = int(position)
        if position < 0 or position >= self.num_samples:
            raise IndexError(position)
        plan_cycle = int(plan_cycle)
        draw_id = plan_cycle * self.num_samples + position
        if self.strategy == "concat":
            dataset_name, row_index = self._resolve_concat(draw_id)
        else:
            dataset_name, row_index = self._resolve_weighted(
                position,
                draw_id=draw_id,
                plan_cycle=plan_cycle,
            )
        return ShaftSampleRef(
            dataset_name=dataset_name,
            row_index=row_index,
            context=ShaftSampleContext(
                draw_id=draw_id,
                plan_cycle=plan_cycle,
                transform_seed=_splitmix64(self.seed ^ draw_id ^ 0xA0761D6478BD642F),
            ),
        )

    def _resolve_concat(self, draw_id: int) -> tuple[str, int]:
        source_cycle, position = divmod(draw_id, self.base_size)
        if self.shuffle:
            position = _affine_permute(
                position,
                self.base_size,
                seed=_splitmix64(self.seed ^ source_cycle),
            )
        source_index = bisect_right(self._source_ends, position)
        source_start = 0 if source_index == 0 else self._source_ends[source_index - 1]
        return self.source_names[source_index], position - source_start

    def _resolve_weighted(
        self,
        position: int,
        *,
        draw_id: int,
        plan_cycle: int,
    ) -> tuple[str, int]:
        if self.shuffle:
            source_random = _splitmix64(self.seed ^ (draw_id * 0xD1342543DE82EF95))
            source_value = (source_random >> 11) / float(1 << 53)
            source_index = bisect_right(self._weight_ends, source_value)
            row_random = _splitmix64(source_random ^ 0xE7037ED1A0B428DB)
            row_index = int(row_random % self.source_sizes[source_index])
            return self.source_names[source_index], row_index

        source_value = (position + 0.5) / self.num_samples
        source_index = bisect_right(self._weight_ends, source_value)
        source_start = 0.0 if source_index == 0 else self._weight_ends[source_index - 1]
        local_position = max(
            int(position - math.floor(source_start * self.num_samples)),
            0,
        )
        row_index = (local_position + plan_cycle) % self.source_sizes[source_index]
        return self.source_names[source_index], row_index
