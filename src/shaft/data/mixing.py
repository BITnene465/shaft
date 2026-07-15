from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import repeat
import math

_MASK_64 = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
_WEIGHTED_TICKET_TARGET_SIZE = 4096
_WEIGHTED_TICKET_MAX_SIZE = 16384
_WEIGHTED_MAX_RELATIVE_ERROR = 0.05
_WEIGHTED_SCHEDULE_VERSION = "shaft-weighted-ticket-schedule-v2"
_WEIGHTED_PLAN_VERSION = "shaft-weighted-ticket-plan-v2"
_TICKET_SHUFFLE_SALT = 0xD2B74407B1CE6E93
_TICKET_BLOCK_SALT = 0xCA5A826395121157
_SOURCE_CYCLE_SALT = 0x9E3779B185EBCA87
_AFFINE_OFFSET_SALT = 0xD1B54A32D192ED03
_TRANSFORM_SEED_SALT = 0xA0761D6478BD642F
_FEISTEL_ROUND_SALT = 0x9E3779B97F4A7C15
_FEISTEL_ROUNDS = 6


def _splitmix64(value: int) -> int:
    value = (int(value) + _SPLITMIX_INCREMENT) & _MASK_64
    value = ((value ^ (value >> 30)) * _SPLITMIX_MULTIPLIER_1) & _MASK_64
    value = ((value ^ (value >> 27)) * _SPLITMIX_MULTIPLIER_2) & _MASK_64
    return (value ^ (value >> 31)) & _MASK_64


def _affine_permute(position: int, size: int, *, seed: int) -> int:
    if size <= 1:
        return 0
    multiplier = int(_splitmix64(seed) % size) | 1
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 2) % size
        if multiplier == 0:
            multiplier = 1
    offset = int(_splitmix64(seed ^ _AFFINE_OFFSET_SALT) % size)
    return (multiplier * int(position) + offset) % size


def _feistel_permute(position: int, size: int, *, seed: int) -> int:
    """Stateless keyed permutation over ``range(size)`` via cycle walking."""

    if size <= 1:
        return 0
    domain_bits = max(2, (int(size) - 1).bit_length())
    if domain_bits % 2:
        domain_bits += 1
    half_bits = domain_bits // 2
    half_mask = (1 << half_bits) - 1

    def permute_domain(value: int) -> int:
        left = int(value) >> half_bits
        right = int(value) & half_mask
        for round_index in range(_FEISTEL_ROUNDS):
            round_value = _splitmix64(
                seed ^ (round_index * _FEISTEL_ROUND_SALT) ^ right
            )
            left, right = right, left ^ (round_value & half_mask)
        return (left << half_bits) | right

    candidate = int(position)
    while True:
        candidate = permute_domain(candidate)
        if candidate < size:
            return candidate


@dataclass(frozen=True, slots=True)
class _ResolvedSources:
    names: tuple[str, ...]
    sizes: tuple[int, ...]
    raw_weights: tuple[float, ...]
    normalized_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _TicketBlock:
    source_indices: array
    source_positions: tuple[array, ...]
    digest: str


def _resolve_active_sources(
    source_sizes: dict[str, int],
    source_weights: dict[str, float],
) -> _ResolvedSources:
    active = [
        (str(name), int(size), float(source_weights.get(name, 1.0)))
        for name, size in sorted(source_sizes.items())
        if int(size) > 0 and float(source_weights.get(name, 1.0)) > 0
    ]
    if not active:
        raise ValueError("No active datasets for sample scheduling.")
    raw_weights = tuple(item[2] for item in active)
    max_weight = max(raw_weights)
    scaled_weights = tuple(weight / max_weight for weight in raw_weights)
    total_weight = sum(scaled_weights)
    return _ResolvedSources(
        names=tuple(item[0] for item in active),
        sizes=tuple(item[1] for item in active),
        raw_weights=raw_weights,
        normalized_weights=tuple(weight / total_weight for weight in scaled_weights),
    )


def _exact_ticket_ratios(scaled_weights: tuple[float, ...]) -> tuple[int, ...] | None:
    fractions = tuple(
        Fraction(str(weight)).limit_denominator(_WEIGHTED_TICKET_MAX_SIZE)
        for weight in scaled_weights
    )
    if any(fraction <= 0 for fraction in fractions):
        return None
    common_denominator = 1
    for fraction in fractions:
        common_denominator = math.lcm(common_denominator, fraction.denominator)
        if common_denominator > _WEIGHTED_TICKET_MAX_SIZE:
            return None
    ratios = tuple(
        fraction.numerator * (common_denominator // fraction.denominator)
        for fraction in fractions
    )
    divisor = math.gcd(*ratios)
    reduced = tuple(ratio // divisor for ratio in ratios)
    if sum(reduced) > _WEIGHTED_TICKET_MAX_SIZE:
        return None
    return reduced


def _hamilton_ticket_quotas(
    normalized_weights: tuple[float, ...],
    *,
    block_size: int,
) -> tuple[int, ...]:
    expected = tuple(weight * block_size for weight in normalized_weights)
    quotas = [math.floor(value) for value in expected]
    remaining = block_size - sum(quotas)
    order = sorted(
        range(len(expected)),
        key=lambda index: (expected[index] - quotas[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        quotas[index] += 1
    return tuple(quotas)


def _ticket_relative_errors(
    quotas: tuple[int, ...],
    normalized_weights: tuple[float, ...],
) -> tuple[float, ...]:
    block_size = sum(quotas)
    return tuple(
        abs((quota / block_size) - expected) / expected
        for quota, expected in zip(quotas, normalized_weights, strict=True)
    )


def _resolve_ticket_quotas(
    source_names: tuple[str, ...],
    raw_weights: tuple[float, ...],
    normalized_weights: tuple[float, ...],
) -> tuple[int, ...]:
    if len(raw_weights) > _WEIGHTED_TICKET_MAX_SIZE:
        raise ValueError(
            "Weighted sampling has more active sources than the maximum ticket block size: "
            f"sources={len(raw_weights)}, max_tickets={_WEIGHTED_TICKET_MAX_SIZE}."
        )
    max_weight = max(raw_weights)
    scaled_weights = tuple(weight / max_weight for weight in raw_weights)
    exact_ratios = _exact_ticket_ratios(scaled_weights)
    if exact_ratios is not None:
        base_size = sum(exact_ratios)
        multiplier = (
            max(_WEIGHTED_TICKET_TARGET_SIZE // base_size, 1)
            if base_size <= _WEIGHTED_TICKET_TARGET_SIZE
            else 1
        )
        quotas = tuple(ratio * multiplier for ratio in exact_ratios)
        if (
            sum(quotas) <= _WEIGHTED_TICKET_MAX_SIZE
            and max(_ticket_relative_errors(quotas, normalized_weights))
            <= _WEIGHTED_MAX_RELATIVE_ERROR
        ):
            return quotas

    candidate_sizes = tuple(
        dict.fromkeys(
            (
                _WEIGHTED_TICKET_TARGET_SIZE,
                min(_WEIGHTED_TICKET_TARGET_SIZE * 2, _WEIGHTED_TICKET_MAX_SIZE),
                _WEIGHTED_TICKET_MAX_SIZE,
            )
        )
    )
    candidates: list[tuple[float, float, int, tuple[int, ...]]] = []
    for block_size in candidate_sizes:
        quotas = _hamilton_ticket_quotas(
            normalized_weights,
            block_size=block_size,
        )
        if all(quota > 0 for quota in quotas):
            relative_errors = _ticket_relative_errors(quotas, normalized_weights)
            max_relative_error = max(relative_errors)
            resolved = tuple(quota / block_size for quota in quotas)
            total_absolute_error = sum(
                abs(actual - expected)
                for actual, expected in zip(
                    resolved,
                    normalized_weights,
                    strict=True,
                )
            )
            candidates.append(
                (
                    max_relative_error,
                    total_absolute_error,
                    block_size,
                    quotas,
                )
            )
    if candidates:
        best = min(candidates, key=lambda candidate: candidate[:3])
        if best[0] <= _WEIGHTED_MAX_RELATIVE_ERROR:
            return best[3]
        best_errors = _ticket_relative_errors(best[3], normalized_weights)
        worst_index = max(
            range(len(normalized_weights)),
            key=best_errors.__getitem__,
        )
        resolved_probability = best[3][worst_index] / best[2]
        raise ValueError(
            "Weighted sampling ticket quantization exceeds the relative-error limit: "
            f"source={source_names[worst_index]!r}, "
            f"target_probability={normalized_weights[worst_index]:.12g}, "
            f"resolved_probability={resolved_probability:.12g}, "
            f"quota={best[3][worst_index]}/{best[2]}, "
            f"relative_error={best[0]:.6g}, "
            f"max_relative_error={_WEIGHTED_MAX_RELATIVE_ERROR:.6g}. "
            "Increase that source weight or simplify the weight ratios."
        )
    raise ValueError(
        "Weighted sampling contains a positive source weight too small to receive a "
        f"ticket within {_WEIGHTED_TICKET_MAX_SIZE} entries. Increase that source "
        "weight or remove the source explicitly."
    )


def _build_ticket_block(
    quotas: tuple[int, ...],
    *,
    seed: int,
) -> _TicketBlock:
    block_size = sum(quotas)
    source_indices = array("H")
    for source_index, quota in enumerate(quotas):
        source_indices.extend(repeat(source_index, quota))
    state = _splitmix64(seed ^ _TICKET_SHUFFLE_SALT)
    for upper in range(block_size - 1, 0, -1):
        state = _splitmix64(state)
        lower = int(state % (upper + 1))
        source_indices[upper], source_indices[lower] = (
            source_indices[lower],
            source_indices[upper],
        )
    source_positions = tuple(array("H") for _ in quotas)
    digest = hashlib.sha256()
    for position, source_index in enumerate(source_indices):
        source_positions[source_index].append(position)
        digest.update(int(source_index).to_bytes(2, "big"))
    return _TicketBlock(
        source_indices=source_indices,
        source_positions=source_positions,
        digest=digest.hexdigest(),
    )


def _stable_source_seed(source_name: str) -> int:
    digest = hashlib.sha256(str(source_name).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


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


class ShaftSampleSchedule:
    """Horizon-independent mapping from a logical draw id to one source row.

    Planned batching consumes this schedule directly. Its identity deliberately
    excludes training duration so extending ``max_steps`` does not change the draw
    prefix used by an existing run.
    """

    def __init__(
        self,
        source_sizes: dict[str, int],
        source_weights: dict[str, float],
        *,
        strategy: str = "weighted",
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.strategy = str(strategy).strip().lower()
        if self.strategy not in {"concat", "weighted"}:
            raise ValueError(f"Unsupported mix strategy: {self.strategy!r}.")
        if self.strategy == "weighted" and not bool(shuffle):
            raise ValueError(
                "Horizon-independent weighted sampling requires shuffle=true. "
                "Use concat or enable shuffle for planned batching."
            )
        resolved = _resolve_active_sources(source_sizes, source_weights)
        self.source_names = resolved.names
        self.source_sizes = resolved.sizes
        self.source_weights = resolved.normalized_weights
        self.shuffle = bool(shuffle)
        self.seed = int(seed) & _MASK_64
        self.base_size = sum(self.source_sizes)

        total = 0
        self._source_ends: list[int] = []
        for size in self.source_sizes:
            total += size
            self._source_ends.append(total)
        self.source_quotas: tuple[int, ...] = ()
        self.ticket_block_size = 0
        self.ticket_block_digest = ""
        self._ticket_block: _TicketBlock | None = None
        self._source_cycle_seeds: tuple[int, ...] = ()
        if self.strategy == "weighted":
            self.source_quotas = _resolve_ticket_quotas(
                resolved.names,
                resolved.raw_weights,
                resolved.normalized_weights,
            )
            self.ticket_block_size = sum(self.source_quotas)
            self._ticket_block = _build_ticket_block(
                self.source_quotas,
                seed=self.seed,
            )
            self.ticket_block_digest = self._ticket_block.digest
            self._source_cycle_seeds = tuple(
                _stable_source_seed(source_name) for source_name in self.source_names
            )
            payload = (
                _WEIGHTED_SCHEDULE_VERSION,
                self.source_names,
                self.source_sizes,
                self.source_weights,
                self.source_quotas,
                self.ticket_block_size,
                self.ticket_block_digest,
                _WEIGHTED_TICKET_TARGET_SIZE,
                _WEIGHTED_TICKET_MAX_SIZE,
                _WEIGHTED_MAX_RELATIVE_ERROR,
                _TICKET_SHUFFLE_SALT,
                _TICKET_BLOCK_SALT,
                _SOURCE_CYCLE_SALT,
                _AFFINE_OFFSET_SALT,
                _TRANSFORM_SEED_SALT,
                _FEISTEL_ROUND_SALT,
                _FEISTEL_ROUNDS,
                (
                    _SPLITMIX_INCREMENT,
                    _SPLITMIX_MULTIPLIER_1,
                    _SPLITMIX_MULTIPLIER_2,
                ),
                "splitmix64-fisher-yates-base-v1",
                "block-cycle-affine-rotation-v1",
                "per-source-feistel6-cyclewalk-v1",
                "source-seed-sha256-first8-big-v1",
                self.seed,
            )
        else:
            payload = (
                "shaft-sample-schedule-v1",
                self.strategy,
                self.source_names,
                self.source_sizes,
                self.source_weights,
                self.shuffle,
                self.seed,
            )
        self.fingerprint = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    def ref_at(self, draw_id: int) -> ShaftSampleRef:
        draw_id = int(draw_id)
        if draw_id < 0:
            raise IndexError(draw_id)
        if self.strategy == "concat":
            dataset_name, row_index = self._resolve_concat(draw_id)
        else:
            dataset_name, row_index = self._resolve_weighted(draw_id)
        return ShaftSampleRef(
            dataset_name=dataset_name,
            row_index=row_index,
            context=ShaftSampleContext(
                draw_id=draw_id,
                plan_cycle=0,
                transform_seed=_splitmix64(self.seed ^ draw_id ^ _TRANSFORM_SEED_SALT),
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

    def _resolve_weighted(self, draw_id: int) -> tuple[str, int]:
        block_id, offset = divmod(draw_id, self.ticket_block_size)
        block_cycle, block_position = divmod(block_id, self.ticket_block_size)
        rotation = _affine_permute(
            block_position,
            self.ticket_block_size,
            seed=_splitmix64(
                self.seed ^ _TICKET_BLOCK_SALT ^ (block_cycle * _SOURCE_CYCLE_SALT)
            ),
        )
        ticket_position = (offset + rotation) % self.ticket_block_size
        ticket_block = self._ticket_block
        assert ticket_block is not None
        source_index = int(ticket_block.source_indices[ticket_position])
        source_positions = ticket_block.source_positions[source_index]
        start_rank = bisect_left(source_positions, rotation)
        ticket_rank = bisect_left(source_positions, ticket_position)
        if ticket_position >= rotation:
            local_occurrence = ticket_rank - start_rank
        else:
            local_occurrence = len(source_positions) - start_rank + ticket_rank
        source_occurrence = (
            block_id * self.source_quotas[source_index]
            + local_occurrence
        )
        source_cycle, source_position = divmod(
            source_occurrence,
            self.source_sizes[source_index],
        )
        row_index = _feistel_permute(
            source_position,
            self.source_sizes[source_index],
            seed=_splitmix64(
                self.seed
                ^ self._source_cycle_seeds[source_index]
                ^ (source_cycle * _SOURCE_CYCLE_SALT)
            ),
        )
        return self.source_names[source_index], row_index


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
        resolved = _resolve_active_sources(source_sizes, source_weights)
        self.source_names = resolved.names
        self.source_sizes = resolved.sizes
        self.source_weights = resolved.normalized_weights
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
        self._schedule = (
            ShaftSampleSchedule(
                dict(zip(self.source_names, self.source_sizes, strict=True)),
                dict(zip(self.source_names, resolved.raw_weights, strict=True)),
                strategy=self.strategy,
                shuffle=self.shuffle,
                seed=self.seed,
            )
            if self.strategy != "weighted" or self.shuffle
            else None
        )
        if self.strategy == "weighted" and self.shuffle:
            assert self._schedule is not None
            fingerprint_payload = (
                _WEIGHTED_PLAN_VERSION,
                self._schedule.fingerprint,
                self.num_samples,
            )
        else:
            fingerprint_payload = (
                self.strategy,
                self.source_names,
                self.source_sizes,
                self.source_weights,
                self.num_samples,
                self.shuffle,
                self.seed,
            )
        self.fingerprint = hashlib.sha256(
            repr(fingerprint_payload).encode("utf-8")
        ).hexdigest()

    @property
    def schedule(self) -> ShaftSampleSchedule:
        if self._schedule is None:
            raise ValueError(
                "This finite weighted, unshuffled SamplePlan has no horizon-independent schedule."
            )
        return self._schedule

    @property
    def stream_fingerprint(self) -> str:
        """Identify the logical draw stream independently of finite-plan length when possible.

        Weighted, unshuffled sampling is defined by the finite plan horizon, so its
        finite plan fingerprint is the only honest stream identity.
        """

        if self._schedule is None:
            return self.fingerprint
        return self._schedule.fingerprint

    def __len__(self) -> int:
        return self.num_samples

    def ref_at(self, position: int, *, plan_cycle: int = 0) -> ShaftSampleRef:
        position = int(position)
        if position < 0 or position >= self.num_samples:
            raise IndexError(position)
        plan_cycle = int(plan_cycle)
        draw_id = plan_cycle * self.num_samples + position
        if self._schedule is not None:
            scheduled = self._schedule.ref_at(draw_id)
            return ShaftSampleRef(
                dataset_name=scheduled.dataset_name,
                row_index=scheduled.row_index,
                context=ShaftSampleContext(
                    draw_id=draw_id,
                    plan_cycle=plan_cycle,
                    transform_seed=scheduled.context.transform_seed,
                ),
            )
        dataset_name, row_index = self._resolve_unshuffled_weighted(
            position,
            plan_cycle=plan_cycle,
        )
        return ShaftSampleRef(
            dataset_name=dataset_name,
            row_index=row_index,
            context=ShaftSampleContext(
                draw_id=draw_id,
                plan_cycle=plan_cycle,
                transform_seed=_splitmix64(self.seed ^ draw_id ^ _TRANSFORM_SEED_SALT),
            ),
        )

    def _resolve_unshuffled_weighted(
        self,
        position: int,
        *,
        plan_cycle: int,
    ) -> tuple[str, int]:
        source_value = (position + 0.5) / self.num_samples
        source_index = bisect_right(self._weight_ends, source_value)
        source_start = 0.0 if source_index == 0 else self._weight_ends[source_index - 1]
        local_position = max(
            int(position - math.floor(source_start * self.num_samples)),
            0,
        )
        row_index = (local_position + plan_cycle) % self.source_sizes[source_index]
        return self.source_names[source_index], row_index
