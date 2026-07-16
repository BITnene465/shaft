from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import heapq
from itertools import repeat
import math

_MASK_64 = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
_WEIGHTED_TICKET_TARGET_SIZE = 4096
_WEIGHTED_TICKET_MAX_SIZE = 16384
_WEIGHTED_MAX_RELATIVE_ERROR = 0.05
_WEIGHTED_EXACT_RELATIVE_TOLERANCE = 1e-12
_WEIGHTED_EXHAUSTIVE_SOURCE_LIMIT = 64
_WEIGHTED_FAST_CANDIDATE_LIMIT = 32
_WEIGHTED_ROTATION_PHASE_BLOCKS = 256
_WEIGHTED_ROTATION_MAX_BALANCED_WORLD_SIZE = 64
_WEIGHTED_ROTATION_RANK_MODULUS = math.lcm(
    *range(1, _WEIGHTED_ROTATION_MAX_BALANCED_WORLD_SIZE + 1)
)
_WEIGHTED_SCHEDULE_VERSION = "shaft-weighted-ticket-schedule-v3"
_WEIGHTED_PLAN_VERSION = "shaft-weighted-ticket-plan-v3"
_WEIGHTED_UNSHUFFLED_PLAN_VERSION = "shaft-weighted-unshuffled-ticket-plan-v3"
_WEIGHTED_UNSHUFFLED_STREAM_VERSION = "shaft-weighted-unshuffled-ticket-stream-v3"
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


@lru_cache(maxsize=None)
def _ticket_rotation_step(block_size: int) -> int:
    """Resolve a short full-cycle step whose global-draw slope is rank-coprime."""

    block_size = int(block_size)
    if block_size <= 1:
        return 0
    step = 1
    while True:
        if (
            math.gcd(step, block_size) == 1
            and math.gcd(
                block_size + step,
                _WEIGHTED_ROTATION_RANK_MODULUS,
            )
            == 1
        ):
            return step
        step += 1


def _ticket_block_rotation(block_id: int, block_size: int, *, seed: int) -> int:
    if block_size <= 1:
        return 0
    phase_group, group_position = divmod(
        int(block_id),
        _WEIGHTED_ROTATION_PHASE_BLOCKS,
    )
    phase = int(
        _splitmix64(int(seed) ^ _TICKET_BLOCK_SALT ^ phase_group) % block_size
    )
    return (
        phase - (_ticket_rotation_step(block_size) * group_position)
    ) % block_size


def validate_sample_schedule_world_size(
    *,
    strategy: str,
    shuffle: bool,
    world_size: int,
) -> None:
    """Reject data-parallel geometries outside the weighted rotation proof.

    The weighted schedule deliberately stays independent of training topology.
    Its counter rotation is proven rank-balanced for every world size in
    ``[1, 64]``.  Silently accepting a larger world can lock a rare ticket to a
    strict subset of ranks when ``ticket_block_size + rotation_step`` shares a
    factor with that world size, so unsupported topologies must fail closed.
    """

    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("Data world size must be > 0.")
    if (
        str(strategy).strip().lower() == "weighted"
        and bool(shuffle)
        and world_size > _WEIGHTED_ROTATION_MAX_BALANCED_WORLD_SIZE
    ):
        raise ValueError(
            "Weighted shuffled sampling only guarantees rank-balanced ticket "
            "rotation for data_world_size <= "
            f"{_WEIGHTED_ROTATION_MAX_BALANCED_WORLD_SIZE}, got {world_size}. "
            "Use concat sampling or run with at most 64 data-parallel ranks."
        )


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


def _fast_ticket_candidate_sizes(
    normalized_weights: tuple[float, ...],
) -> tuple[int, ...]:
    """Return bounded denominator-derived candidates for unusually many sources.

    Mixtures with at most ``_WEIGHTED_EXHAUSTIVE_SOURCE_LIMIT`` sources are
    searched exhaustively and therefore get the global Hamilton optimum within
    the configured maximum block size. Above that boundary this function provides
    a fixed-size fast path derived from the rarest probabilities and distribution
    quantiles. If the shortlist cannot satisfy the error contract, the resolver
    falls back to a complete search so a representable mixture is never rejected.
    The common large-catalog path evaluates at most ``candidate_limit`` blocks;
    only pathological inputs pay for the complete fallback.
    """

    minimum_size = len(normalized_weights)
    anchors = {
        _WEIGHTED_TICKET_TARGET_SIZE,
        min(_WEIGHTED_TICKET_TARGET_SIZE * 2, _WEIGHTED_TICKET_MAX_SIZE),
        _WEIGHTED_TICKET_MAX_SIZE,
    }
    anchors = {
        size
        for size in anchors
        if minimum_size <= size <= _WEIGHTED_TICKET_MAX_SIZE
    }
    ordered = sorted(normalized_weights)
    representative_indices = set(range(min(16, len(ordered))))
    representative_indices.update(
        round(index * (len(ordered) - 1) / 16) for index in range(17)
    )
    representative_probabilities = tuple(
        ordered[index] for index in sorted(representative_indices)
    )
    candidates = set(anchors)
    for probability in representative_probabilities:
        reciprocal = round(1.0 / probability)
        candidates.update(range(reciprocal - 2, reciprocal + 3))
        denominator = Fraction(str(probability)).limit_denominator(
            _WEIGHTED_TICKET_MAX_SIZE
        ).denominator
        candidates.add(denominator)
        for anchor in (_WEIGHTED_TICKET_TARGET_SIZE, _WEIGHTED_TICKET_MAX_SIZE):
            quotient = anchor / denominator
            for multiplier in {math.floor(quotient), round(quotient), math.ceil(quotient)}:
                resolved = denominator * max(int(multiplier), 1)
                candidates.update((resolved - 1, resolved, resolved + 1))
    candidates = {
        size
        for size in candidates
        if minimum_size <= size <= _WEIGHTED_TICKET_MAX_SIZE
    }

    def approximation_score(block_size: int) -> tuple[float, int, int]:
        worst_error = max(
            abs((max(round(probability * block_size), 1) / block_size) - probability)
            / probability
            for probability in representative_probabilities
        )
        return (
            worst_error,
            abs(block_size - _WEIGHTED_TICKET_TARGET_SIZE),
            block_size,
        )

    shortlist = set(anchors)
    shortlist.update(
        sorted(candidates - anchors, key=approximation_score)[
            : max(_WEIGHTED_FAST_CANDIDATE_LIMIT - len(shortlist), 0)
        ]
    )
    return tuple(sorted(shortlist))


def _ticket_candidate_sizes(
    normalized_weights: tuple[float, ...],
) -> range | tuple[int, ...]:
    if len(normalized_weights) <= _WEIGHTED_EXHAUSTIVE_SOURCE_LIMIT:
        return range(len(normalized_weights), _WEIGHTED_TICKET_MAX_SIZE + 1)
    return _fast_ticket_candidate_sizes(normalized_weights)


def _ticket_quota_candidate(
    normalized_weights: tuple[float, ...],
    *,
    block_size: int,
) -> tuple[float, float, int, tuple[int, ...]] | None:
    quotas = _hamilton_ticket_quotas(
        normalized_weights,
        block_size=block_size,
    )
    if not all(quota > 0 for quota in quotas):
        return None
    relative_errors = _ticket_relative_errors(quotas, normalized_weights)
    resolved = tuple(quota / block_size for quota in quotas)
    total_absolute_error = sum(
        abs(actual - expected)
        for actual, expected in zip(
            resolved,
            normalized_weights,
            strict=True,
        )
    )
    return max(relative_errors), total_absolute_error, block_size, quotas


def _ticket_individually_feasible_block_sizes(
    normalized_weights: tuple[float, ...],
) -> tuple[int, ...]:
    """Return block sizes where every source can meet the relative-error bound.

    A source with probability ``p`` and integer quota ``q`` accepts a contiguous
    block-size interval. The total number of intervals over all sources is
    ``O(max_tickets + source_count)`` because their maximum quotas sum to roughly
    ``max_tickets``. Merging equal-probability sources and range-adding those
    intervals avoids the previous large-catalog ``block_sizes * N log N`` scan.
    This is a necessary filter only; Hamilton tie-breaking is checked separately.
    """

    minimum_size = len(normalized_weights)
    maximum_size = _WEIGHTED_TICKET_MAX_SIZE
    coverage_delta = [0] * (maximum_size + 2)
    probability_counts = Counter(normalized_weights)
    for probability, multiplicity in probability_counts.items():
        maximum_quota = math.floor(
            math.nextafter(
                maximum_size
                * probability
                * (1.0 + _WEIGHTED_MAX_RELATIVE_ERROR),
                math.inf,
            )
        )
        intervals: list[tuple[int, int]] = []
        for quota in range(1, maximum_quota + 1):
            lower = math.ceil(
                math.nextafter(
                    quota
                    / (probability * (1.0 + _WEIGHTED_MAX_RELATIVE_ERROR)),
                    -math.inf,
                )
            )
            upper = math.floor(
                math.nextafter(
                    quota
                    / (probability * (1.0 - _WEIGHTED_MAX_RELATIVE_ERROR)),
                    math.inf,
                )
            )
            lower = max(lower, minimum_size)
            upper = min(upper, maximum_size)
            if lower > upper:
                continue
            if intervals and lower <= intervals[-1][1] + 1:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], upper))
            else:
                intervals.append((lower, upper))
        for lower, upper in intervals:
            coverage_delta[lower] += multiplicity
            coverage_delta[upper + 1] -= multiplicity

    source_count = len(normalized_weights)
    coverage = 0
    feasible: list[int] = []
    for block_size in range(minimum_size, maximum_size + 1):
        coverage += coverage_delta[block_size]
        if coverage == source_count:
            feasible.append(block_size)
    return tuple(feasible)


def _hamilton_candidate_meets_error_contract(
    normalized_weights: tuple[float, ...],
    *,
    block_size: int,
) -> bool:
    """Check a Hamilton block in O(source_count) without materializing a sort."""

    expected = tuple(weight * block_size for weight in normalized_weights)
    floors = tuple(math.floor(value) for value in expected)
    remaining = block_size - sum(floors)
    required_increment_min_key: tuple[float, int] | None = None
    forbidden_increment_max_key: tuple[float, int] | None = None
    minimum_quota_sum = 0
    maximum_quota_sum = 0

    for index, (probability, expected_quota, floor_quota) in enumerate(
        zip(normalized_weights, expected, floors, strict=True)
    ):
        ceil_quota = floor_quota + 1
        floor_allowed = floor_quota > 0 and (
            abs((floor_quota / block_size) - probability) / probability
            <= _WEIGHTED_MAX_RELATIVE_ERROR
        )
        ceil_allowed = (
            abs((ceil_quota / block_size) - probability) / probability
            <= _WEIGHTED_MAX_RELATIVE_ERROR
        )
        if not floor_allowed and not ceil_allowed:
            return False
        minimum_quota_sum += floor_quota if floor_allowed else ceil_quota
        maximum_quota_sum += ceil_quota if ceil_allowed else floor_quota
        key = (expected_quota - floor_quota, -index)
        if ceil_allowed and not floor_allowed:
            if required_increment_min_key is None or key < required_increment_min_key:
                required_increment_min_key = key
        elif floor_allowed and not ceil_allowed:
            if forbidden_increment_max_key is None or key > forbidden_increment_max_key:
                forbidden_increment_max_key = key

    if not minimum_quota_sum <= block_size <= maximum_quota_sum:
        return False
    keys = tuple(
        (expected_quota - floor_quota, -index)
        for index, (expected_quota, floor_quota) in enumerate(
            zip(expected, floors, strict=True)
        )
    )
    if required_increment_min_key is not None:
        if sum(key > required_increment_min_key for key in keys) >= remaining:
            return False
    if forbidden_increment_max_key is not None:
        if sum(key > forbidden_increment_max_key for key in keys) < remaining:
            return False
    return True


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
    minimum_representable_probability = 1.0 / (
        _WEIGHTED_TICKET_MAX_SIZE * (1.0 + _WEIGHTED_MAX_RELATIVE_ERROR)
    )
    unrepresentable_indices = [
        index
        for index, probability in enumerate(normalized_weights)
        if probability < minimum_representable_probability
    ]
    if unrepresentable_indices:
        worst_index = min(
            unrepresentable_indices,
            key=normalized_weights.__getitem__,
        )
        raise ValueError(
            "Weighted sampling ticket quantization exceeds the relative-error limit: "
            f"source={source_names[worst_index]!r}, "
            f"target_probability={normalized_weights[worst_index]:.12g}, "
            f"minimum_representable_probability={minimum_representable_probability:.12g}, "
            f"max_tickets={_WEIGHTED_TICKET_MAX_SIZE}, "
            f"max_relative_error={_WEIGHTED_MAX_RELATIVE_ERROR:.6g}. "
            "Increase that source weight or remove the source explicitly."
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
            <= _WEIGHTED_EXACT_RELATIVE_TOLERANCE
        ):
            return quotas

    best: tuple[float, float, int, tuple[int, ...]] | None = None
    candidate_sizes = _ticket_candidate_sizes(normalized_weights)
    for block_size in candidate_sizes:
        candidate = _ticket_quota_candidate(
            normalized_weights,
            block_size=block_size,
        )
        if candidate is not None and (best is None or candidate[:3] < best[:3]):
            best = candidate

    if len(normalized_weights) <= _WEIGHTED_EXHAUSTIVE_SOURCE_LIMIT:
        if best is not None and best[0] <= _WEIGHTED_MAX_RELATIVE_ERROR:
            return best[3]
    elif best is not None and best[0] <= _WEIGHTED_MAX_RELATIVE_ERROR:
        return best[3]
    else:
        evaluated_sizes = set(candidate_sizes)
        target_size = max(_WEIGHTED_TICKET_TARGET_SIZE, len(normalized_weights))
        fallback_sizes = sorted(
            _ticket_individually_feasible_block_sizes(normalized_weights),
            key=lambda size: (abs(size - target_size), size),
        )
        for block_size in fallback_sizes:
            if block_size in evaluated_sizes:
                continue
            if not _hamilton_candidate_meets_error_contract(
                normalized_weights,
                block_size=block_size,
            ):
                continue
            candidate = _ticket_quota_candidate(
                normalized_weights,
                block_size=block_size,
            )
            if candidate is None:
                continue
            if best is None or candidate[:3] < best[:3]:
                best = candidate
            if candidate[0] <= _WEIGHTED_MAX_RELATIVE_ERROR:
                return candidate[3]

    if best is not None:
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


def _build_unshuffled_ticket_block(quotas: tuple[int, ...]) -> _TicketBlock:
    """Merge per-source occurrence midpoints into a deterministic low-discrepancy block.

    This is deliberately not a random permutation.  Sorting occurrence midpoints
    spreads each source's exact integer quota across the block, so finite-plan
    boundaries cannot permanently award the same rounding remainder or starve a
    positive-weight source.  ``Fraction`` keeps the ordering platform-independent.
    """

    pending: list[tuple[Fraction, int, int]] = [
        (Fraction(1, 2 * quota), source_index, 0)
        for source_index, quota in enumerate(quotas)
    ]
    heapq.heapify(pending)
    source_indices = array("H")
    source_positions = tuple(array("H") for _ in quotas)
    digest = hashlib.sha256()
    while pending:
        _, source_index, occurrence = heapq.heappop(pending)
        position = len(source_indices)
        source_indices.append(source_index)
        source_positions[source_index].append(position)
        digest.update(int(source_index).to_bytes(2, "big"))
        next_occurrence = occurrence + 1
        if next_occurrence < quotas[source_index]:
            heapq.heappush(
                pending,
                (
                    Fraction(2 * next_occurrence + 1, 2 * quotas[source_index]),
                    source_index,
                    next_occurrence,
                ),
            )
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
        self.ticket_rotation_step = 0
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
            self.ticket_rotation_step = _ticket_rotation_step(self.ticket_block_size)
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
                _WEIGHTED_EXACT_RELATIVE_TOLERANCE,
                _WEIGHTED_EXHAUSTIVE_SOURCE_LIMIT,
                _WEIGHTED_FAST_CANDIDATE_LIMIT,
                _WEIGHTED_ROTATION_PHASE_BLOCKS,
                _WEIGHTED_ROTATION_MAX_BALANCED_WORLD_SIZE,
                _WEIGHTED_ROTATION_RANK_MODULUS,
                self.ticket_rotation_step,
                _TICKET_SHUFFLE_SALT,
                _TICKET_BLOCK_SALT,
                _SOURCE_CYCLE_SALT,
                _TRANSFORM_SEED_SALT,
                _FEISTEL_ROUND_SALT,
                _FEISTEL_ROUNDS,
                (
                    _SPLITMIX_INCREMENT,
                    _SPLITMIX_MULTIPLIER_1,
                    _SPLITMIX_MULTIPLIER_2,
                ),
                "splitmix64-fisher-yates-base-v1",
                "splitmix64-phase-coprime-counter-rotation-v1",
                "smallest-step-coprime-with-block-and-rank-modulus-v1",
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

    def validate_data_world_size(self, world_size: int) -> None:
        validate_sample_schedule_world_size(
            strategy=self.strategy,
            shuffle=self.shuffle,
            world_size=world_size,
        )

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
        rotation = _ticket_block_rotation(
            block_id,
            self.ticket_block_size,
            seed=self.seed,
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
        self._unshuffled_source_quotas: tuple[int, ...] = ()
        self._unshuffled_ticket_block_size = 0
        self._unshuffled_ticket_block: _TicketBlock | None = None
        if self.strategy == "weighted" and not self.shuffle:
            self._unshuffled_source_quotas = _resolve_ticket_quotas(
                resolved.names,
                resolved.raw_weights,
                resolved.normalized_weights,
            )
            self._unshuffled_ticket_block_size = sum(
                self._unshuffled_source_quotas
            )
            self._unshuffled_ticket_block = _build_unshuffled_ticket_block(
                self._unshuffled_source_quotas
            )
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
            self._stream_fingerprint = self._schedule.fingerprint
            fingerprint_payload = (
                _WEIGHTED_PLAN_VERSION,
                self._stream_fingerprint,
                self.num_samples,
            )
        elif self.strategy == "weighted":
            stream_payload = (
                _WEIGHTED_UNSHUFFLED_STREAM_VERSION,
                self.source_names,
                self.source_sizes,
                self.source_weights,
                self._unshuffled_source_quotas,
                self._unshuffled_ticket_block_size,
                self._unshuffled_ticket_block.digest,
                _WEIGHTED_TICKET_TARGET_SIZE,
                _WEIGHTED_TICKET_MAX_SIZE,
                _WEIGHTED_MAX_RELATIVE_ERROR,
                _WEIGHTED_EXACT_RELATIVE_TOLERANCE,
                self.seed,
            )
            self._stream_fingerprint = hashlib.sha256(
                repr(stream_payload).encode("utf-8")
            ).hexdigest()
            fingerprint_payload = (
                _WEIGHTED_UNSHUFFLED_PLAN_VERSION,
                self._stream_fingerprint,
                self.num_samples,
            )
        else:
            assert self._schedule is not None
            self._stream_fingerprint = self._schedule.fingerprint
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
                "This finite weighted, unshuffled SamplePlan has no public "
                "ShaftSampleSchedule adapter."
            )
        return self._schedule

    @property
    def stream_fingerprint(self) -> str:
        """Identify the logical draw stream independently of finite-plan length."""

        return self._stream_fingerprint

    def validate_data_world_size(self, world_size: int) -> None:
        validate_sample_schedule_world_size(
            strategy=self.strategy,
            shuffle=self.shuffle,
            world_size=world_size,
        )

    def __len__(self) -> int:
        return self.num_samples

    def ref_at(self, position: int, *, plan_cycle: int = 0) -> ShaftSampleRef:
        position = int(position)
        if position < 0 or position >= self.num_samples:
            raise IndexError(position)
        plan_cycle = int(plan_cycle)
        if plan_cycle < 0:
            raise IndexError(plan_cycle)
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
        draw_id = plan_cycle * self.num_samples + position
        block_id, block_position = divmod(
            draw_id,
            self._unshuffled_ticket_block_size,
        )
        ticket_block = self._unshuffled_ticket_block
        assert ticket_block is not None
        source_index = int(ticket_block.source_indices[block_position])
        local_occurrence = bisect_left(
            ticket_block.source_positions[source_index],
            block_position,
        )
        source_occurrence = (
            block_id * self._unshuffled_source_quotas[source_index]
            + local_occurrence
        )
        row_index = source_occurrence % self.source_sizes[source_index]
        return self.source_names[source_index], row_index
