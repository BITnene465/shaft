from __future__ import annotations

from collections import Counter
import math
import random

import pytest
import shaft.data.mixing as mixing
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler

from shaft.data import (
    ShaftGroupedSampleContract,
    ShaftGroupedSampleSampler,
    ShaftSamplePlan,
    ShaftSampleRef,
    ShaftSampleSchedule,
    ShaftSampleSampler,
)


def _brute_force_hamilton_oracle(
    weights: tuple[float, ...],
) -> tuple[int, tuple[int, ...]]:
    total = sum(weights)
    probabilities = tuple(weight / total for weight in weights)
    best: tuple[float, float, int, tuple[int, ...]] | None = None
    for block_size in range(len(weights), 16_385):
        expected = tuple(probability * block_size for probability in probabilities)
        quotas = [math.floor(value) for value in expected]
        remaining = block_size - sum(quotas)
        remainder_order = sorted(
            range(len(expected)),
            key=lambda index: (expected[index] - quotas[index], -index),
            reverse=True,
        )
        for index in remainder_order[:remaining]:
            quotas[index] += 1
        if not all(quotas):
            continue
        resolved = tuple(quota / block_size for quota in quotas)
        relative_error = max(
            abs(actual - expected_probability) / expected_probability
            for actual, expected_probability in zip(resolved, probabilities, strict=True)
        )
        total_absolute_error = sum(
            abs(actual - expected_probability)
            for actual, expected_probability in zip(resolved, probabilities, strict=True)
        )
        candidate = (relative_error, total_absolute_error, block_size, tuple(quotas))
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    assert best is not None
    return best[2], best[3]


def _counter_rotation_discrepancy_bound(
    block_size: int,
    *,
    prefix_blocks: int,
) -> int:
    remaining = prefix_blocks
    step = mixing._ticket_rotation_step(block_size)
    bound = 0
    while remaining:
        group_length = min(remaining, mixing._WEIGHTED_ROTATION_PHASE_BLOCKS)
        bound += 1 + math.ceil(step * max(group_length - 1, 0) / block_size)
        remaining -= group_length
    return bound


def test_concat_plan_covers_every_row_without_materialized_indices() -> None:
    plan = ShaftSamplePlan(
        {"a": 3, "b": 2},
        {"a": 1.0, "b": 1.0},
        strategy="concat",
        shuffle=True,
        seed=1,
    )

    refs = [plan.ref_at(index) for index in range(len(plan))]

    assert len(plan) == 5
    assert {(ref.dataset_name, ref.row_index) for ref in refs} == {
        ("a", 0),
        ("a", 1),
        ("a", 2),
        ("b", 0),
        ("b", 1),
    }
    assert not hasattr(plan, "indices")


def test_weighted_schedule_matches_ticket_quotas_without_materializing_horizon() -> None:
    schedule = ShaftSampleSchedule(
        {"a": 10, "b": 10},
        {"a": 1.0, "b": 3.0},
        strategy="weighted",
        shuffle=True,
        seed=7,
    )

    counts = Counter(
        schedule.ref_at(draw_id).dataset_name
        for draw_id in range(schedule.ticket_block_size)
    )
    quotas = dict(zip(schedule.source_names, schedule.source_quotas, strict=True))

    assert counts == quotas
    assert counts["a"] / schedule.ticket_block_size == pytest.approx(0.25, abs=1e-3)
    assert counts["b"] / schedule.ticket_block_size == pytest.approx(0.75, abs=1e-3)
    assert len(schedule.ticket_block_digest) == 64


def test_weighted_schedule_uses_best_available_hamilton_resolution() -> None:
    schedule = ShaftSampleSchedule(
        {"rare": 10, "a": 10, "b": 10},
        {"rare": 0.0002, "a": 1.0, "b": 0.6180339887},
        strategy="weighted",
        shuffle=True,
        seed=11,
    )
    quotas = dict(zip(schedule.source_names, schedule.source_quotas, strict=True))
    target = 0.0002 / (0.0002 + 1.0 + 0.6180339887)
    actual = quotas["rare"] / schedule.ticket_block_size

    assert schedule.ticket_block_size == 8091
    assert abs(actual - target) / target < 0.05


@pytest.mark.parametrize(
    ("weights", "expected_block_size", "max_relative_error"),
    [
        (
            {"rare": 0.0001, "a": 0.6180339887, "b": 0.3818660113},
            10_000,
            1e-4,
        ),
        (
            {
                "rare": 1 / 7_800,
                "a": 0.6180339887,
                "b": 0.38183780617179486,
            },
            15_600,
            1e-4,
        ),
    ],
)
def test_weighted_schedule_searches_nonstandard_ticket_block_sizes(
    weights: dict[str, float],
    expected_block_size: int,
    max_relative_error: float,
) -> None:
    schedule = ShaftSampleSchedule(
        {name: 10 for name in weights},
        weights,
        strategy="weighted",
        shuffle=True,
        seed=11,
    )
    target_total = sum(weights.values())
    quotas = dict(zip(schedule.source_names, schedule.source_quotas, strict=True))
    relative_errors = {
        name: abs((quotas[name] / schedule.ticket_block_size) - (weight / target_total))
        / (weight / target_total)
        for name, weight in weights.items()
    }

    assert schedule.ticket_block_size == expected_block_size
    assert max(relative_errors.values()) < max_relative_error


def test_weighted_schedule_matches_brute_force_hamilton_oracle() -> None:
    generator = random.Random(20260715)
    for source_count in range(2, 7):
        weights = tuple(
            generator.uniform(0.05, 1.0) + math.sqrt(index + 2) * 1e-7
            for index in range(source_count)
        )
        expected_block_size, expected_quotas = _brute_force_hamilton_oracle(weights)
        schedule = ShaftSampleSchedule(
            {f"source_{index:02d}": 10 for index in range(source_count)},
            {f"source_{index:02d}": weight for index, weight in enumerate(weights)},
            strategy="weighted",
            shuffle=True,
            seed=17,
        )

        assert schedule.ticket_block_size == expected_block_size
        assert schedule.source_quotas == expected_quotas


def test_weighted_large_source_quota_search_has_bounded_candidate_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_count = 256
    weights = {
        f"source_{index:03d}": 1.0 + math.sqrt(index + 2) * 1e-4
        for index in range(source_count)
    }
    source_sizes = dict.fromkeys(weights, 1)
    original = mixing._hamilton_ticket_quotas
    evaluated_block_sizes: list[int] = []

    def record_candidate(
        normalized_weights: tuple[float, ...],
        *,
        block_size: int,
    ) -> tuple[int, ...]:
        evaluated_block_sizes.append(block_size)
        return original(normalized_weights, block_size=block_size)

    monkeypatch.setattr(mixing, "_hamilton_ticket_quotas", record_candidate)
    schedule = ShaftSampleSchedule(
        source_sizes,
        weights,
        strategy="weighted",
        shuffle=True,
        seed=23,
    )

    assert schedule.ticket_block_size >= source_count
    assert len(evaluated_block_sizes) <= 32


def test_weighted_large_source_search_falls_back_when_shortlist_misses_valid_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = {f"rare_{index:02d}": 0.0001 for index in range(63)}
    weights["a"] = 0.6140339887
    weights["b"] = 1.0 - sum(weights.values())
    monkeypatch.setattr(
        mixing,
        "_fast_ticket_candidate_sizes",
        lambda _weights: (4096, 8192, 16384),
    )

    schedule = ShaftSampleSchedule(
        dict.fromkeys(weights, 1),
        weights,
        strategy="weighted",
        shuffle=True,
        seed=29,
    )
    relative_errors = mixing._ticket_relative_errors(
        schedule.source_quotas,
        schedule.source_weights,
    )

    assert schedule.ticket_block_size not in {4096, 8192, 16384}
    assert max(relative_errors) <= 0.05


def test_weighted_large_source_impossible_mixture_has_bounded_fallback_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_count = 7_000
    weights = {
        f"source_{index:04d}": 0.8 if index < source_count // 2 else 1.2
        for index in range(source_count)
    }
    original = mixing._hamilton_ticket_quotas
    evaluated_block_sizes: list[int] = []

    def record_candidate(
        normalized_weights: tuple[float, ...],
        *,
        block_size: int,
    ) -> tuple[int, ...]:
        evaluated_block_sizes.append(block_size)
        return original(normalized_weights, block_size=block_size)

    monkeypatch.setattr(mixing, "_hamilton_ticket_quotas", record_candidate)

    with pytest.raises(ValueError, match="relative-error limit"):
        ShaftSampleSchedule(
            dict.fromkeys(weights, 1),
            weights,
            strategy="weighted",
            shuffle=True,
            seed=31,
        )

    assert len(evaluated_block_sizes) <= mixing._WEIGHTED_FAST_CANDIDATE_LIMIT


def test_weighted_hamilton_contract_predicate_matches_materialized_candidate() -> None:
    generator = random.Random(20260716)
    for source_count in (65, 97):
        for _ in range(8):
            weights = tuple(generator.uniform(0.2, 1.0) for _ in range(source_count))
            total = sum(weights)
            probabilities = tuple(weight / total for weight in weights)
            for block_size in (
                source_count,
                4096,
                generator.randint(source_count, 16_384),
                16_384,
            ):
                candidate = mixing._ticket_quota_candidate(
                    probabilities,
                    block_size=block_size,
                )
                expected = (
                    candidate is not None
                    and candidate[0] <= mixing._WEIGHTED_MAX_RELATIVE_ERROR
                )

                assert (
                    mixing._hamilton_candidate_meets_error_contract(
                        probabilities,
                        block_size=block_size,
                    )
                    is expected
                )


def test_weighted_schedule_rejects_large_ticket_probability_distortion() -> None:
    with pytest.raises(ValueError, match="relative-error limit.*rare"):
        ShaftSampleSchedule(
            {"rare": 10, "a": 10, "b": 10},
            {"rare": 0.0000306, "a": 0.3, "b": 0.6999694},
            strategy="weighted",
            shuffle=True,
            seed=11,
        )


def test_weighted_schedule_rotates_rare_source_across_rank_residues() -> None:
    schedule = ShaftSampleSchedule(
        {"rare": 100, "bulk": 100},
        {"rare": 1.0, "bulk": 4095.0},
        strategy="weighted",
        shuffle=True,
        seed=1,
    )
    rare_draw_residues: set[int] = set()

    for block_id in range(32):
        block_start = block_id * schedule.ticket_block_size
        rare_draws = [
            draw_id
            for draw_id in range(block_start, block_start + schedule.ticket_block_size)
            if schedule.ref_at(draw_id).dataset_name == "rare"
        ]
        assert len(rare_draws) == 1
        rare_draw_residues.add(rare_draws[0] % 8)

    assert rare_draw_residues == set(range(8))


def test_weighted_schedule_avoids_adversarial_rare_source_rank_lock() -> None:
    schedule = ShaftSampleSchedule(
        {"rare": 10, "bulk": 10},
        {"rare": 1.0, "bulk": 2048.0},
        strategy="weighted",
        shuffle=True,
        seed=371,
    )
    rank_counts: Counter[int] = Counter()

    for block_id in range(16):
        block_start = block_id * schedule.ticket_block_size
        rare_draws = [
            draw_id
            for draw_id in range(block_start, block_start + schedule.ticket_block_size)
            if schedule.ref_at(draw_id).dataset_name == "rare"
        ]
        assert len(rare_draws) == 1
        rank_counts[rare_draws[0] % 8] += 1

    assert set(rank_counts) == set(range(8))
    assert max(rank_counts.values()) - min(rank_counts.values()) <= 3


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_weighted_rare_source_rank_prefixes_are_balanced_across_seeds(
    world_size: int,
) -> None:
    for seed in (0, 1, 42, 371, 999):
        schedule = ShaftSampleSchedule(
            {"rare": 10, "bulk": 10},
            {"rare": 1.0, "bulk": 2048.0},
            strategy="weighted",
            shuffle=True,
            seed=seed,
        )
        ticket_block = schedule._ticket_block
        assert ticket_block is not None
        rare_source_index = schedule.source_names.index("rare")
        rare_ticket_position = ticket_block.source_positions[rare_source_index][0]

        for prefix_blocks in (64, 256):
            rank_counts: Counter[int] = Counter()
            for block_id in range(prefix_blocks):
                rotation = mixing._ticket_block_rotation(
                    block_id,
                    schedule.ticket_block_size,
                    seed=schedule.seed,
                )
                offset = (rare_ticket_position - rotation) % schedule.ticket_block_size
                draw_id = block_id * schedule.ticket_block_size + offset
                rank_counts[draw_id % world_size] += 1

            assert set(rank_counts) == set(range(world_size))
            if prefix_blocks == 256:
                expected = prefix_blocks / world_size
                assert max(
                    abs(count - expected) for count in rank_counts.values()
                ) <= expected * 0.5


@pytest.mark.parametrize(
    "block_size",
    [2049, 4095, 4096, 4097, 5000, 8191, 8192, 10_000, 16_384],
)
@pytest.mark.parametrize("world_size", [2, 4, 8, 16, 32, 64])
def test_weighted_counter_rotation_has_bounded_rank_prefix_discrepancy(
    block_size: int,
    world_size: int,
) -> None:
    step = mixing._ticket_rotation_step(block_size)
    assert math.gcd(step, block_size) == 1
    assert math.gcd(
        block_size + step,
        mixing._WEIGHTED_ROTATION_RANK_MODULUS,
    ) == 1

    for seed in (0, 1, 42, 128, 371, 999, 1164):
        for ticket_position in (
            0,
            block_size // 3,
            block_size - 1,
            mixing._splitmix64(seed ^ 0x1234) % block_size,
        ):
            for prefix_blocks in (64, 256, 1024):
                rank_counts: Counter[int] = Counter()
                for block_id in range(prefix_blocks):
                    rotation = mixing._ticket_block_rotation(
                        block_id,
                        block_size,
                        seed=seed,
                    )
                    offset = (ticket_position - rotation) % block_size
                    draw_id = block_id * block_size + offset
                    rank_counts[draw_id % world_size] += 1

                if prefix_blocks >= 4 * world_size:
                    assert set(rank_counts) == set(range(world_size))
                rank_values = [rank_counts[rank] for rank in range(world_size)]
                assert max(rank_values) - min(rank_values) <= (
                    _counter_rotation_discrepancy_bound(
                        block_size,
                        prefix_blocks=prefix_blocks,
                    )
                )


def test_weighted_rotation_rejects_world_outside_proven_rank_bound() -> None:
    schedule = ShaftSampleSchedule(
        {"rare": 10, "bulk": 10},
        {"rare": 1.0, "bulk": 4482.0},
        strategy="weighted",
        shuffle=True,
        seed=1,
    )

    assert schedule.ticket_block_size == 4483
    assert schedule.ticket_rotation_step == 6
    assert math.gcd(schedule.ticket_block_size + schedule.ticket_rotation_step, 67) == 67
    schedule.validate_data_world_size(64)
    with pytest.raises(ValueError, match=r"data_world_size <= 64, got 67"):
        schedule.validate_data_world_size(67)


def test_concat_schedule_allows_world_larger_than_weighted_rotation_bound() -> None:
    schedule = ShaftSampleSchedule(
        {"a": 10, "b": 10},
        {"a": 1.0, "b": 1.0},
        strategy="concat",
        shuffle=True,
        seed=1,
    )

    schedule.validate_data_world_size(67)


def test_weighted_schedule_exhausts_each_source_before_repeating_rows() -> None:
    source_sizes = {"a": 5, "b": 7}
    schedule = ShaftSampleSchedule(
        source_sizes,
        {"a": 1.0, "b": 3.0},
        strategy="weighted",
        shuffle=True,
        seed=19,
    )
    rows_by_source = {name: [] for name in source_sizes}

    for draw_id in range(schedule.ticket_block_size):
        ref = schedule.ref_at(draw_id)
        rows = rows_by_source[ref.dataset_name]
        if len(rows) < 2 * source_sizes[ref.dataset_name]:
            rows.append(ref.row_index)
        if all(len(rows_by_source[name]) == 2 * size for name, size in source_sizes.items()):
            break

    for name, size in source_sizes.items():
        rows = rows_by_source[name]
        assert len(rows) == 2 * size
        assert set(rows[:size]) == set(range(size))
        assert set(rows[size : 2 * size]) == set(range(size))


@pytest.mark.parametrize("source_size", [2, 3, 5, 16, 17, 257])
def test_weighted_source_shuffle_is_a_bijection_for_varied_domain_sizes(
    source_size: int,
) -> None:
    schedule = ShaftSampleSchedule(
        {"only": source_size},
        {"only": 1.0},
        strategy="weighted",
        shuffle=True,
        seed=41,
    )

    assert schedule.ticket_block_size == 4096
    assert schedule.ticket_rotation_step == mixing._ticket_rotation_step(4096)
    assert {
        schedule.ref_at(draw_id).row_index for draw_id in range(source_size)
    } == set(range(source_size))


def test_weighted_schedule_supports_deterministic_random_access() -> None:
    kwargs = {
        "source_sizes": {"a": 11, "b": 17, "c": 5},
        "source_weights": {"a": 2.0, "b": 3.0, "c": 1.0},
        "strategy": "weighted",
        "shuffle": True,
        "seed": 23,
    }
    first = ShaftSampleSchedule(**kwargs)
    second = ShaftSampleSchedule(**kwargs)
    positions = [
        first.ticket_block_size * 3 + 7,
        0,
        first.ticket_block_size - 1,
        13,
        first.ticket_block_size + 2,
        13,
    ]

    expected = {position: first.ref_at(position) for position in sorted(set(positions))}

    assert [second.ref_at(position) for position in positions] == [
        expected[position] for position in positions
    ]


def test_weighted_plan_extension_preserves_existing_stream_prefix() -> None:
    short = ShaftSamplePlan(
        {"a": 13, "b": 19},
        {"a": 2.0, "b": 1.0},
        strategy="weighted",
        num_samples=128,
        shuffle=True,
        seed=29,
    )
    long = ShaftSamplePlan(
        {"a": 13, "b": 19},
        {"a": 2.0, "b": 1.0},
        strategy="weighted",
        num_samples=4096,
        shuffle=True,
        seed=29,
    )

    assert short.stream_fingerprint == long.stream_fingerprint
    assert short.fingerprint != long.fingerprint
    assert short.schedule.ticket_block_size == long.schedule.ticket_block_size
    assert [short.ref_at(index) for index in range(len(short))] == [
        long.ref_at(index) for index in range(len(short))
    ]


def test_weighted_v3_and_unshuffled_v3_fingerprints_are_versioned() -> None:
    weighted = ShaftSampleSchedule(
        {"a": 3, "b": 5},
        {"a": 1.0, "b": 3.0},
        strategy="weighted",
        shuffle=True,
        seed=7,
    )
    weighted_plan = ShaftSamplePlan(
        {"a": 3, "b": 5},
        {"a": 1.0, "b": 3.0},
        strategy="weighted",
        num_samples=64,
        shuffle=True,
        seed=7,
    )
    concat = ShaftSampleSchedule(
        {"a": 3, "b": 5},
        {"a": 1.0, "b": 3.0},
        strategy="concat",
        shuffle=True,
        seed=7,
    )
    unshuffled = ShaftSamplePlan(
        {"a": 3, "b": 5},
        {"a": 1.0, "b": 3.0},
        strategy="weighted",
        num_samples=64,
        shuffle=False,
        seed=7,
    )

    assert weighted.fingerprint != (
        "b1e4c4fb8ac319126ad002a853e25c4e9959cd85e862e4ab6d065871e5dde997"
    )
    assert weighted_plan.fingerprint != (
        "dc5b7a1db3d7f5ba1ebcec3a697d8ef03e59f933ca9cb7871e79f44c8be3cebc"
    )
    assert concat.fingerprint == (
        "66a65da279e606b14a22585bd49c2448a89e545122a78f50eca2c9b5b3c164c9"
    )
    assert unshuffled.fingerprint == (
        "dd466f03d62e734f1a3a26e622fb33ba6a353805486e5f4e41efa7603268d61a"
    )
    assert unshuffled.stream_fingerprint == (
        "e16dd1f1993f6076226c20d7fdac556bc6f995e5fdd3353d47674ed605fba922"
    )
    assert weighted.ticket_block_digest == (
        "fada3d4d247e243b9f1f5fecc33034ffe85d9f76b053cb5410c3954d0e3a8336"
    )
    assert weighted.fingerprint != (
        "dd4c51e93765175f3e695e9d0cd5bda30054705c943a21981b94edcd96489c30"
    )
    assert weighted_plan.fingerprint != (
        "27e055d1d6393680fb7d56bfb43a185613e0c8b4dcf74214a0cad933dd59af8e"
    )
    assert weighted.fingerprint == (
        "6ca023aaacbb1791658bd16a634fffe9937a07771d365d6bf28f9a3c27a006ff"
    )
    assert weighted_plan.fingerprint == (
        "0053ac596b57d99b2476f7926378dd561b5481c17cc5dc04392a460665f27f23"
    )
    probe_positions = [0, 1, 2, 7, 31, 127, 4095, 4096, 4097, 12295]
    assert [
        (
            weighted.ref_at(draw_id).dataset_name,
            weighted.ref_at(draw_id).row_index,
            weighted.ref_at(draw_id).context.transform_seed,
        )
        for draw_id in probe_positions
    ] == [
        ("b", 1, 6551058038977729289),
        ("b", 0, 12733251116385539102),
        ("b", 3, 7700560105411520951),
        ("a", 0, 4870315401550313391),
        ("b", 1, 2239024681287515773),
        ("b", 1, 15171383588411787973),
        ("b", 4, 17766263492751155199),
        ("b", 2, 12397394164509198338),
        ("b", 1, 16566888492203895366),
        ("b", 3, 11329217410363271126),
    ]


def test_weighted_schedule_rejects_unrepresentably_small_positive_weight() -> None:
    with pytest.raises(ValueError, match="ticket"):
        ShaftSampleSchedule(
            {"tiny": 2, "large": 2},
            {"tiny": 1e-12, "large": 1.0},
            strategy="weighted",
            shuffle=True,
            seed=31,
        )


def test_weighted_plan_normalizes_large_finite_weights_without_overflow() -> None:
    plan = ShaftSamplePlan(
        {"a": 2, "b": 2},
        {"a": 1e308, "b": 1e308},
        strategy="weighted",
        num_samples=2000,
        seed=17,
    )

    counts = Counter(plan.ref_at(index).dataset_name for index in range(len(plan)))

    assert counts["a"] / len(plan) == pytest.approx(0.5, abs=0.05)
    assert counts["b"] / len(plan) == pytest.approx(0.5, abs=0.05)


def test_step_plan_uses_exact_global_sample_budget() -> None:
    plan = ShaftSamplePlan(
        {"a": 3, "b": 2},
        {"a": 1.0, "b": 1.0},
        strategy="weighted",
        num_samples=64,
        seed=3,
    )

    assert len(plan) == 64
    assert plan.ref_at(63).context.draw_id == 63


def test_unshuffled_weighted_plan_advances_by_each_sources_cycle_cardinality() -> None:
    plan = ShaftSamplePlan(
        {"a": 100, "b": 100},
        {"a": 1.0, "b": 1.0},
        strategy="weighted",
        num_samples=100,
        shuffle=False,
        seed=3,
    )

    cycles = [
        [plan.ref_at(position, plan_cycle=cycle) for position in range(len(plan))]
        for cycle in range(3)
    ]
    per_cycle = [
        {
            name: [ref.row_index for ref in refs if ref.dataset_name == name]
            for name in ("a", "b")
        }
        for refs in cycles
    ]

    assert per_cycle[0] == {"a": list(range(50)), "b": list(range(50))}
    assert per_cycle[1] == {"a": list(range(50, 100)), "b": list(range(50, 100))}
    assert per_cycle[2] == per_cycle[0]
    assert all(
        set(per_cycle[0][name]).isdisjoint(per_cycle[1][name])
        for name in ("a", "b")
    )


def test_unshuffled_weighted_plan_carries_fractional_quota_across_cycles() -> None:
    plan = ShaftSamplePlan(
        {"a": 100, "b": 100},
        {"a": 1.0, "b": 1.0},
        strategy="weighted",
        num_samples=3,
        shuffle=False,
        seed=3,
    )

    counts = Counter(
        plan.ref_at(position, plan_cycle=cycle).dataset_name
        for cycle in range(10)
        for position in range(len(plan))
    )

    assert counts == {"a": 15, "b": 15}


def test_unshuffled_weighted_plan_does_not_permanently_starve_positive_sources() -> None:
    plan = ShaftSamplePlan(
        {"common": 2000, "rare": 20},
        {"common": 1000.0, "rare": 1.0},
        strategy="weighted",
        num_samples=10,
        shuffle=False,
        seed=3,
    )

    refs = [
        plan.ref_at(position, plan_cycle=cycle)
        for cycle in range(101)
        for position in range(len(plan))
    ]

    assert Counter(ref.dataset_name for ref in refs) == {"common": 1009, "rare": 1}
    assert [ref.row_index for ref in refs if ref.dataset_name == "rare"] == [0]


def test_unshuffled_weighted_plan_rotates_equal_source_remainders() -> None:
    plan = ShaftSamplePlan(
        {"a": 10, "b": 10, "c": 10},
        {"a": 1.0, "b": 1.0, "c": 1.0},
        strategy="weighted",
        num_samples=2,
        shuffle=False,
        seed=3,
    )

    refs = [
        plan.ref_at(position, plan_cycle=cycle)
        for cycle in range(3)
        for position in range(len(plan))
    ]

    assert Counter(ref.dataset_name for ref in refs) == {"a": 2, "b": 2, "c": 2}


def test_unshuffled_weighted_draw_prefix_is_independent_of_epoch_horizon() -> None:
    plans = [
        ShaftSamplePlan(
            {"a": 11, "b": 13, "c": 17},
            {"a": 0.1, "b": 0.3, "c": 0.6},
            strategy="weighted",
            num_samples=horizon,
            shuffle=False,
            seed=19,
        )
        for horizon in (2, 7, 17)
    ]

    streams = [
        [
            (
                ref.dataset_name,
                ref.row_index,
                ref.context.draw_id,
                ref.context.transform_seed,
            )
            for draw_id in range(100)
            for ref in (
                plan.ref_at(draw_id % len(plan), plan_cycle=draw_id // len(plan)),
            )
        ]
        for plan in plans
    ]

    assert streams[1:] == [streams[0], streams[0]]
    assert plans[0].stream_fingerprint == plans[1].stream_fingerprint
    assert plans[1].stream_fingerprint == plans[2].stream_fingerprint
    assert len({plan.fingerprint for plan in plans}) == len(plans)


@pytest.mark.parametrize("num_samples", [1, 2, 3, 7, 17, 64, 127])
def test_unshuffled_weighted_plan_has_contiguous_source_occurrences_across_cycles(
    num_samples: int,
) -> None:
    source_sizes = {"a": 11, "b": 13, "c": 17}
    plan = ShaftSamplePlan(
        source_sizes,
        {"a": 0.1, "b": 0.3, "c": 0.6},
        strategy="weighted",
        num_samples=num_samples,
        shuffle=False,
        seed=19,
    )
    refs = [
        plan.ref_at(position, plan_cycle=cycle)
        for cycle in range(4)
        for position in range(num_samples)
    ]
    for source_name, source_size in source_sizes.items():
        rows = [ref.row_index for ref in refs if ref.dataset_name == source_name]
        assert rows == [occurrence % source_size for occurrence in range(len(rows))]


def test_sample_plan_rejects_negative_plan_cycle_consistently() -> None:
    for strategy, shuffle in (("concat", False), ("weighted", False), ("weighted", True)):
        plan = ShaftSamplePlan(
            {"a": 3, "b": 5},
            {"a": 1.0, "b": 1.0},
            strategy=strategy,
            num_samples=8,
            shuffle=shuffle,
            seed=19,
        )

        with pytest.raises(IndexError):
            plan.ref_at(0, plan_cycle=-1)


def test_sample_sampler_emits_context_for_each_plan_cycle() -> None:
    plan = ShaftSamplePlan(
        {"a": 4, "b": 4},
        {"a": 1.0, "b": 1.0},
        strategy="concat",
        shuffle=True,
        seed=7,
    )
    sampler = ShaftSampleSampler(plan, rank=0, world_size=1)

    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert all(isinstance(ref, ShaftSampleRef) for ref in first)
    assert [ref.context.draw_id for ref in first] == list(range(8))
    assert [ref.context.draw_id for ref in second] == list(range(8, 16))
    assert first != second


def test_sample_sampler_shards_positions_without_copying_plan() -> None:
    plan = ShaftSamplePlan(
        {"a": 3, "b": 1},
        {"a": 1.0, "b": 1.0},
        strategy="concat",
        shuffle=False,
        seed=3,
    )
    sampler_rank0 = ShaftSampleSampler(plan, rank=0, world_size=2)
    sampler_rank1 = ShaftSampleSampler(plan, rank=1, world_size=2)

    assert len(sampler_rank0) == len(sampler_rank1) == 2
    assert {ref.context.draw_id for ref in sampler_rank0}.isdisjoint(
        ref.context.draw_id for ref in sampler_rank1
    )


def test_sample_sampler_rejects_distributed_epoch_tail_that_would_desynchronize_ranks() -> None:
    plan = ShaftSamplePlan(
        {"a": 5},
        {"a": 1.0},
        strategy="concat",
        shuffle=False,
        seed=3,
    )
    sampler = ShaftSampleSampler(plan, rank=0, world_size=1)

    with pytest.raises(ValueError, match="unequal per-rank train step counts"):
        sampler.validate_epoch_sharding(
            per_device_batch_size=2,
            data_world_size=2,
            dataloader_drop_last=False,
        )


def test_sample_sampler_allows_sft_smaller_final_batch_when_rank_step_counts_match() -> None:
    plan = ShaftSamplePlan(
        {"a": 3},
        {"a": 1.0},
        strategy="concat",
        shuffle=False,
        seed=3,
    )
    sampler = ShaftSampleSampler(plan, rank=0, world_size=1)

    assert sampler.validate_epoch_sharding(
        per_device_batch_size=2,
        data_world_size=2,
        dataloader_drop_last=False,
    ) == 1

    batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=False)
    rank_batches = [
        list(
            BatchSamplerShard(
                batch_sampler,
                num_processes=2,
                process_index=rank,
                split_batches=False,
                even_batches=False,
            )
        )
        for rank in range(2)
    ]
    executed_draws = sorted(
        ref.context.draw_id
        for batches in rank_batches
        for batch in batches
        for ref in batch
    )

    assert [len(batches) for batches in rank_batches] == [1, 1]
    assert executed_draws == [0, 1, 2]


def test_sample_sampler_rejects_dpo_rank_local_cardinality_mismatch() -> None:
    plan = ShaftSamplePlan(
        {"a": 3},
        {"a": 1.0},
        strategy="concat",
        shuffle=False,
        seed=3,
    )
    sampler = ShaftSampleSampler(plan, rank=0, world_size=1)

    with pytest.raises(ValueError, match="equal rank-local batch cardinality"):
        sampler.validate_epoch_sharding(
            per_device_batch_size=2,
            data_world_size=2,
            dataloader_drop_last=False,
            require_equal_rank_batch_cardinality=True,
        )

    assert sampler.validate_epoch_sharding(
        per_device_batch_size=2,
        data_world_size=1,
        dataloader_drop_last=False,
        require_equal_rank_batch_cardinality=True,
    ) == 2


def test_weighted_sample_sampler_shards_one_global_draw_stream() -> None:
    plan = ShaftSamplePlan(
        {"a": 7, "b": 11},
        {"a": 1.0, "b": 2.0},
        strategy="weighted",
        num_samples=96,
        shuffle=True,
        seed=37,
    )
    rank_refs = [
        list(ShaftSampleSampler(plan, rank=rank, world_size=3))
        for rank in range(3)
    ]
    merged = sorted(
        (ref for refs in rank_refs for ref in refs),
        key=lambda ref: ref.context.draw_id,
    )

    assert [ref.context.draw_id for ref in merged] == list(range(len(plan)))
    assert merged == [plan.ref_at(position) for position in range(len(plan))]


def test_grouped_sampler_repeats_refs_and_resumes_from_plan_cycle() -> None:
    plan = ShaftSamplePlan(
        {"a": 8},
        {"a": 1.0},
        strategy="concat",
        shuffle=True,
        seed=5,
    )

    def build_sampler() -> ShaftGroupedSampleSampler:
        return ShaftGroupedSampleSampler(
            plan,
            contract=ShaftGroupedSampleContract(
                mini_repeat_count=2,
                batch_size=2,
                iteration_count=1,
                steps_per_iteration=2,
            ),
        )

    uninterrupted = build_sampler()
    epoch_zero = list(uninterrupted)
    uninterrupted.set_epoch(1)
    epoch_one = list(uninterrupted)

    resumed = build_sampler()
    resumed.set_epoch(1)

    assert len(epoch_zero) == len(epoch_one) == 32
    assert epoch_zero[0] == epoch_zero[1]
    assert epoch_zero[:4] == epoch_zero[4:8]
    assert epoch_zero != epoch_one
    assert list(resumed) == epoch_one
    assert {ref.context.plan_cycle for ref in epoch_one} == {1}
    assert min(ref.context.draw_id for ref in epoch_one) >= len(plan)


def test_grouped_contract_resolves_step_bounded_unique_sample_horizon() -> None:
    contract = ShaftGroupedSampleContract(
        mini_repeat_count=3,
        batch_size=3,
        iteration_count=1,
        steps_per_iteration=9,
    )

    assert contract.finite_sample_plan_size(
        max_steps=1,
        gradient_accumulation_steps=7,
    ) == 3
    assert contract.finite_sample_plan_size(
        max_steps=-1,
        gradient_accumulation_steps=7,
    ) is None
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        contract.finite_sample_plan_size(
            max_steps=1,
            gradient_accumulation_steps=0,
        )


def test_grouped_contract_step_horizon_matches_expanded_stream_oracle() -> None:
    for batch_size in range(1, 5):
        for iteration_count in range(1, 5):
            for steps_per_iteration in range(1, 8):
                contract = ShaftGroupedSampleContract(
                    mini_repeat_count=3,
                    batch_size=batch_size,
                    iteration_count=iteration_count,
                    steps_per_iteration=steps_per_iteration,
                )
                for max_steps in range(1, 8):
                    for gradient_accumulation_steps in range(1, 8):
                        required_microsteps = max_steps * gradient_accumulation_steps
                        expected_group_count = 0
                        oracle_microsteps = 0
                        while oracle_microsteps < required_microsteps:
                            expected_group_count += 1
                            oracle_microsteps += contract.repeat_count
                        sample_count = contract.finite_sample_plan_size(
                            max_steps=max_steps,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                        )

                        assert sample_count == expected_group_count * batch_size
                        assert sample_count is not None
                        available_microsteps = (
                            sample_count // batch_size
                        ) * contract.repeat_count
                        assert available_microsteps >= required_microsteps
                        assert (
                            available_microsteps - contract.repeat_count
                            < required_microsteps
                        )


def test_grouped_sampler_rejects_incomplete_prompt_group_instead_of_dropping_tail() -> None:
    plan = ShaftSamplePlan(
        {"a": 5},
        {"a": 1.0},
        strategy="concat",
        shuffle=True,
        seed=5,
    )

    with pytest.raises(ValueError, match="complete grouped batches"):
        ShaftGroupedSampleSampler(
            plan,
            contract=ShaftGroupedSampleContract(
                mini_repeat_count=2,
                batch_size=4,
                iteration_count=1,
                steps_per_iteration=2,
            ),
        )


def test_grouped_sampler_proves_equal_distributed_epoch_steps() -> None:
    plan = ShaftSamplePlan(
        {"a": 8},
        {"a": 1.0},
        strategy="concat",
        shuffle=True,
        seed=5,
    )
    sampler = ShaftGroupedSampleSampler(
        plan,
        contract=ShaftGroupedSampleContract(
            mini_repeat_count=2,
            batch_size=2,
            iteration_count=1,
            steps_per_iteration=2,
        ),
    )

    assert sampler.validate_epoch_sharding(
        per_device_generation_batch_size=2,
        data_world_size=2,
        dataloader_drop_last=False,
    ) == 8
    with pytest.raises(ValueError, match="distributed generation batch"):
        sampler.validate_epoch_sharding(
            per_device_generation_batch_size=1,
            data_world_size=2,
            dataloader_drop_last=False,
        )


def test_grouped_sample_contract_fingerprints_repeat_cadence() -> None:
    common = {
        "mini_repeat_count": 2,
        "batch_size": 2,
    }
    contracts = [
        ShaftGroupedSampleContract(
            **common,
            iteration_count=1,
            steps_per_iteration=4,
        ),
        ShaftGroupedSampleContract(
            **common,
            iteration_count=2,
            steps_per_iteration=2,
        ),
        ShaftGroupedSampleContract(
            mini_repeat_count=4,
            batch_size=1,
            iteration_count=1,
            steps_per_iteration=4,
        ),
    ]

    assert {contract.repeat_count for contract in contracts} == {4}
    assert len(
        {
            contract.execution_fingerprint("base-execution")
            for contract in contracts
        }
    ) == len(contracts)


def test_grouped_sampler_preserves_weighted_source_coverage_order() -> None:
    source_sizes = {"a": 5, "b": 7}
    plan = ShaftSamplePlan(
        source_sizes,
        {"a": 1.0, "b": 3.0},
        strategy="weighted",
        num_samples=64,
        shuffle=True,
        seed=19,
    )
    sampler = ShaftGroupedSampleSampler(
        plan,
        contract=ShaftGroupedSampleContract(
            mini_repeat_count=1,
            batch_size=2,
            iteration_count=1,
            steps_per_iteration=1,
        ),
    )
    rows_by_source = {name: [] for name in source_sizes}

    for ref in sampler:
        rows = rows_by_source[ref.dataset_name]
        if len(rows) < source_sizes[ref.dataset_name]:
            rows.append(ref.row_index)

    for name, size in source_sizes.items():
        assert len(rows_by_source[name]) == size
        assert set(rows_by_source[name]) == set(range(size))
