from __future__ import annotations

from collections import Counter

import pytest

from shaft.data import (
    ShaftGroupedSampleContract,
    ShaftGroupedSampleSampler,
    ShaftSamplePlan,
    ShaftSampleRef,
    ShaftSampleSchedule,
    ShaftSampleSampler,
)


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

    assert schedule.ticket_block_size == 16384
    assert abs(actual - target) / target < 0.05


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

    for block_id in range(8):
        block_start = block_id * schedule.ticket_block_size
        rare_draws = [
            draw_id
            for draw_id in range(block_start, block_start + schedule.ticket_block_size)
            if schedule.ref_at(draw_id).dataset_name == "rare"
        ]
        assert len(rare_draws) == 1
        rare_draw_residues.add(rare_draws[0] % 8)

    assert rare_draw_residues == set(range(8))


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


def test_weighted_v2_fingerprints_change_without_invalidating_legacy_modes() -> None:
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
        "4c0568630b642ad4658c7bf82ecb5f5a42c53236bd426924d4e2ae08bc32aeb7"
    )
    assert weighted.ticket_block_digest == (
        "fada3d4d247e243b9f1f5fecc33034ffe85d9f76b053cb5410c3954d0e3a8336"
    )
    assert weighted.fingerprint == (
        "dd4c51e93765175f3e695e9d0cd5bda30054705c943a21981b94edcd96489c30"
    )
    assert weighted_plan.fingerprint == (
        "27e055d1d6393680fb7d56bfb43a185613e0c8b4dcf74214a0cad933dd59af8e"
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
        ("a", 1, 12733251116385539102),
        ("b", 0, 7700560105411520951),
        ("b", 3, 4870315401550313391),
        ("b", 3, 2239024681287515773),
        ("b", 4, 15171383588411787973),
        ("b", 4, 17766263492751155199),
        ("a", 1, 12397394164509198338),
        ("b", 2, 16566888492203895366),
        ("a", 2, 11329217410363271126),
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
