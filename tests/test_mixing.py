from __future__ import annotations

from collections import Counter

import pytest

from shaft.data import (
    ShaftGroupedSampleSampler,
    ShaftSamplePlan,
    ShaftSampleRef,
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


def test_weighted_plan_uses_probabilities_with_replacement() -> None:
    plan = ShaftSamplePlan(
        {"a": 10, "b": 10},
        {"a": 1.0, "b": 3.0},
        strategy="weighted",
        num_samples=20_000,
        shuffle=True,
        seed=7,
    )

    counts = Counter(plan.ref_at(index).dataset_name for index in range(len(plan)))

    assert counts["a"] / len(plan) == pytest.approx(0.25, abs=0.02)
    assert counts["b"] / len(plan) == pytest.approx(0.75, abs=0.02)
    assert plan.ref_at(1234) == plan.ref_at(1234)


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
            mini_repeat_count=2,
            batch_size=2,
            repeat_count=2,
            shuffle=True,
            seed=11,
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
