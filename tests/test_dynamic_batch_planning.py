from __future__ import annotations

import hashlib
import json
import logging

import pytest
from accelerate.data_loader import BatchSamplerShard

import shaft.data.dynamic_batching as dynamic_batching
from shaft.data import (
    ShaftBatchPlanner,
    ShaftPlannedBatchSampler,
    ShaftBatchPlanningSpec,
    ShaftBatchPlanningState,
    ShaftPartitionSearchBudgetExceeded,
    ShaftBufferedSample,
    ShaftSampleCost,
    ShaftSamplePlan,
    ShaftSampleSchedule,
)


pytestmark = pytest.mark.component


class CountingCostProvider:
    def __init__(
        self,
        lengths: list[int],
        *,
        vision: list[int] | None = None,
        exact: bool = True,
        fingerprint: str = "counting-cost-v1",
    ) -> None:
        self.lengths = list(lengths)
        self.vision = list(vision or [0] * len(lengths))
        self.exact = bool(exact)
        self.fingerprint = fingerprint
        self.calls: list[int] = []

    def __call__(self, sample_ref):
        draw_id = int(sample_ref.context.draw_id)
        self.calls.append(draw_id)
        index = draw_id % len(self.lengths)
        length = int(self.lengths[index])
        return ShaftSampleCost(
            llm_tokens=length,
            supervised_tokens=max(length - 1, 0),
            vision_patches=int(self.vision[index]),
            exact=self.exact,
        )


def _schedule() -> ShaftSampleSchedule:
    return ShaftSampleSchedule(
        {"ds": 1_000_000},
        {"ds": 1.0},
        strategy="concat",
        shuffle=False,
        seed=17,
    )


def _spec(
    provider: CountingCostProvider,
    *,
    world_size: int = 2,
    buffer_size: int = 8,
    local_batch_size: int = 1,
    cardinality: str = "fixed",
    max_tokens: int = 16,
    max_vision: int | None = None,
) -> ShaftBatchPlanningSpec:
    schedule = _schedule()
    return ShaftBatchPlanningSpec(
        data_world_size=world_size,
        buffer_size=buffer_size,
        cardinality=cardinality,
        per_device_microbatch_size=local_batch_size,
        max_tokens_per_microbatch=max_tokens,
        resource_budgets=(
            () if max_vision is None else (("vision_patches", max_vision),)
        ),
        seed=23,
        sample_schedule_fingerprint=schedule.fingerprint,
        cost_fingerprint=provider.fingerprint,
    )


def _planner(provider: CountingCostProvider, **kwargs) -> ShaftBatchPlanner:
    schedule = _schedule()
    spec = _spec(provider, **kwargs)
    return ShaftBatchPlanner(
        schedule=schedule,
        cost_provider=provider,
        spec=spec,
    )


def _draw_ids(plan) -> list[int]:
    return [
        ref.context.draw_id
        for batch in plan.rank_microbatches
        for ref in batch.sample_refs
    ]


def test_bounded_spec_is_duration_independent_and_validates_buffer_geometry() -> None:
    provider = CountingCostProvider([1])
    first = _spec(provider)
    second = _spec(provider)

    assert first.fingerprint == second.fingerprint
    assert "optimizer_step" not in first.to_dict()
    with pytest.raises(ValueError, match="one complete minimum global microbatch"):
        _spec(provider, world_size=4, buffer_size=3)
    with pytest.raises(ValueError, match="required_samples=8"):
        _spec(
            provider,
            world_size=4,
            buffer_size=4,
            local_batch_size=2,
            cardinality="fixed",
        )
    assert _spec(
        provider,
        world_size=4,
        buffer_size=4,
        local_batch_size=2,
        cardinality="token_budget",
    ).local_pack_count_bounds == (1, 2)
    with pytest.raises(ValueError, match="Unsupported planned batching resource"):
        ShaftBatchPlanningSpec(
            data_world_size=1,
            buffer_size=1,
            per_device_microbatch_size=1,
            max_tokens_per_microbatch=16,
            resource_budgets=(("audio_frames", 16),),
            seed=23,
            sample_schedule_fingerprint=_schedule().fingerprint,
            cost_fingerprint=provider.fingerprint,
        )


@pytest.mark.parametrize(
    "legacy_version",
    [
        "shaft-bounded-cost-batching-v1",
        "shaft-bounded-fixed-batching-v2",
        "shaft-bounded-batching-v3",
    ],
)
def test_legacy_bounded_spec_is_rejected(legacy_version: str) -> None:
    provider = CountingCostProvider([1])
    payload = _spec(provider).to_dict()
    payload["version"] = legacy_version

    with pytest.raises(ValueError, match="Unsupported planned batching spec version"):
        ShaftBatchPlanningSpec.from_dict(payload)


@pytest.mark.parametrize(
    "legacy_version",
    [
        "shaft-bounded-cost-batching-v1",
        "shaft-bounded-fixed-batching-v2",
        "shaft-bounded-batching-v3",
    ],
)
def test_legacy_bounded_state_is_rejected(legacy_version: str) -> None:
    with pytest.raises(ValueError, match="Unsupported planned batching state version"):
        ShaftBatchPlanningState.from_dict({"version": legacy_version})


def test_sample_schedule_prefix_does_not_change_with_finite_plan_horizon() -> None:
    short = ShaftSamplePlan(
        {"a": 7, "b": 3},
        {"a": 2.0, "b": 1.0},
        strategy="weighted",
        num_samples=10,
        shuffle=True,
        seed=31,
    )
    long = ShaftSamplePlan(
        {"a": 7, "b": 3},
        {"a": 2.0, "b": 1.0},
        strategy="weighted",
        num_samples=10_000,
        shuffle=True,
        seed=31,
    )

    assert short.fingerprint != long.fingerprint
    assert short.schedule.fingerprint == long.schedule.fingerprint
    assert [short.schedule.ref_at(index) for index in range(100)] == [
        long.schedule.ref_at(index) for index in range(100)
    ]


def test_first_microstep_only_costs_the_bounded_lookahead() -> None:
    provider = CountingCostProvider([1])
    planner = _planner(provider, buffer_size=64)

    plan = planner.next_global_microbatch()

    assert provider.calls == list(range(64))
    assert len(plan.state_after.buffer) <= 64
    assert plan.state_after.next_draw_id == 64


def test_batch_size_one_groups_similar_rank_costs_without_local_padding() -> None:
    provider = CountingCostProvider([8, 1, 1, 1, 1, 2])
    planner = _planner(
        provider,
        buffer_size=6,
        max_tokens=10,
    )

    plan = planner.next_global_microbatch()
    batch_lengths = sorted(len(batch.sample_refs) for batch in plan.rank_microbatches)

    assert batch_lengths == [1, 1]
    selected = set(_draw_ids(plan))
    buffered = {
        entry.sample_ref.context.draw_id for entry in plan.state_after.buffer
    }
    assert selected == {0, 5}
    assert selected.isdisjoint(buffered)
    assert selected | buffered == set(range(6))
    assert all(batch.padded_llm_tokens <= 10 for batch in plan.rank_microbatches)


def test_fixed_local_batch_size_is_filled_exactly() -> None:
    provider = CountingCostProvider([1])
    planner = _planner(
        provider,
        world_size=4,
        buffer_size=8,
        local_batch_size=2,
        max_tokens=4,
    )

    plan = planner.next_global_microbatch()

    assert sorted(len(batch.sample_refs) for batch in plan.rank_microbatches) == [2, 2, 2, 2]


def test_token_budget_varies_local_batch_size() -> None:
    provider = CountingCostProvider([9, 1, 1])
    planner = _planner(
        provider,
        world_size=2,
        buffer_size=3,
        local_batch_size=2,
        cardinality="token_budget",
        max_tokens=10,
    )

    plan = planner.next_global_microbatch()

    assert sorted(len(batch.sample_refs) for batch in plan.rank_microbatches) == [1, 2]
    assert plan.stats.sample_count == 3
    assert plan.state_after.emitted_samples == 3
    assert all(batch.padded_llm_tokens <= 10 for batch in plan.rank_microbatches)


def test_token_budget_respects_batch_cap() -> None:
    provider = CountingCostProvider([12, 6, 4, 3, 1, 1, 1, 1, 1, 1])
    planner = _planner(
        provider,
        world_size=4,
        buffer_size=10,
        local_batch_size=4,
        cardinality="token_budget",
        max_tokens=12,
    )

    plan = planner.next_global_microbatch()

    assert sorted(len(batch.sample_refs) for batch in plan.rank_microbatches) == [
        1,
        2,
        3,
        4,
    ]
    assert all(batch.padded_llm_tokens <= 12 for batch in plan.rank_microbatches)


def test_token_budget_allows_a_long_sample_that_fits_only_as_batch_one() -> None:
    provider = CountingCostProvider([9, 1, 1])
    planner = _planner(
        provider,
        world_size=2,
        buffer_size=3,
        local_batch_size=2,
        cardinality="token_budget",
        max_tokens=10,
    )

    plan = planner.next_global_microbatch()

    long_batch = next(
        batch
        for batch in plan.rank_microbatches
        if any(cost.llm_tokens == 9 for cost in batch.sample_costs)
    )
    assert len(long_batch.sample_refs) == 1


def test_token_budget_uses_aggregate_resource_budget_for_cardinality() -> None:
    provider = CountingCostProvider([1, 1, 1], vision=[9, 2, 2])
    planner = _planner(
        provider,
        world_size=2,
        buffer_size=3,
        local_batch_size=2,
        cardinality="token_budget",
        max_tokens=10,
        max_vision=10,
    )

    plan = planner.next_global_microbatch()

    assert sorted(len(batch.sample_refs) for batch in plan.rank_microbatches) == [1, 2]
    assert all(batch.vision_patches <= 10 for batch in plan.rank_microbatches)


def test_token_budget_fallback_avoids_underfilling() -> None:
    provider = CountingCostProvider(
        [4, 5, 4, 3],
        vision=[4, 6, 8, 2],
    )
    planner = _planner(
        provider,
        world_size=2,
        buffer_size=4,
        local_batch_size=2,
        cardinality="token_budget",
        max_tokens=10,
        max_vision=10,
    )

    plan = planner.next_global_microbatch()

    assert sorted(len(batch.sample_refs) for batch in plan.rank_microbatches) == [2, 2]
    assert all(batch.padded_llm_tokens <= 10 for batch in plan.rank_microbatches)
    assert all(batch.vision_patches <= 10 for batch in plan.rank_microbatches)


def test_full_partition_search_budget_is_a_safe_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dynamic_batching, "_FULL_PARTITION_SEARCH_NODE_LIMIT", 0)
    provider = CountingCostProvider([4, 5, 4, 3], vision=[4, 6, 8, 2])
    fixed = _planner(
        provider,
        world_size=2,
        buffer_size=4,
        local_batch_size=2,
        max_tokens=10,
        max_vision=10,
    )
    with pytest.raises(ShaftPartitionSearchBudgetExceeded):
        fixed.next_global_microbatch()

    variable_provider = CountingCostProvider(
        [4, 5, 4, 3],
        vision=[4, 6, 8, 2],
    )
    variable = _planner(
        variable_provider,
        world_size=2,
        buffer_size=4,
        local_batch_size=2,
        cardinality="token_budget",
        max_tokens=10,
        max_vision=10,
    )
    plan = variable.next_global_microbatch()
    assert plan.stats.sample_count == 3
    assert all(batch.padded_llm_tokens <= 10 for batch in plan.rank_microbatches)
    assert all(batch.vision_patches <= 10 for batch in plan.rank_microbatches)


def test_refill_preserves_draw_multiset_without_loss_or_duplication() -> None:
    provider = CountingCostProvider([1, 2, 3, 4, 5])
    planner = _planner(provider, buffer_size=11, max_tokens=15)
    emitted: list[int] = []

    for _ in range(50):
        plan = planner.next_global_microbatch()
        emitted.extend(_draw_ids(plan))
        buffered = [
            entry.sample_ref.context.draw_id for entry in plan.state_after.buffer
        ]
        assert len(buffered) <= 11
        assert len(buffered) == len(set(buffered))
        assert set(emitted).isdisjoint(buffered)
        assert sorted([*emitted, *buffered]) == list(
            range(plan.state_after.next_draw_id)
        )


def test_token_budget_long_stream_invariants() -> None:
    provider = CountingCostProvider(
        [12, 6, 4, 3, 2, 1, 1],
        vision=[10, 6, 5, 4, 3, 1, 1],
    )
    planner = _planner(
        provider,
        world_size=3,
        buffer_size=18,
        local_batch_size=4,
        cardinality="token_budget",
        max_tokens=12,
        max_vision=12,
    )
    emitted: list[int] = []
    previous_oldest: int | None = None
    observed_sizes: set[int] = set()

    for _ in range(40):
        plan = planner.next_global_microbatch()
        draw_ids = _draw_ids(plan)
        if previous_oldest is not None:
            assert previous_oldest in draw_ids
        emitted.extend(draw_ids)
        for batch in plan.rank_microbatches:
            observed_sizes.add(len(batch.sample_refs))
            assert 1 <= len(batch.sample_refs) <= 4
            assert batch.padded_llm_tokens <= 12
            assert batch.vision_patches <= 12
        buffered = [
            entry.sample_ref.context.draw_id for entry in plan.state_after.buffer
        ]
        assert set(emitted).isdisjoint(buffered)
        assert sorted([*emitted, *buffered]) == list(
            range(plan.state_after.next_draw_id)
        )
        previous_oldest = buffered[0] if buffered else None

    assert min(observed_sizes) == 1
    assert max(observed_sizes) == 4


def test_oldest_entries_are_anchors_and_cannot_starve() -> None:
    provider = CountingCostProvider([7, 1, 1, 1, 1, 1])
    planner = _planner(provider, buffer_size=12, max_tokens=8)
    previous_buffer: list[int] = []

    for _ in range(10):
        plan = planner.next_global_microbatch()
        emitted = set(_draw_ids(plan))
        if previous_buffer:
            assert previous_buffer[0] in emitted
        previous_buffer = [
            entry.sample_ref.context.draw_id for entry in plan.state_after.buffer
        ]


def test_every_emitted_batch_respects_multidimensional_hard_caps() -> None:
    provider = CountingCostProvider(
        [6, 5, 4, 3, 2, 1],
        vision=[7, 6, 5, 4, 3, 2],
    )
    planner = _planner(
        provider,
        buffer_size=12,
        local_batch_size=2,
        max_tokens=12,
        max_vision=10,
    )

    for plan in planner.iter_global_microbatches(20):
        for batch in plan.rank_microbatches:
            assert len(batch.sample_refs) == 2
            assert batch.padded_llm_tokens <= 12
            assert batch.vision_patches <= 10


def test_exact_fallback_finds_a_feasible_fixed_partition_missed_by_greedy() -> None:
    # The closest-cost anchor and least-loaded placement heuristic dead-end on
    # this layout, although ((0, 1), (3, 6)) is a valid fixed partition.
    provider = CountingCostProvider(
        [5, 9, 3, 8, 2, 2, 10],
        vision=[14, 6, 12, 10, 14, 15, 9],
    )
    planner = _planner(
        provider,
        world_size=2,
        buffer_size=7,
        local_batch_size=2,
        max_tokens=20,
        max_vision=20,
    )

    plan = planner.next_global_microbatch()

    assert 0 in _draw_ids(plan)
    assert sorted(len(batch.sample_refs) for batch in plan.rank_microbatches) == [2, 2]
    assert all(batch.padded_llm_tokens <= 20 for batch in plan.rank_microbatches)
    assert all(batch.vision_patches <= 20 for batch in plan.rank_microbatches)


@pytest.mark.parametrize(
    ("provider", "kwargs", "message"),
    [
        (CountingCostProvider([17]), {"max_tokens": 16}, "llm_tokens=17"),
        (
            CountingCostProvider([1], vision=[11]),
            {"max_vision": 10},
            "vision_patches=11",
        ),
        (CountingCostProvider([1], exact=False), {}, "is inexact"),
    ],
)
def test_invalid_sample_fails_at_first_observation_without_scanning_further(
    provider: CountingCostProvider,
    kwargs: dict[str, int],
    message: str,
) -> None:
    planner = _planner(provider, buffer_size=8, **kwargs)

    with pytest.raises(ValueError, match=message):
        planner.next_global_microbatch()

    assert provider.calls == [0]


def test_later_oversize_sample_stops_at_its_first_observation() -> None:
    provider = CountingCostProvider([1, 17, 1])
    planner = _planner(provider, buffer_size=8, max_tokens=16)

    with pytest.raises(ValueError, match="draw_id=1.*llm_tokens=17"):
        planner.next_global_microbatch()

    assert provider.calls == [0, 1]


def test_same_contract_and_state_are_deterministic_without_locking_one_layout() -> None:
    first_provider = CountingCostProvider([8, 1, 2, 3, 5])
    second_provider = CountingCostProvider([8, 1, 2, 3, 5])
    first = _planner(first_provider, buffer_size=16, max_tokens=16)
    second = _planner(second_provider, buffer_size=16, max_tokens=16)

    first_plans = [first.next_global_microbatch() for _ in range(12)]
    second_plans = [second.next_global_microbatch() for _ in range(12)]

    assert [_draw_ids(plan) for plan in first_plans] == [
        _draw_ids(plan) for plan in second_plans
    ]
    assert first.state == second.state


def test_state_json_roundtrip_continues_exact_stream() -> None:
    provider = CountingCostProvider([1, 2, 7, 3])
    planner = _planner(provider, buffer_size=10, max_tokens=12)
    _ = tuple(planner.iter_global_microbatches(7))
    payload = json.loads(json.dumps(planner.state.to_dict()))
    restored_state = ShaftBatchPlanningState.from_dict(payload)

    continued = [_draw_ids(planner.next_global_microbatch()) for _ in range(10)]
    restored_provider = CountingCostProvider([1, 2, 7, 3])
    restored = ShaftBatchPlanner(
        schedule=_schedule(),
        cost_provider=restored_provider,
        spec=_spec(restored_provider, buffer_size=10, max_tokens=12),
        state=restored_state,
    )

    assert [_draw_ids(restored.next_global_microbatch()) for _ in range(10)] == continued


def test_token_budget_state_roundtrip_continues_exact_variable_stream() -> None:
    provider = CountingCostProvider([9, 1, 1])
    planner = _planner(
        provider,
        world_size=2,
        buffer_size=5,
        local_batch_size=2,
        cardinality="token_budget",
        max_tokens=10,
    )
    _ = tuple(planner.iter_global_microbatches(7))
    restored_state = ShaftBatchPlanningState.from_dict(
        json.loads(json.dumps(planner.state.to_dict()))
    )
    continued = [_draw_ids(planner.next_global_microbatch()) for _ in range(10)]

    restored_provider = CountingCostProvider([9, 1, 1])
    restored = ShaftBatchPlanner(
        schedule=_schedule(),
        cost_provider=restored_provider,
        spec=_spec(
            restored_provider,
            world_size=2,
            buffer_size=5,
            local_batch_size=2,
            cardinality="token_budget",
            max_tokens=10,
        ),
        state=restored_state,
    )

    assert [_draw_ids(restored.next_global_microbatch()) for _ in range(10)] == continued


def test_state_fingerprint_rejects_tampering() -> None:
    provider = CountingCostProvider([1])
    state = _planner(provider).next_global_microbatch().state_after.to_dict()
    state["next_draw_id"] += 1

    with pytest.raises(ValueError, match="fingerprint"):
        ShaftBatchPlanningState.from_dict(state)


def test_state_semantics_reject_rehashed_draw_conservation_tampering() -> None:
    provider = CountingCostProvider([1])
    payload = _planner(provider).next_global_microbatch().state_after.to_dict()
    payload["next_draw_id"] += 1
    unsigned = dict(payload)
    unsigned.pop("fingerprint")
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="draw conservation"):
        ShaftBatchPlanningState.from_dict(payload)


def test_state_semantics_reject_rehashed_fixed_cardinality_progress_tampering() -> None:
    provider = CountingCostProvider([1])
    spec = _spec(provider)
    payload = _planner(provider).next_global_microbatch().state_after.to_dict()
    payload["emitted_samples"] += spec.data_world_size
    payload["next_draw_id"] += spec.data_world_size
    unsigned = dict(payload)
    unsigned.pop("fingerprint")
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    state = ShaftBatchPlanningState.from_dict(payload)
    with pytest.raises(ValueError, match="logical segments.*physical packs"):
        state.validate_against_spec(spec)


def test_state_rejects_an_oldest_draw_that_should_already_be_emitted() -> None:
    provider = CountingCostProvider([1])
    spec = _spec(provider, world_size=2)
    schedule = _schedule()
    state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=1,
        next_draw_id=4,
        emitted_samples=2,
        buffer=tuple(
            ShaftBufferedSample(
                sample_ref=schedule.ref_at(draw_id),
                cost=provider(schedule.ref_at(draw_id)),
            )
            for draw_id in (0, 3)
        ),
    )

    with pytest.raises(ValueError, match="already have been emitted"):
        state.validate_against_spec(spec)


@pytest.mark.parametrize("local_batch_size", [1, 2])
def test_batch_sampler_matches_accelerate_fixed_batch_sharding(
    local_batch_size: int,
) -> None:
    provider = CountingCostProvider([1, 2, 5, 7])
    spec = _spec(
        provider,
        world_size=2,
        buffer_size=12,
        local_batch_size=local_batch_size,
        max_tokens=16,
    )
    sampler = ShaftPlannedBatchSampler(
        _schedule(),
        cost_provider=provider,
        spec=spec,
        global_microstep_count=6,
        planning_frame_size=2,
    )
    rank_batches = [
        list(
            BatchSamplerShard(
                sampler,
                num_processes=2,
                process_index=rank,
                split_batches=False,
                even_batches=False,
            )
        )
        for rank in range(2)
    ]

    assert [len(batches) for batches in rank_batches] == [6, 6]
    for local_step in range(6):
        assert len(rank_batches[0][local_step]) == local_batch_size
        assert len(rank_batches[1][local_step]) == local_batch_size
        assert set(rank_batches[0][local_step]).isdisjoint(rank_batches[1][local_step])


def test_batch_sampler_shards_variable_token_budget_batches_without_rank_drift() -> None:
    provider = CountingCostProvider([9, 1, 1])
    spec = _spec(
        provider,
        world_size=2,
        buffer_size=5,
        local_batch_size=2,
        cardinality="token_budget",
        max_tokens=10,
    )
    sampler = ShaftPlannedBatchSampler(
        _schedule(),
        cost_provider=provider,
        spec=spec,
        global_microstep_count=6,
        planning_frame_size=2,
    )
    rank_batches = [
        list(
            BatchSamplerShard(
                sampler,
                num_processes=2,
                process_index=rank,
                split_batches=False,
                even_batches=False,
            )
        )
        for rank in range(2)
    ]

    assert [len(batches) for batches in rank_batches] == [6, 6]
    observed_sizes = {
        len(batch) for batches in rank_batches for batch in batches
    }
    assert observed_sizes == {1, 2}
    for local_step in range(6):
        assert set(rank_batches[0][local_step]).isdisjoint(rank_batches[1][local_step])


@pytest.mark.parametrize("reuse_preflight", [False, True])
def test_first_yield_cost_work_is_bounded_by_one_atomic_planning_frame(
    reuse_preflight: bool,
) -> None:
    provider = CountingCostProvider([1, 2, 3, 4])
    spec = _spec(provider, world_size=2, buffer_size=8, max_tokens=16)
    preflight_plan = None
    if reuse_preflight:
        preflight_plan = ShaftBatchPlanner(
            schedule=_schedule(),
            cost_provider=provider,
            spec=spec,
        ).next_global_microbatch()
    sampler = ShaftPlannedBatchSampler(
        _schedule(),
        cost_provider=provider,
        spec=spec,
        global_microstep_count=1_000_000,
        planning_frame_size=4,
        preflight_plan=preflight_plan,
    )

    first_batch = next(iter(sampler))

    assert first_batch
    expected_calls = 8 + (4 - 1) * 2 * spec.per_device_microbatch_size
    assert len(provider.calls) == expected_calls
    assert len(provider.calls) < 1_000_000


def test_planner_summary_reports_only_planned_microsteps(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="shaft.data.sampler")
    provider = CountingCostProvider([1])
    sampler = ShaftPlannedBatchSampler(
        _schedule(),
        cost_provider=provider,
        spec=_spec(provider, world_size=2, buffer_size=8),
        global_microstep_count=6,
        planning_frame_size=2,
    )

    iterator = iter(sampler)
    next(iterator)
    iterator.close()

    summaries = [
        record.getMessage()
        for record in caplog.records
            if "[batch-plan-summary]" in record.getMessage()
    ]
    assert len(summaries) == 1
    assert "microsteps=2" in summaries[0]


@pytest.mark.parametrize("local_batch_size", [1, 2])
@pytest.mark.parametrize("resume_after_frame", [False, True])
def test_preflight_reuse_matches_replanned_stream_and_state(
    local_batch_size: int,
    resume_after_frame: bool,
) -> None:
    seed_provider = CountingCostProvider([1, 2, 5, 7])
    spec = _spec(
        seed_provider,
        world_size=2,
        buffer_size=12,
        local_batch_size=local_batch_size,
        max_tokens=16,
    )
    initial_state = None
    if resume_after_frame:
        seed_planner = ShaftBatchPlanner(
            schedule=_schedule(),
            cost_provider=seed_provider,
            spec=spec,
        )
        seed_planner.next_global_microbatch()
        initial_state = seed_planner.next_global_microbatch().state_after

    initial_microstep = 0 if initial_state is None else initial_state.global_microstep
    total_microsteps = initial_microstep + 4
    replanned_provider = CountingCostProvider([1, 2, 5, 7])
    replanned = ShaftPlannedBatchSampler(
        _schedule(),
        cost_provider=replanned_provider,
        spec=spec,
        global_microstep_count=total_microsteps,
        planning_frame_size=2,
        initial_state=initial_state,
    )
    replanned_batches = list(replanned)

    reused_provider = CountingCostProvider([1, 2, 5, 7])
    preflight_plan = ShaftBatchPlanner(
        schedule=_schedule(),
        cost_provider=reused_provider,
        spec=spec,
        state=initial_state,
    ).next_global_microbatch()
    reused = ShaftPlannedBatchSampler(
        _schedule(),
        cost_provider=reused_provider,
        spec=spec,
        global_microstep_count=total_microsteps,
        planning_frame_size=2,
        initial_state=initial_state,
        preflight_plan=preflight_plan,
    )

    assert list(reused) == replanned_batches
    assert reused.latest_planned_state == replanned.latest_planned_state


def test_committed_optimizer_boundary_ignores_prefetched_live_cursor() -> None:
    provider = CountingCostProvider([1, 2, 5, 7])
    spec = _spec(provider, world_size=2, buffer_size=12, max_tokens=12)
    sampler = ShaftPlannedBatchSampler(
        _schedule(),
        cost_provider=provider,
        spec=spec,
        global_microstep_count=8,
        planning_frame_size=2,
    )
    uninterrupted = list(sampler)
    committed = sampler.commit_global_microstep(2)

    assert committed.global_microstep == 2
    assert committed.global_microstep < sampler.latest_planned_state.global_microstep

    resumed_provider = CountingCostProvider([1, 2, 5, 7])
    resumed = ShaftPlannedBatchSampler(
        _schedule(),
        cost_provider=resumed_provider,
        spec=_spec(resumed_provider, world_size=2, buffer_size=12, max_tokens=12),
        global_microstep_count=8,
        planning_frame_size=2,
        initial_state=committed,
    )

    batches_per_optimizer_step = 2 * 2
    assert list(resumed) == uninterrupted[batches_per_optimizer_step:]


def test_planning_frame_balances_cumulative_rank_load() -> None:
    provider = CountingCostProvider([8, 1, 8, 1])
    spec = _spec(
        provider,
        world_size=2,
        buffer_size=2,
        local_batch_size=1,
        max_tokens=8,
    )
    sampler = ShaftPlannedBatchSampler(
        _schedule(),
        cost_provider=provider,
        spec=spec,
        global_microstep_count=2,
        planning_frame_size=2,
    )

    batches = list(sampler)
    rank_loads = [0, 0]
    for microstep in range(2):
        for rank in range(2):
            batch = batches[microstep * 2 + rank]
            rank_loads[rank] += sum(
                provider.lengths[ref.context.draw_id % len(provider.lengths)]
                for ref in batch
            )

    assert rank_loads == [9, 9]
