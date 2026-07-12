from __future__ import annotations

import hashlib
import json

import pytest
from accelerate.data_loader import BatchSamplerShard

from shaft.data import (
    ShaftBoundedBatchPlanner,
    ShaftBoundedBatchSampler,
    ShaftBoundedBatchingSpec,
    ShaftBoundedBatchingState,
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
    max_samples: int = 4,
    max_tokens: int = 16,
    max_vision: int | None = None,
) -> ShaftBoundedBatchingSpec:
    schedule = _schedule()
    return ShaftBoundedBatchingSpec(
        data_world_size=world_size,
        buffer_size=buffer_size,
        max_samples_per_microbatch=max_samples,
        max_padded_tokens=max_tokens,
        max_vision_patches=max_vision,
        seed=23,
        sample_schedule_fingerprint=schedule.fingerprint,
        cost_fingerprint=provider.fingerprint,
    )


def _planner(provider: CountingCostProvider, **kwargs) -> ShaftBoundedBatchPlanner:
    schedule = _schedule()
    spec = _spec(provider, **kwargs)
    return ShaftBoundedBatchPlanner(
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
    with pytest.raises(ValueError, match="one anchor per data rank"):
        _spec(provider, world_size=4, buffer_size=3)


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


def test_adversarial_lengths_pack_short_samples_without_exact_sample_target() -> None:
    provider = CountingCostProvider([8, 1, 1, 1, 1, 2])
    planner = _planner(
        provider,
        buffer_size=6,
        max_samples=4,
        max_tokens=10,
    )

    plan = planner.next_global_microbatch()
    batch_lengths = sorted(len(batch.sample_refs) for batch in plan.rank_microbatches)

    assert batch_lengths == [1, 4]
    selected = set(_draw_ids(plan))
    buffered = {
        entry.sample_ref.context.draw_id for entry in plan.state_after.buffer
    }
    assert {0, 1}.issubset(selected)
    assert selected.isdisjoint(buffered)
    assert selected | buffered == set(range(6))
    assert all(batch.padded_llm_tokens <= 10 for batch in plan.rank_microbatches)


def test_equal_cost_candidates_are_balanced_across_rank_bins() -> None:
    provider = CountingCostProvider([1])
    planner = _planner(
        provider,
        world_size=4,
        buffer_size=8,
        max_samples=4,
        max_tokens=4,
    )

    plan = planner.next_global_microbatch()

    assert sorted(len(batch.sample_refs) for batch in plan.rank_microbatches) == [2, 2, 2, 2]


def test_refill_preserves_draw_multiset_without_loss_or_duplication() -> None:
    provider = CountingCostProvider([1, 2, 3, 4, 5])
    planner = _planner(provider, buffer_size=11, max_samples=3, max_tokens=15)
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


def test_oldest_entries_are_anchors_and_cannot_starve() -> None:
    provider = CountingCostProvider([7, 1, 1, 1, 1, 1])
    planner = _planner(provider, buffer_size=12, max_samples=2, max_tokens=8)
    previous_buffer: list[int] = []

    for _ in range(10):
        plan = planner.next_global_microbatch()
        emitted = set(_draw_ids(plan))
        if previous_buffer:
            assert set(previous_buffer[:2]).issubset(emitted)
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
        max_samples=3,
        max_tokens=12,
        max_vision=10,
    )

    for plan in planner.iter_global_microbatches(20):
        for batch in plan.rank_microbatches:
            assert len(batch.sample_refs) <= 3
            assert batch.padded_llm_tokens <= 12
            assert batch.vision_patches <= 10


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
    restored_state = ShaftBoundedBatchingState.from_dict(payload)

    continued = [_draw_ids(planner.next_global_microbatch()) for _ in range(10)]
    restored_provider = CountingCostProvider([1, 2, 7, 3])
    restored = ShaftBoundedBatchPlanner(
        schedule=_schedule(),
        cost_provider=restored_provider,
        spec=_spec(restored_provider, buffer_size=10, max_tokens=12),
        state=restored_state,
    )

    assert [_draw_ids(restored.next_global_microbatch()) for _ in range(10)] == continued


def test_state_fingerprint_rejects_tampering() -> None:
    provider = CountingCostProvider([1])
    state = _planner(provider).next_global_microbatch().state_after.to_dict()
    state["next_draw_id"] += 1

    with pytest.raises(ValueError, match="fingerprint"):
        ShaftBoundedBatchingState.from_dict(state)


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
        ShaftBoundedBatchingState.from_dict(payload)


def test_batch_sampler_matches_accelerate_variable_batch_sharding() -> None:
    provider = CountingCostProvider([1, 2, 5, 7])
    spec = _spec(provider, world_size=2, buffer_size=12, max_tokens=12)
    sampler = ShaftBoundedBatchSampler(
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
        assert rank_batches[0][local_step]
        assert rank_batches[1][local_step]
        assert set(rank_batches[0][local_step]).isdisjoint(rank_batches[1][local_step])


def test_first_yield_cost_work_is_bounded_by_one_atomic_planning_frame() -> None:
    provider = CountingCostProvider([1, 2, 3, 4])
    spec = _spec(provider, world_size=2, buffer_size=8, max_samples=4, max_tokens=16)
    sampler = ShaftBoundedBatchSampler(
        _schedule(),
        cost_provider=provider,
        spec=spec,
        global_microstep_count=1_000_000,
        planning_frame_size=4,
    )

    first_batch = next(iter(sampler))

    assert first_batch
    assert len(provider.calls) <= 8 + (4 - 1) * 2 * 4
    assert len(provider.calls) < 1_000_000


def test_committed_optimizer_boundary_ignores_prefetched_live_cursor() -> None:
    provider = CountingCostProvider([1, 2, 5, 7])
    spec = _spec(provider, world_size=2, buffer_size=12, max_tokens=12)
    sampler = ShaftBoundedBatchSampler(
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
    resumed = ShaftBoundedBatchSampler(
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
        max_samples=1,
        max_tokens=8,
    )
    sampler = ShaftBoundedBatchSampler(
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
