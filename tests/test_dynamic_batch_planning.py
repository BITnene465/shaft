from __future__ import annotations

from collections import Counter

import pytest
from accelerate.data_loader import BatchSamplerShard

from shaft.data import (
    ShaftBatchPlanningSignature,
    ShaftDynamicBatchPlanner,
    ShaftDynamicPartitionSearchBudgetExceeded,
    ShaftDynamicBatchPlanningContract,
    ShaftDynamicBatchPlanningSpec,
    ShaftDynamicBatchSampler,
    ShaftRowInvariantCostProvider,
    ShaftSampleCost,
    ShaftSamplePlan,
)


pytestmark = pytest.mark.component


def _build_plan(sample_count: int) -> ShaftSamplePlan:
    return ShaftSamplePlan(
        {"dataset": sample_count},
        {"dataset": 1.0},
        strategy="concat",
        shuffle=False,
        seed=17,
    )


def _build_provider(
    lengths: list[int],
    *,
    supervised_tokens: list[int] | None = None,
    vision_patches: list[int] | None = None,
    exact: bool = True,
) -> ShaftRowInvariantCostProvider:
    supervised_tokens = supervised_tokens or [max(length - 1, 0) for length in lengths]
    vision_patches = vision_patches or [length * 2 for length in lengths]
    return ShaftRowInvariantCostProvider(
        {
            ("dataset", index): ShaftSampleCost(
                llm_tokens=length,
                supervised_tokens=supervised_tokens[index],
                vision_patches=vision_patches[index],
                exact=exact,
            )
            for index, length in enumerate(lengths)
        },
        fingerprint=f"dynamic-costs-exact-{exact}",
    )


def _build_spec(
    plan: ShaftSamplePlan,
    *,
    optimizer_step_count: int,
    data_world_size: int,
    gradient_accumulation_steps: int,
    max_samples_per_microbatch: int,
    max_padded_tokens: int,
    target_samples: int | None = None,
    target_supervised_tokens: int | None = None,
    max_vision_patches: int | None = None,
    planning_window: int = 64,
) -> ShaftDynamicBatchPlanningSpec:
    return ShaftDynamicBatchPlanningSpec.from_plan(
        plan,
        optimizer_step_count=optimizer_step_count,
        data_world_size=data_world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_samples_per_microbatch=max_samples_per_microbatch,
        max_padded_tokens=max_padded_tokens,
        max_vision_patches=max_vision_patches,
        target_samples=target_samples,
        target_supervised_tokens=target_supervised_tokens,
        planning_window=planning_window,
        seed=23,
        rank_balance=True,
    )


def test_dynamic_contract_is_the_single_draw_horizon_source() -> None:
    compatibility = ShaftDynamicBatchPlanningContract.resolve(
        optimizer_step_count=3,
        per_device_train_batch_size=2,
        data_world_size=2,
        gradient_accumulation_steps=2,
        max_samples_per_microbatch=4,
        max_padded_tokens=100,
        max_vision_patches=None,
        target_samples=None,
        target_supervised_tokens=None,
        planning_window=16,
        seed=7,
        rank_balance=True,
    )
    token_bounded = ShaftDynamicBatchPlanningContract.resolve(
        optimizer_step_count=3,
        per_device_train_batch_size=2,
        data_world_size=2,
        gradient_accumulation_steps=2,
        max_samples_per_microbatch=4,
        max_padded_tokens=100,
        max_vision_patches=None,
        target_samples=None,
        target_supervised_tokens=20,
        planning_window=10,
        seed=7,
        rank_balance=True,
    )

    assert compatibility.target_samples == 8
    assert compatibility.draw_capacity_per_optimizer_step == 8
    assert compatibility.sample_plan_horizon == 24
    assert token_bounded.draw_capacity_per_optimizer_step == 10
    assert token_bounded.sample_plan_horizon == 30


def test_dynamic_token_target_rejects_weighted_unshuffled_sample_plan() -> None:
    plan = ShaftSamplePlan(
        {"a": 10, "b": 10},
        {"a": 1.0, "b": 1.0},
        strategy="weighted",
        num_samples=12,
        shuffle=False,
        seed=17,
    )
    spec = _build_spec(
        plan,
        optimizer_step_count=2,
        data_world_size=1,
        gradient_accumulation_steps=2,
        max_samples_per_microbatch=3,
        max_padded_tokens=12,
        target_supervised_tokens=4,
        planning_window=6,
    )

    with pytest.raises(ValueError, match="weighted, unshuffled"):
        ShaftDynamicBatchPlanner(
            plan=plan,
            cost_provider=ShaftRowInvariantCostProvider(
                {
                    **{
                        ("a", index): ShaftSampleCost(
                            llm_tokens=2,
                            supervised_tokens=1,
                            exact=True,
                        )
                        for index in range(10)
                    },
                    **{
                        ("b", index): ShaftSampleCost(
                            llm_tokens=2,
                            supervised_tokens=1,
                            exact=True,
                        )
                        for index in range(10)
                    },
                },
                fingerprint="weighted-unshuffled-v1",
            ),
            spec=spec,
        )


def test_dynamic_planner_executes_8_1_1_1_1_2_as_variable_microbatches() -> None:
    lengths = [8, 1, 1, 1, 1, 2]
    plan = _build_plan(len(lengths))
    planner = ShaftDynamicBatchPlanner(
        plan=plan,
        cost_provider=_build_provider(lengths),
        spec=_build_spec(
            plan,
            optimizer_step_count=1,
            data_world_size=1,
            gradient_accumulation_steps=2,
            max_samples_per_microbatch=5,
            max_padded_tokens=10,
            target_samples=6,
        ),
    )

    (optimizer_batch,) = tuple(planner.iter_optimizer_steps())
    local_batches = [
        local_batch
        for microstep in optimizer_batch.microsteps
        for local_batch in microstep.rank_microbatches
    ]

    assert sorted(len(batch.sample_refs) for batch in local_batches) == [1, 5]
    assert sorted(
        sorted(cost.llm_tokens for cost in batch.sample_costs)
        for batch in local_batches
    ) == [[1, 1, 1, 1, 2], [8]]
    assert all(batch.padded_llm_tokens <= 10 for batch in local_batches)
    assert Counter(
        ref.context.draw_id for ref in optimizer_batch.sample_refs
    ) == Counter(range(6))
    assert optimizer_batch.stats.min_local_batch_size == 1
    assert optimizer_batch.stats.max_local_batch_size == 5
    assert optimizer_batch.stats.padding_ratio == pytest.approx(1.0 - 14 / 18)
    assert (
        optimizer_batch.stats.padding_ratio
        < optimizer_batch.stats.baseline_padding_ratio
    )


def test_dynamic_planner_pairs_similar_variable_batches_across_ranks() -> None:
    lengths = [8, 1, 1, 1, 1, 2, 8, 1, 1, 1, 1, 2]
    plan = _build_plan(len(lengths))
    planner = ShaftDynamicBatchPlanner(
        plan=plan,
        cost_provider=_build_provider(lengths),
        spec=_build_spec(
            plan,
            optimizer_step_count=1,
            data_world_size=2,
            gradient_accumulation_steps=2,
            max_samples_per_microbatch=5,
            max_padded_tokens=10,
            target_samples=12,
        ),
    )

    (optimizer_batch,) = tuple(planner.iter_optimizer_steps())

    assert len(optimizer_batch.microsteps) == 2
    for microstep in optimizer_batch.microsteps:
        rank_costs = [batch.padded_llm_tokens for batch in microstep.rank_microbatches]
        assert rank_costs[0] == rank_costs[1]
    assert optimizer_batch.stats.max_rank_cost_skew == pytest.approx(0.0)


def test_dynamic_planner_fallback_finds_feasible_multidimensional_partition() -> None:
    lengths = [7, 1, 5]
    plan = _build_plan(len(lengths))
    planner = ShaftDynamicBatchPlanner(
        plan=plan,
        cost_provider=_build_provider(
            lengths,
            vision_patches=[2, 7, 3],
        ),
        spec=_build_spec(
            plan,
            optimizer_step_count=1,
            data_world_size=1,
            gradient_accumulation_steps=2,
            max_samples_per_microbatch=3,
            max_padded_tokens=20,
            max_vision_patches=7,
            target_samples=3,
        ),
    )

    (optimizer_batch,) = tuple(planner.iter_optimizer_steps())
    local_batches = [
        batch
        for microstep in optimizer_batch.microsteps
        for batch in microstep.rank_microbatches
    ]

    assert sorted(
        sorted(cost.vision_patches for cost in batch.sample_costs)
        for batch in local_batches
    ) == [[2, 3], [7]]


def test_dynamic_fallback_distinguishes_search_budget_from_infeasibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lengths = [7, 1, 5]
    plan = _build_plan(len(lengths))
    planner = ShaftDynamicBatchPlanner(
        plan=plan,
        cost_provider=_build_provider(
            lengths,
            vision_patches=[2, 7, 3],
        ),
        spec=_build_spec(
            plan,
            optimizer_step_count=1,
            data_world_size=1,
            gradient_accumulation_steps=2,
            max_samples_per_microbatch=3,
            max_padded_tokens=20,
            max_vision_patches=7,
            target_samples=3,
        ),
    )
    monkeypatch.setattr(
        "shaft.data.dynamic_batching._DYNAMIC_PARTITION_SEARCH_NODE_LIMIT",
        0,
    )

    with pytest.raises(ShaftDynamicPartitionSearchBudgetExceeded, match="node limit"):
        tuple(planner.iter_optimizer_steps())


def test_dynamic_planner_rejects_infeasible_aggregate_vision_partition() -> None:
    lengths = [2, 2, 2]
    plan = _build_plan(len(lengths))
    planner = ShaftDynamicBatchPlanner(
        plan=plan,
        cost_provider=_build_provider(
            lengths,
            vision_patches=[4, 4, 2],
        ),
        spec=_build_spec(
            plan,
            optimizer_step_count=1,
            data_world_size=1,
            gradient_accumulation_steps=2,
            max_samples_per_microbatch=2,
            max_padded_tokens=8,
            max_vision_patches=5,
            target_samples=3,
        ),
    )

    with pytest.raises(ValueError, match="could not partition"):
        tuple(planner.iter_optimizer_steps())


@pytest.mark.parametrize("max_vision_patches", [None, 1000])
def test_dynamic_fallback_ignores_nonbinding_vision_for_feasibility(
    max_vision_patches: int | None,
) -> None:
    lengths = [2] * 8 + [1] * 16
    plan = _build_plan(len(lengths))
    planner = ShaftDynamicBatchPlanner(
        plan=plan,
        cost_provider=_build_provider(
            lengths,
            vision_patches=list(range(1, 25)),
        ),
        spec=_build_spec(
            plan,
            optimizer_step_count=1,
            data_world_size=1,
            gradient_accumulation_steps=8,
            max_samples_per_microbatch=4,
            max_padded_tokens=4,
            max_vision_patches=max_vision_patches,
            target_samples=24,
        ),
    )

    (optimizer_batch,) = tuple(planner.iter_optimizer_steps())
    local_batches = [
        batch
        for microstep in optimizer_batch.microsteps
        for batch in microstep.rank_microbatches
    ]

    assert sorted(len(batch.sample_refs) for batch in local_batches) == [
        2,
        2,
        2,
        2,
        4,
        4,
        4,
        4,
    ]


def test_dynamic_fallback_finds_feasible_partition_with_active_vision_cap() -> None:
    lengths = [2] * 8 + [1] * 16
    vision_patches = list(range(1, 25))
    plan = _build_plan(len(lengths))
    planner = ShaftDynamicBatchPlanner(
        plan=plan,
        cost_provider=_build_provider(
            lengths,
            vision_patches=vision_patches,
        ),
        spec=_build_spec(
            plan,
            optimizer_step_count=1,
            data_world_size=1,
            gradient_accumulation_steps=8,
            max_samples_per_microbatch=4,
            max_padded_tokens=4,
            max_vision_patches=70,
            target_samples=24,
        ),
    )

    (optimizer_batch,) = tuple(planner.iter_optimizer_steps())
    local_batches = [
        batch
        for microstep in optimizer_batch.microsteps
        for batch in microstep.rank_microbatches
    ]

    assert sorted(len(batch.sample_refs) for batch in local_batches) == [
        2,
        2,
        2,
        2,
        4,
        4,
        4,
        4,
    ]
    assert max(batch.padded_llm_tokens for batch in local_batches) <= 4
    assert max(batch.vision_patches for batch in local_batches) <= 70


def test_dynamic_fallback_handles_feasible_depth_beyond_python_recursion_limit() -> None:
    lengths = [500] * 8 + [1] * 1000
    plan = _build_plan(len(lengths))
    planner = ShaftDynamicBatchPlanner(
        plan=plan,
        cost_provider=_build_provider(lengths),
        spec=_build_spec(
            plan,
            optimizer_step_count=1,
            data_world_size=1,
            gradient_accumulation_steps=8,
            max_samples_per_microbatch=250,
            max_padded_tokens=1000,
            target_samples=len(lengths),
            planning_window=len(lengths),
        ),
    )

    (optimizer_batch,) = tuple(planner.iter_optimizer_steps())
    local_batches = [
        batch
        for microstep in optimizer_batch.microsteps
        for batch in microstep.rank_microbatches
    ]

    assert sorted(len(batch.sample_refs) for batch in local_batches) == [
        2,
        2,
        2,
        2,
        250,
        250,
        250,
        250,
    ]
    assert max(batch.padded_llm_tokens for batch in local_batches) <= 1000
    assert Counter(
        ref.context.draw_id for ref in optimizer_batch.sample_refs
    ) == Counter(range(len(lengths)))


def test_dynamic_token_target_consumes_only_a_contiguous_draw_prefix() -> None:
    lengths = [2, 2, 3, 100, 2, 2, 50, 2, 2, 2, 2, 2]
    supervised = [1, 1, 3, 100, 1, 1, 50, 1, 1, 1, 1, 1]
    plan = _build_plan(len(lengths))
    planner = ShaftDynamicBatchPlanner(
        plan=plan,
        cost_provider=_build_provider(lengths, supervised_tokens=supervised),
        spec=_build_spec(
            plan,
            optimizer_step_count=2,
            data_world_size=1,
            gradient_accumulation_steps=2,
            max_samples_per_microbatch=3,
            max_padded_tokens=300,
            target_supervised_tokens=5,
            planning_window=6,
        ),
    )

    optimizer_batches = tuple(planner.iter_optimizer_steps())
    summary = planner.summarize()

    assert planner.count_selected_samples() == 5
    assert summary.optimizer_step_count == 2
    assert summary.selected_sample_count == 5
    assert [batch.draw_start for batch in optimizer_batches] == [0, 3]
    assert [batch.draw_stop for batch in optimizer_batches] == [3, 5]
    assert [
        sorted(ref.context.draw_id for ref in batch.sample_refs)
        for batch in optimizer_batches
    ] == [[0, 1, 2], [3, 4]]
    assert [batch.stats.supervised_tokens for batch in optimizer_batches] == [5, 101]


@pytest.mark.parametrize(
    ("lengths", "exact", "message"),
    [([11, 1], True, "oversize"), ([2, 2], False, "exact sample costs")],
)
def test_dynamic_planner_rejects_unsafe_hard_budget_inputs(
    lengths: list[int],
    exact: bool,
    message: str,
) -> None:
    plan = _build_plan(len(lengths))
    planner = ShaftDynamicBatchPlanner(
        plan=plan,
        cost_provider=_build_provider(lengths, exact=exact),
        spec=_build_spec(
            plan,
            optimizer_step_count=1,
            data_world_size=1,
            gradient_accumulation_steps=2,
            max_samples_per_microbatch=1,
            max_padded_tokens=10,
            target_samples=2,
        ),
    )

    with pytest.raises(ValueError, match=message):
        tuple(planner.iter_optimizer_steps())


def test_dynamic_spec_rejects_infeasible_optimizer_batch_geometry() -> None:
    plan = _build_plan(6)

    with pytest.raises(ValueError, match="at least one sample per local microbatch"):
        _build_spec(
            plan,
            optimizer_step_count=1,
            data_world_size=2,
            gradient_accumulation_steps=2,
            max_samples_per_microbatch=2,
            max_padded_tokens=10,
            target_samples=3,
        )
    with pytest.raises(ValueError, match="exceeds the per-step sample capacity"):
        _build_spec(
            plan,
            optimizer_step_count=1,
            data_world_size=1,
            gradient_accumulation_steps=2,
            max_samples_per_microbatch=2,
            max_padded_tokens=10,
            target_samples=6,
        )


def test_dynamic_batch_sampler_is_shardable_without_fixed_batch_size() -> None:
    lengths = [8, 1, 1, 1, 1, 2, 8, 1, 1, 1, 1, 2]
    plan = _build_plan(len(lengths))
    sampler = ShaftDynamicBatchSampler(
        plan,
        cost_provider=_build_provider(lengths),
        spec=_build_spec(
            plan,
            optimizer_step_count=1,
            data_world_size=2,
            gradient_accumulation_steps=2,
            max_samples_per_microbatch=5,
            max_padded_tokens=10,
            target_samples=12,
        ),
    )
    expected = next(sampler.planner.iter_optimizer_steps())

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

    assert sampler.batch_size is None
    assert sampler.drop_last is True
    assert len(sampler) == 4
    assert sampler.signature.strategy == "dynamic_cost_aware"
    assert sampler.signature.planning_spec_fingerprint == sampler.planner.spec.fingerprint
    assert ShaftBatchPlanningSignature.from_dict(
        sampler.signature.to_dict()
    ) == sampler.signature
    for rank in range(2):
        assert rank_batches[rank] == [
            list(microstep.rank_microbatches[rank].sample_refs)
            for microstep in expected.microsteps
        ]
