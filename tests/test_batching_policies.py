from __future__ import annotations

import hashlib
import json
from enum import IntEnum
from pathlib import Path

from accelerate.data_loader import BatchSamplerShard
import pytest
from PIL import Image

from shaft.data import (
    ShaftBatchPlanner,
    ShaftBatchPlanningSpec,
    ShaftBatchPlanningState,
    ShaftBatchContext,
    ShaftGreedySequencePacker,
    ShaftLengthBatchGrouping,
    ShaftLocalMicroBatchPlan,
    ShaftLogicalSegmentPlan,
    ShaftPhysicalPackPlan,
    ShaftPlannedBatchSampler,
    ShaftPlannedSampleRef,
    ShaftSampleContext,
    ShaftSampleCost,
    ShaftSampleRef,
    ShaftSampleSchedule,
    ShaftRowInvariantCostProvider,
    ShaftVarlenLayoutPlan,
    ShaftVarlenSegmentLayout,
    SFTDataset,
    SFTRecord,
)


class _JsonIntImpostor(IntEnum):
    ONE = 1


def _segment(draw_id: int, length: int, *, vision_patches: int = 0) -> ShaftLogicalSegmentPlan:
    return ShaftLogicalSegmentPlan(
        sample_ref=ShaftSampleRef(
            dataset_name="train",
            row_index=draw_id,
            context=ShaftSampleContext(
                draw_id=draw_id,
                plan_cycle=0,
                transform_seed=1000 + draw_id,
            ),
        ),
        cost=ShaftSampleCost(
            llm_tokens=length,
            supervised_tokens=max(length - 1, 0),
            vision_patches=vision_patches,
            exact=True,
        ),
    )


def test_plan_hierarchy_has_one_structural_truth_and_derived_flat_views() -> None:
    first = _segment(0, 8, vision_patches=16)
    second = _segment(1, 2, vision_patches=8)
    third = _segment(2, 1, vision_patches=4)
    local = ShaftLocalMicroBatchPlan(
        packs=(
            ShaftPhysicalPackPlan(segments=(first,)),
            ShaftPhysicalPackPlan(segments=(second, third)),
        )
    )

    assert local.physical_pack_count == 2
    assert local.logical_segment_count == 3
    assert local.sample_refs == (
        first.sample_ref,
        second.sample_ref,
        third.sample_ref,
    )
    assert local.sample_costs == (first.cost, second.cost, third.cost)
    assert local.useful_llm_tokens == 11
    assert local.supervised_tokens == 8
    assert local.vision_patches == 28
    assert local.padded_llm_tokens == 16


def test_logical_segment_json_roundtrip_preserves_identity_and_cost() -> None:
    segment = _segment(3, 8, vision_patches=16)

    payload = json.loads(json.dumps(segment.to_dict()))

    assert ShaftLogicalSegmentPlan.from_dict(payload) == segment


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    (
        (("dataset_name",), Path("train")),
        (("row_index",), True),
        (("context", "draw_id"), "3"),
        (("context", "plan_cycle"), 0.0),
        (("context", "transform_seed"), _JsonIntImpostor.ONE),
        (("cost", "llm_tokens"), False),
        (("cost", "supervised_tokens"), "7"),
        (("cost", "vision_patches"), 16.0),
        (("cost", "loss_weight_sum"), float("-inf")),
        (("cost", "exact"), 1),
    ),
)
def test_logical_segment_rejects_noncanonical_json_values(
    path: tuple[str, ...],
    invalid_value: object,
) -> None:
    payload = _segment(3, 8, vision_patches=16).to_dict()
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        ShaftLogicalSegmentPlan.from_dict(payload)


@pytest.mark.parametrize("owner", ("segment", "context", "cost"))
def test_logical_segment_rejects_unknown_versioned_schema_fields(owner: str) -> None:
    payload = _segment(3, 8, vision_patches=16).to_dict()
    target = payload if owner == "segment" else payload[owner]
    target["unknown"] = 1

    with pytest.raises(ValueError, match="unknown"):
        ShaftLogicalSegmentPlan.from_dict(payload)


def test_varlen_layout_rejects_duplicate_processor_rows() -> None:
    with pytest.raises(ValueError, match="processor rows"):
        ShaftVarlenLayoutPlan(
            global_microstep=0,
            plan_fingerprint="duplicate-row",
            local_batch_id=0,
            pack_lengths=(4,),
            segments=(
                ShaftVarlenSegmentLayout(0, 0, 0, 0, 2),
                ShaftVarlenSegmentLayout(0, 0, 1, 2, 4),
            ),
        )


def test_varlen_layout_rejects_pack_local_length_mismatch() -> None:
    with pytest.raises(ValueError, match="pack lengths"):
        ShaftVarlenLayoutPlan(
            global_microstep=0,
            plan_fingerprint="pack-length-mismatch",
            local_batch_id=0,
            pack_lengths=(4, 3),
            segments=(
                ShaftVarlenSegmentLayout(0, 0, 0, 0, 4),
                ShaftVarlenSegmentLayout(1, 0, 1, 4, 7),
            ),
        )


def test_length_grouping_orders_candidates_without_consuming_or_mutating_fifo() -> None:
    fifo = tuple(
        _segment(draw_id, length)
        for draw_id, length in enumerate((3, 8, 2, 8, 1, 5))
    )

    ordered = ShaftLengthBatchGrouping.build(fifo, seed=17, global_microstep=4)

    assert tuple(segment.cost.llm_tokens for segment in ordered) == (8, 8, 5, 3, 2, 1)
    assert tuple(segment.sample_ref.context.draw_id for segment in ordered[:2]) == (1, 3)
    assert tuple(segment.sample_ref.context.draw_id for segment in fifo) == tuple(range(6))
    assert sorted(segment.sample_ref.context.draw_id for segment in ordered) == list(range(6))


def test_greedy_whole_sample_packing_matches_golden_case() -> None:
    segments = tuple(
        _segment(draw_id, length)
        for draw_id, length in enumerate((8, 1, 1, 1, 1, 2))
    )
    ordered = ShaftLengthBatchGrouping.build(segments, seed=0, global_microstep=0)

    packs = ShaftGreedySequencePacker.build(
        ordered,
        physical_pack_count=2,
        max_length=8,
        required_draw_id=0,
    )

    assert [[segment.cost.llm_tokens for segment in pack.segments] for pack in packs] == [
        [8],
        [2, 1, 1, 1, 1],
    ]
    assert sorted(
        segment.sample_ref.context.draw_id
        for pack in packs
        for segment in pack.segments
    ) == list(range(6))
    assert all(pack.useful_llm_tokens <= 8 for pack in packs)


def test_greedy_packing_keeps_unselected_candidates_outside_packs() -> None:
    segments = tuple(_segment(draw_id, 6) for draw_id in range(3))
    packs = ShaftGreedySequencePacker.build(
        segments,
        physical_pack_count=2,
        max_length=10,
        required_draw_id=0,
    )

    assert [[segment.sample_ref.context.draw_id for segment in pack.segments] for pack in packs] == [
        [0],
        [1],
    ]


def test_greedy_packing_rejects_true_oversize_without_split_or_drop() -> None:
    with pytest.raises(ValueError, match=r"draw_id=0.*llm_tokens=9.*max_length=8"):
        ShaftGreedySequencePacker.build(
            (_segment(0, 9), _segment(1, 1)),
            physical_pack_count=2,
            max_length=8,
            required_draw_id=0,
        )


def test_planned_sample_ref_requires_complete_batch_identity() -> None:
    sample_ref = _segment(3, 4).sample_ref
    context = ShaftBatchContext(
        global_microstep=7,
        plan_fingerprint="plan-abc",
        local_batch_id=1,
        pack_index=0,
        segment_index=1,
        pack_segment_count=2,
    )
    planned = ShaftPlannedSampleRef(sample_ref=sample_ref, batch_context=context)

    assert planned.sample_ref is sample_ref
    assert planned.batch_context.to_dict() == {
        "global_microstep": 7,
        "plan_fingerprint": "plan-abc",
        "local_batch_id": 1,
        "pack_index": 0,
        "segment_index": 1,
        "pack_segment_count": 2,
    }
    assert ShaftBatchContext.from_dict(
        json.loads(json.dumps(planned.batch_context.to_dict()))
    ) == planned.batch_context

    with pytest.raises(ValueError, match="segment_index"):
        ShaftBatchContext(
            global_microstep=7,
            plan_fingerprint="plan-abc",
            local_batch_id=1,
            pack_index=0,
            segment_index=2,
            pack_segment_count=2,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("global_microstep", True),
        ("plan_fingerprint", Path("plan-abc")),
        ("local_batch_id", "1"),
        ("pack_index", 0.0),
        ("segment_index", _JsonIntImpostor.ONE),
        ("pack_segment_count", False),
    ),
)
def test_batch_context_rejects_noncanonical_json_values(
    field_name: str,
    invalid_value: object,
) -> None:
    payload = ShaftBatchContext(
        global_microstep=7,
        plan_fingerprint="plan-abc",
        local_batch_id=1,
        pack_index=0,
        segment_index=1,
        pack_segment_count=2,
    ).to_dict()
    payload[field_name] = invalid_value

    with pytest.raises(TypeError):
        ShaftBatchContext.from_dict(payload)


def test_batch_context_rejects_unknown_and_non_string_keys() -> None:
    payload = ShaftBatchContext(
        global_microstep=7,
        plan_fingerprint="plan-abc",
        local_batch_id=1,
        pack_index=0,
        segment_index=1,
        pack_segment_count=2,
    ).to_dict()
    payload["unknown"] = 1
    with pytest.raises(ValueError, match="unknown"):
        ShaftBatchContext.from_dict(payload)

    payload.pop("unknown")
    payload[1] = "non-string-key"
    with pytest.raises(TypeError, match="keys"):
        ShaftBatchContext.from_dict(payload)


def _planning_spec(
    *,
    world_size: int = 1,
    per_device_packs: int = 2,
    packing: str = "none",
    resource_budgets: tuple[tuple[str, int], ...] = (),
) -> ShaftBatchPlanningSpec:
    return ShaftBatchPlanningSpec(
        grouping="length",
        cardinality="fixed",
        packing=packing,
        layout="varlen" if packing == "greedy" else "padded",
        data_world_size=world_size,
        buffer_size=6,
        per_device_microbatch_size=per_device_packs,
        max_sequence_length=8,
        max_tokens_per_microbatch=per_device_packs * 8,
        resource_budgets=resource_budgets,
        seed=13,
        sample_schedule_fingerprint="schedule",
        cost_fingerprint="costs",
    )


def _length_planner(
    lengths: tuple[int, ...],
    *,
    spec: ShaftBatchPlanningSpec,
    vision_patches: tuple[int, ...] | None = None,
    state: ShaftBatchPlanningState | None = None,
) -> ShaftBatchPlanner:
    schedule = ShaftSampleSchedule(
        {"train": len(lengths)},
        {"train": 1.0},
        strategy="concat",
        shuffle=False,
        seed=0,
    )
    costs = {
        ("train", index): ShaftSampleCost(
            llm_tokens=length,
            supervised_tokens=max(length - 1, 0),
            vision_patches=(0 if vision_patches is None else vision_patches[index]),
            exact=True,
        )
        for index, length in enumerate(lengths)
    }
    provider = ShaftRowInvariantCostProvider(costs, fingerprint="costs")
    spec = ShaftBatchPlanningSpec(
        **{
            **spec.to_init_dict(),
            "sample_schedule_fingerprint": schedule.fingerprint,
        }
    )
    return ShaftBatchPlanner(
        schedule=schedule,
        cost_provider=provider,
        spec=spec,
        state=state,
    )


def test_length_planner_tracks_physical_packs_separately_from_logical_segments() -> None:
    spec = _planning_spec(packing="greedy")
    planner = _length_planner((8, 1, 1, 1, 1, 2), spec=spec)

    plan = planner.next_global_microbatch()

    assert plan.stats.physical_pack_count == 2
    assert plan.stats.logical_segment_count == 6
    assert plan.state_after.emitted_physical_packs == 2
    assert plan.state_after.emitted_logical_segments == 6
    assert plan.state_after.next_draw_id == 6
    assert not plan.state_after.buffer
    assert [
        [segment.cost.llm_tokens for segment in pack.segments]
        for pack in plan.rank_microbatches[0].packs
    ] == [[8], [2, 1, 1, 1, 1]]


def test_length_planner_resume_replays_the_same_future_pack_mapping() -> None:
    spec = _planning_spec(packing="greedy")
    uninterrupted = _length_planner((8, 1, 1, 1, 1, 2), spec=spec)
    first = uninterrupted.next_global_microbatch()
    expected = uninterrupted.next_global_microbatch()

    resumed = _length_planner(
        (8, 1, 1, 1, 1, 2),
        spec=spec,
        state=ShaftBatchPlanningState.from_dict(first.state_after.to_dict()),
    )
    actual = resumed.next_global_microbatch()

    assert actual.fingerprint == expected.fingerprint


@pytest.mark.parametrize(
    "missing_field",
    (
        "grouping",
        "packing",
        "layout",
        "max_sequence_length",
        "resource_budgets",
        "fingerprint",
    ),
)
def test_v4_planning_spec_rejects_missing_canonical_fields(
    missing_field: str,
) -> None:
    payload = _planning_spec(packing="none").to_dict()
    payload.pop(missing_field)

    with pytest.raises(ValueError, match=rf"missing required fields.*{missing_field}"):
        ShaftBatchPlanningSpec.from_dict(payload)


@pytest.mark.parametrize(
    "missing_field",
    (
        "global_microstep",
        "next_draw_id",
        "buffer",
        "emitted_samples",
        "emitted_physical_packs",
        "emitted_llm_tokens",
        "emitted_supervised_tokens",
        "emitted_vision_patches",
    ),
)
def test_v4_planning_state_rejects_missing_fields_after_refingerprint(
    missing_field: str,
) -> None:
    payload = ShaftBatchPlanningState(contract_fingerprint="contract").to_dict()
    payload.pop(missing_field)
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

    with pytest.raises(ValueError, match=rf"missing required fields.*{missing_field}"):
        ShaftBatchPlanningState.from_dict(payload)


def test_length_greedy_enforces_local_aggregate_vision_guard() -> None:
    spec = _planning_spec(
        packing="greedy",
        resource_budgets=(("vision_patches", 12),),
    )
    planner = _length_planner(
        (4, 4, 2, 2, 1, 1),
        spec=spec,
        vision_patches=(8, 8, 2, 2, 1, 1),
    )

    plan = planner.next_global_microbatch()

    local = plan.rank_microbatches[0]
    assert local.vision_patches <= 12
    assert plan.state_after.emitted_logical_segments < 6
    assert plan.state_after.buffer


def test_length_greedy_world_two_batch_two_shards_complete_physical_packs() -> None:
    planner = _length_planner(
        (8, 7, 6, 2, 1, 1),
        spec=_planning_spec(
            world_size=2,
            per_device_packs=2,
            packing="greedy",
        ),
    )
    plan = planner.next_global_microbatch()

    assert len(plan.rank_microbatches) == 2
    assert [batch.physical_pack_count for batch in plan.rank_microbatches] == [2, 2]
    assert all(batch.useful_llm_tokens <= 16 for batch in plan.rank_microbatches)
    assert plan.stats.physical_pack_count == 4
    assert plan.stats.logical_segment_count == 6
    assert {
        segment.draw_id
        for batch in plan.rank_microbatches
        for pack in batch.packs
        for segment in pack.segments
    } == set(range(6))

    sampler = ShaftPlannedBatchSampler(
        planner.schedule,
        cost_provider=planner.cost_provider,
        spec=planner.spec,
        global_microstep_count=2,
        planning_frame_size=1,
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

    assert [len(batches) for batches in rank_batches] == [2, 2]
    for global_microstep in range(2):
        draw_sets = []
        for rank in range(2):
            refs = rank_batches[rank][global_microstep]
            contexts = [ref.batch_context for ref in refs]
            assert {context.local_batch_id for context in contexts} == {rank}
            assert {context.pack_index for context in contexts} == {0, 1}
            assert {context.global_microstep for context in contexts} == {
                global_microstep
            }
            draw_sets.append({ref.sample_ref.context.draw_id for ref in refs})
        assert draw_sets[0].isdisjoint(draw_sets[1])


def test_microbatch_plan_flattens_typed_refs_in_pack_segment_order() -> None:
    spec = _planning_spec(packing="greedy")
    planner = _length_planner((8, 1, 1, 1, 1, 2), spec=spec)
    plan = planner.next_global_microbatch()

    planned_refs = plan.planned_refs_for_rank(0)

    assert [ref.sample_ref.context.draw_id for ref in planned_refs] == [0, 5, 1, 2, 3, 4]
    assert [ref.batch_context.pack_index for ref in planned_refs] == [0, 1, 1, 1, 1, 1]
    assert [ref.batch_context.segment_index for ref in planned_refs] == [0, 0, 1, 2, 3, 4]
    assert {ref.batch_context.pack_segment_count for ref in planned_refs[1:]} == {5}
    assert {ref.batch_context.global_microstep for ref in planned_refs} == {0}
    assert {ref.batch_context.local_batch_id for ref in planned_refs} == {0}
    assert {ref.batch_context.plan_fingerprint for ref in planned_refs} == {
        plan.fingerprint
    }


def test_dataset_attaches_batch_context_after_online_transforms(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(image_path)
    observed_private_context: list[bool] = []

    def transform(sample):
        observed_private_context.append("_batch_context" in sample)
        return {**sample, "target_text": "rotated"}

    dataset = SFTDataset(
        {
            "train": [
                SFTRecord(
                    image_path=str(image_path),
                    target_text="original",
                    dataset_name="train",
                )
            ]
        },
        online_transforms=[transform],
    )
    planned = ShaftPlannedSampleRef(
        sample_ref=_segment(0, 3).sample_ref,
        batch_context=ShaftBatchContext(
            global_microstep=2,
            plan_fingerprint="plan",
            local_batch_id=0,
            pack_index=1,
            segment_index=0,
            pack_segment_count=1,
        ),
    )

    item = dataset[planned]

    assert observed_private_context == [False]
    assert item["target_text"] == "rotated"
    assert item["_batch_context"] == planned.batch_context.to_dict()
