from __future__ import annotations

from collections import Counter
import logging
from types import SimpleNamespace

import pytest
from accelerate.data_loader import BatchSamplerShard
from PIL import Image
from torch.utils.data import BatchSampler

from shaft.data import (
    SFTDataset,
    SFTRecord,
    ShaftBatchPlanningSignature,
    ShaftCostAwareSampler,
    ShaftFixedBatchPlanner,
    ShaftFixedBatchPlanningSpec,
    ShaftRowInvariantCostProvider,
    ShaftSampleCost,
    ShaftSamplePlan,
    ShaftSFTSampleCostProvider,
    planning_safe_online_transform,
)
from shaft.loss_scale import ShaftLossScaleSpec
from shaft.model import ShaftProcessorCostEstimate, ShaftProcessorTokenLayout
from shaft.template import ShaftSupervisionCostEstimate, ShaftTemplateSupervisionPlan


pytestmark = pytest.mark.component


def _build_plan(lengths: list[int]) -> ShaftSamplePlan:
    return ShaftSamplePlan(
        {"dataset": len(lengths)},
        {"dataset": 1.0},
        strategy="concat",
        shuffle=False,
        seed=17,
    )


def _build_cost_provider(lengths: list[int]) -> ShaftRowInvariantCostProvider:
    return ShaftRowInvariantCostProvider(
        {
            ("dataset", row_index): ShaftSampleCost(
                llm_tokens=length,
                supervised_tokens=max(length - 1, 0),
                vision_patches=length * 2,
                exact=True,
            )
            for row_index, length in enumerate(lengths)
        },
        fingerprint="test-costs-v1",
    )


def _build_spec(
    plan: ShaftSamplePlan,
    *,
    per_device_batch_size: int,
    data_world_size: int,
    planning_window: int,
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
    drop_last: bool = False,
) -> ShaftFixedBatchPlanningSpec:
    return ShaftFixedBatchPlanningSpec.from_plan(
        plan,
        per_device_batch_size=per_device_batch_size,
        data_world_size=data_world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        planning_window=planning_window,
        seed=seed,
        drop_last=drop_last,
    )


def test_sample_cost_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="llm_tokens"):
        ShaftSampleCost(llm_tokens=0)
    with pytest.raises(ValueError, match="supervised_tokens"):
        ShaftSampleCost(llm_tokens=2, supervised_tokens=-1)
    with pytest.raises(ValueError, match="vision_patches"):
        ShaftSampleCost(llm_tokens=2, vision_patches=-1)
    with pytest.raises(ValueError, match="loss_weight_sum"):
        ShaftSampleCost(llm_tokens=2, loss_weight_sum=float("nan"))


def test_fixed_batch_planner_preserves_refs_and_removes_avoidable_padding() -> None:
    lengths = [8, 1, 8, 1, 2, 1, 2, 1]
    plan = _build_plan(lengths)
    planner = ShaftFixedBatchPlanner(
        plan=plan,
        cost_provider=_build_cost_provider(lengths),
        spec=_build_spec(
            plan,
            per_device_batch_size=2,
            data_world_size=1,
            planning_window=8,
            seed=23,
        ),
    )

    (window_plan,) = tuple(planner.iter_window_plans())
    planned_refs = window_plan.sample_refs
    planned_batches = [
        local_batch
        for microstep in window_plan.microsteps
        for local_batch in microstep.rank_microbatches
    ]

    assert Counter(ref.context.draw_id for ref in planned_refs) == Counter(range(8))
    assert [sorted(cost.llm_tokens for cost in batch.sample_costs) for batch in planned_batches] == [
        [1, 1],
        [1, 1],
        [2, 2],
        [8, 8],
    ]
    assert window_plan.stats.useful_llm_tokens == sum(lengths)
    assert window_plan.stats.planned_padded_llm_tokens == sum(lengths)
    assert window_plan.stats.supervised_tokens == sum(length - 1 for length in lengths)
    assert window_plan.stats.loss_weight_sum is None
    assert window_plan.stats.inexact_sample_count == 0
    assert window_plan.stats.baseline_padded_llm_tokens > sum(lengths)
    assert window_plan.stats.padding_ratio == pytest.approx(0.0)
    assert window_plan.stats.baseline_padding_ratio > 0


def test_fixed_batch_planner_pairs_similar_costs_across_data_ranks() -> None:
    lengths = [8, 1, 7, 2, 8, 1, 7, 2]
    plan = _build_plan(lengths)
    planner = ShaftFixedBatchPlanner(
        plan=plan,
        cost_provider=_build_cost_provider(lengths),
        spec=_build_spec(
            plan,
            per_device_batch_size=1,
            data_world_size=2,
            planning_window=8,
            seed=29,
        ),
    )

    (window_plan,) = tuple(planner.iter_window_plans())

    assert len(window_plan.microsteps) == 4
    for microstep in window_plan.microsteps:
        rank_lengths = [batch.max_llm_tokens for batch in microstep.rank_microbatches]
        assert rank_lengths[0] == rank_lengths[1]
    assert window_plan.stats.max_rank_cost_skew == pytest.approx(0.0)


def test_fixed_batch_planner_never_reorders_across_planning_windows() -> None:
    lengths = [8, 1, 8, 1, 4, 2, 4, 2]
    plan = _build_plan(lengths)
    planner = ShaftFixedBatchPlanner(
        plan=plan,
        cost_provider=_build_cost_provider(lengths),
        spec=_build_spec(
            plan,
            per_device_batch_size=1,
            data_world_size=2,
            planning_window=4,
            seed=31,
        ),
    )

    window_plans = tuple(planner.iter_window_plans())

    assert len(window_plans) == 2
    assert {ref.context.draw_id for ref in window_plans[0].sample_refs} == set(range(4))
    assert {ref.context.draw_id for ref in window_plans[1].sample_refs} == set(range(4, 8))


def test_cost_aware_sampler_is_deterministic_and_advances_plan_cycle() -> None:
    lengths = [8, 1, 8, 1, 2, 1, 2, 1]

    def build_sampler() -> ShaftCostAwareSampler:
        plan = _build_plan(lengths)
        return ShaftCostAwareSampler(
            plan=plan,
            cost_provider=_build_cost_provider(lengths),
            spec=_build_spec(
                plan,
                per_device_batch_size=2,
                data_world_size=1,
                planning_window=8,
                seed=37,
            ),
        )

    uninterrupted = build_sampler()
    epoch_zero = list(uninterrupted)
    uninterrupted.set_epoch(1)
    epoch_one = list(uninterrupted)

    resumed = build_sampler()
    resumed.set_epoch(1)

    assert list(resumed) == epoch_one
    assert [ref.context.draw_id for ref in epoch_zero] != [
        ref.context.draw_id for ref in epoch_one
    ]
    assert {ref.context.plan_cycle for ref in epoch_zero} == {0}
    assert {ref.context.plan_cycle for ref in epoch_one} == {1}
    assert ShaftBatchPlanningSignature.from_dict(
        uninterrupted.signature.to_dict()
    ) == uninterrupted.signature


def test_cost_aware_sampler_logs_multi_window_aggregate(caplog) -> None:
    lengths = [8, 1, 8, 1, 4, 2, 4, 2]
    plan = _build_plan(lengths)
    sampler = ShaftCostAwareSampler(
        plan=plan,
        cost_provider=_build_cost_provider(lengths),
        spec=_build_spec(
            plan,
            per_device_batch_size=1,
            data_world_size=2,
            planning_window=4,
            seed=39,
        ),
    )

    with caplog.at_level(logging.INFO, logger="shaft.data.sampler"):
        assert len(list(sampler)) == len(lengths)

    summaries = [
        record.getMessage()
        for record in caplog.records
        if "[batch-plan-summary]" in record.getMessage()
    ]
    assert len(summaries) == 1
    assert "windows=2" in summaries[0]
    assert "samples=8" in summaries[0]
    assert "planning_seconds=" in summaries[0]


def test_cost_aware_sampler_requires_complete_global_microsteps() -> None:
    lengths = [1, 2, 3, 4, 5]
    plan = _build_plan(lengths)
    provider = _build_cost_provider(lengths)

    with pytest.raises(ValueError, match="global microstep"):
        _build_spec(
            plan,
            per_device_batch_size=2,
            data_world_size=2,
            planning_window=4,
        )

    with pytest.raises(ValueError, match="planning_window must contain"):
        _build_spec(
            plan,
            per_device_batch_size=2,
            data_world_size=2,
            planning_window=3,
            drop_last=True,
        )

    sampler = ShaftCostAwareSampler(
        plan=plan,
        cost_provider=provider,
        spec=_build_spec(
            plan,
            per_device_batch_size=2,
            data_world_size=2,
            planning_window=4,
            drop_last=True,
        ),
    )
    assert len(sampler) == 4
    assert len(list(sampler)) == 4

    with pytest.raises(ValueError, match="at least one complete global microstep"):
        _build_spec(
            _build_plan([1]),
            per_device_batch_size=2,
            data_world_size=2,
            planning_window=4,
            drop_last=True,
        )


def test_accelerate_shards_planned_local_batches_by_data_rank() -> None:
    lengths = [8, 1, 7, 2, 8, 1, 7, 2]
    plan = _build_plan(lengths)
    sampler = ShaftCostAwareSampler(
        plan=plan,
        cost_provider=_build_cost_provider(lengths),
        spec=_build_spec(
            plan,
            per_device_batch_size=1,
            data_world_size=2,
            planning_window=8,
            seed=41,
        ),
    )
    expected_plan = next(sampler.planner.iter_window_plans())

    rank_batches = []
    for rank in range(2):
        batch_sampler = BatchSampler(sampler, batch_size=1, drop_last=False)
        rank_batches.append(
            list(
                BatchSamplerShard(
                    batch_sampler,
                    num_processes=2,
                    process_index=rank,
                    split_batches=False,
                )
            )
        )

    for rank in range(2):
        assert rank_batches[rank] == [
            list(microstep.rank_microbatches[rank].sample_refs)
            for microstep in expected_plan.microsteps
        ]
    assert all(
        set(rank_batches[0][step]).isdisjoint(rank_batches[1][step])
        for step in range(len(expected_plan.microsteps))
    )


class _CostTokenizer:
    eos_token_id = 2
    name_or_path = "cost-tokenizer"
    shaft_cost_fingerprint = "test-cost-tokenizer-v1"

    def __call__(self, texts, add_special_tokens=False, return_attention_mask=False):
        _ = add_special_tokens, return_attention_mask
        rows = []
        for text in texts:
            rows.append([20, 21] if text == "target" else [10, 99, 11, 12])
        return {"input_ids": rows}


class _SerializedTokenizerBackend:
    def __init__(self, payload: str):
        self.payload = payload

    def to_str(self) -> str:
        return self.payload


class _CostTokenizerWithBackend(_CostTokenizer):
    def __init__(self, payload: str):
        self.backend_tokenizer = _SerializedTokenizerBackend(payload)


class _UndeclaredCostTokenizer(_CostTokenizer):
    shaft_cost_fingerprint = None


class _CostProcessor:
    image_token_id = 99

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = messages, tokenize, add_generation_prompt
        return "prompt"


class _CostModelAdapter:
    model_type = "test_vlm"
    model_name_or_path = "test-model"
    template_type = "test-template"
    processor_policy = SimpleNamespace(supports_exact_image_cost=True)

    def __init__(self, cost_semantics_version: str = "v1") -> None:
        self.cost_semantics_version = str(cost_semantics_version)

    def processor_cost_semantics_signature(self, **kwargs):
        _ = kwargs
        return ("test-processor-cost-semantics", self.cost_semantics_version)

    def estimate_processor_image_cost(self, **kwargs):
        assert kwargs["image_sizes"] == ((64, 64),)
        return ShaftProcessorCostEstimate(
            processed_image_tokens=4,
            vision_patches=16,
            exact=True,
        )

    def estimate_processor_token_layout(self, **kwargs):
        assert kwargs["rendered_token_ids"] == (10, 99, 11, 12)
        assert len(kwargs["image_costs"]) == 1
        return ShaftProcessorTokenLayout(processed_boundaries=(0, 1, 5, 6, 7))


class _CostTemplate:
    def build_supervision_plan(self, **kwargs):
        item = kwargs["item"]
        assert item["user_prompt"] == "draw-0"
        return ShaftTemplateSupervisionPlan(
            prompt_text="prompt",
            target_text=kwargs["target_text"],
            loss_spec=ShaftLossScaleSpec(
                base_strategy="default",
                prefix_scale=0.5,
                target_scale=2.0,
            ),
            rendered_prefix_token_ids=(10, 99, 11, 12),
            trainable_prefix_spans=((1, 3),),
        )

    def estimate_supervision_cost(self, **kwargs):
        assert kwargs["prefix_token_layout"].processed_boundaries == (0, 1, 5, 6, 7)
        return ShaftSupervisionCostEstimate(
            llm_tokens=10,
            supervised_tokens=8,
            loss_weight_sum=8.5,
        )


def test_sft_cost_provider_matches_processed_and_shifted_loss_contract(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(image_path)
    plan = _build_plan([1])

    @planning_safe_online_transform(fingerprint="test-draw-prompt-v1")
    def apply_draw_prompt(item):
        resolved = dict(item)
        resolved["user_prompt"] = f"draw-{item['_sample_context']['draw_id']}"
        return resolved

    dataset = SFTDataset(
        {
            "dataset": [
                SFTRecord(
                    image_path=str(image_path),
                    target_text="target",
                    dataset_name="dataset",
                )
            ]
        },
        sample_plan=plan,
        online_transforms=[apply_draw_prompt],
    )
    provider = ShaftSFTSampleCostProvider(
        dataset=dataset,
        model_adapter=_CostModelAdapter(),
        template=_CostTemplate(),
        processor=_CostProcessor(),
        tokenizer=_CostTokenizer(),
        min_pixels=None,
        max_pixels=None,
        max_length=None,
        add_eos_token=True,
        loss_scale_name="default",
        image_size_cache_size=1,
    )

    cost = provider(plan.ref_at(0))

    # Four rendered prompt tokens become seven after the image placeholder expands
    # to four tokens. Target contributes two tokens plus EOS.
    assert cost.llm_tokens == 10
    assert cost.vision_patches == 16
    assert cost.supervised_tokens == 8
    assert cost.loss_weight_sum == pytest.approx(8.5)
    assert cost.exact is True
    assert provider.fingerprint


def test_planning_safe_transform_requires_explicit_stable_fingerprint() -> None:
    with pytest.raises(ValueError, match="explicit stable fingerprint"):

        @planning_safe_online_transform
        def implicit_fingerprint(item):
            return item


def test_sft_planning_item_does_not_decode_image(tmp_path) -> None:
    missing_image = tmp_path / "missing.png"
    plan = _build_plan([1])
    dataset = SFTDataset(
        {
            "dataset": [
                SFTRecord(
                    image_path=str(missing_image),
                    target_text="target",
                    dataset_name="dataset",
                )
            ]
        },
        sample_plan=plan,
        online_transforms=[lambda item: {**item, "user_prompt": "planned"}],
    )

    item = dataset.get_planning_item(plan.ref_at(0))

    assert item["image"] is None
    assert item["image_path"] == str(missing_image)
    assert item["user_prompt"] == "planned"


def test_sft_cost_provider_rejects_undeclared_online_transform(tmp_path) -> None:
    plan = _build_plan([1])
    dataset = SFTDataset(
        {
            "dataset": [
                SFTRecord(
                    image_path=str(tmp_path / "image.png"),
                    target_text="target",
                    dataset_name="dataset",
                )
            ]
        },
        sample_plan=plan,
        online_transforms=[lambda item: item],
    )

    with pytest.raises(ValueError, match="image-identity/geometry"):
        ShaftSFTSampleCostProvider(
            dataset=dataset,
            model_adapter=_CostModelAdapter(),
            template=_CostTemplate(),
            processor=_CostProcessor(),
            tokenizer=_CostTokenizer(),
            min_pixels=None,
            max_pixels=None,
            max_length=None,
            add_eos_token=True,
            loss_scale_name="default",
        )


def test_sft_cost_manifest_scans_only_unique_images_in_sample_horizon(
    tmp_path,
    monkeypatch,
) -> None:
    used_image = tmp_path / "used.png"
    unused_image = tmp_path / "unused.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(used_image)
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(unused_image)
    plan = ShaftSamplePlan(
        {"used": 4},
        {"used": 1.0},
        strategy="concat",
        shuffle=False,
        seed=17,
    )
    dataset = SFTDataset(
        {
            "used": [
                SFTRecord(
                    image_path=str(used_image),
                    target_text=f"target-{index}",
                    dataset_name="used",
                )
                for index in range(4)
            ],
            "unused": [
                SFTRecord(
                    image_path=str(unused_image),
                    target_text="unused-target",
                    dataset_name="unused",
                ),
            ]
        },
        sample_plan=plan,
    )
    opened_paths: list[str] = []
    original_identity = __import__(
        "shaft.data.cost", fromlist=["_image_asset_identity"]
    )._image_asset_identity

    def counting_identity(image_path):
        opened_paths.append(str(image_path))
        return original_identity(image_path)

    monkeypatch.setattr("shaft.data.cost._image_asset_identity", counting_identity)
    provider = ShaftSFTSampleCostProvider(
        dataset=dataset,
        model_adapter=_CostModelAdapter(),
        template=_CostTemplate(),
        processor=_CostProcessor(),
        tokenizer=_CostTokenizer(),
        min_pixels=None,
        max_pixels=None,
        max_length=None,
        add_eos_token=True,
        loss_scale_name="default",
        image_size_cache_size=0,
    )

    # Four referenced rows share one image, while another source row is outside
    # this training horizon. Runtime lookups reuse the manifest even when the
    # bounded LRU is disabled.
    assert opened_paths == [str(used_image.resolve())]
    assert provider._get_image_size(str(used_image)) == (64, 64)
    assert provider._get_image_size(str(used_image)) == (64, 64)
    assert opened_paths == [str(used_image.resolve())]


def test_sft_cost_fingerprint_binds_source_record_content(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(image_path)
    plan = _build_plan([1])

    def build_provider(target_text: str) -> ShaftSFTSampleCostProvider:
        dataset = SFTDataset(
            {
                "dataset": [
                    SFTRecord(
                        image_path=str(image_path),
                        target_text=target_text,
                        dataset_name="dataset",
                    )
                ]
            },
            sample_plan=plan,
        )
        return ShaftSFTSampleCostProvider(
            dataset=dataset,
            model_adapter=_CostModelAdapter(),
            template=_CostTemplate(),
            processor=_CostProcessor(),
            tokenizer=_CostTokenizer(),
            min_pixels=None,
            max_pixels=None,
            max_length=None,
            add_eos_token=True,
            loss_scale_name="default",
        )

    assert build_provider("same-len-a").fingerprint != build_provider(
        "same-len-b"
    ).fingerprint


def test_sft_cost_fingerprint_binds_serialized_tokenizer_backend(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(image_path)
    plan = _build_plan([1])
    dataset = SFTDataset(
        {
            "dataset": [
                SFTRecord(
                    image_path=str(image_path),
                    target_text="target",
                    dataset_name="dataset",
                )
            ]
        },
        sample_plan=plan,
    )

    def build_provider(backend_payload: str) -> ShaftSFTSampleCostProvider:
        return ShaftSFTSampleCostProvider(
            dataset=dataset,
            model_adapter=_CostModelAdapter(),
            template=_CostTemplate(),
            processor=_CostProcessor(),
            tokenizer=_CostTokenizerWithBackend(backend_payload),
            min_pixels=None,
            max_pixels=None,
            max_length=None,
            add_eos_token=True,
            loss_scale_name="default",
        )

    assert build_provider("vocab-a").fingerprint != build_provider(
        "vocab-b"
    ).fingerprint


def test_sft_cost_fingerprint_binds_model_policy_cost_semantics(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(image_path)
    plan = _build_plan([1])
    dataset = SFTDataset(
        {
            "dataset": [
                SFTRecord(
                    image_path=str(image_path),
                    target_text="target",
                    dataset_name="dataset",
                )
            ]
        },
        sample_plan=plan,
    )

    def build_provider(cost_semantics_version: str) -> ShaftSFTSampleCostProvider:
        return ShaftSFTSampleCostProvider(
            dataset=dataset,
            model_adapter=_CostModelAdapter(cost_semantics_version),
            template=_CostTemplate(),
            processor=_CostProcessor(),
            tokenizer=_CostTokenizer(),
            min_pixels=None,
            max_pixels=None,
            max_length=None,
            add_eos_token=True,
            loss_scale_name="default",
        )

    assert build_provider("v1").fingerprint != build_provider("v2").fingerprint


def test_sft_cost_provider_rejects_unfingerprinted_slow_tokenizer(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(image_path)
    plan = _build_plan([1])
    dataset = SFTDataset(
        {
            "dataset": [
                SFTRecord(
                    image_path=str(image_path),
                    target_text="target",
                    dataset_name="dataset",
                )
            ]
        },
        sample_plan=plan,
    )

    with pytest.raises(ValueError, match="including merges/unigram state"):
        ShaftSFTSampleCostProvider(
            dataset=dataset,
            model_adapter=_CostModelAdapter(),
            template=_CostTemplate(),
            processor=_CostProcessor(),
            tokenizer=_UndeclaredCostTokenizer(),
            min_pixels=None,
            max_pixels=None,
            max_length=None,
            add_eos_token=True,
            loss_scale_name="default",
        )


def test_sft_cost_fingerprint_binds_image_asset_dimensions(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(image_path)
    plan = _build_plan([1])

    def build_provider() -> ShaftSFTSampleCostProvider:
        dataset = SFTDataset(
            {
                "dataset": [
                    SFTRecord(
                        image_path=str(image_path),
                        target_text="target",
                        dataset_name="dataset",
                    )
                ]
            },
            sample_plan=plan,
        )
        return ShaftSFTSampleCostProvider(
            dataset=dataset,
            model_adapter=_CostModelAdapter(),
            template=_CostTemplate(),
            processor=_CostProcessor(),
            tokenizer=_CostTokenizer(),
            min_pixels=None,
            max_pixels=None,
            max_length=None,
            add_eos_token=True,
            loss_scale_name="default",
        )

    original_fingerprint = build_provider().fingerprint
    Image.new("RGB", (96, 64), color=(0, 0, 0)).save(image_path)

    assert build_provider().fingerprint != original_fingerprint
