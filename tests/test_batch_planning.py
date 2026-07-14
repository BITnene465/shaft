from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import warnings

from PIL import Image
import pytest

from shaft.data import (
    SFTDataset,
    SFTRecord,
    ShaftSFTSampleCostProvider,
    ShaftSampleCost,
    ShaftSamplePlan,
    ShaftVarlenBatchLayout,
    sft_runtime_source_identity,
)
from shaft.data.transforms import planning_safe_online_transform
from shaft.model.types import ShaftProcessorCostEstimate, ShaftProcessorTokenLayout
from shaft.template.types import (
    ShaftLossScaleSpec,
    ShaftSupervisionCostEstimate,
    ShaftTemplateSupervisionPlan,
)


pytestmark = pytest.mark.component


def test_runtime_source_identity_does_not_block_unversioned_fixed_transform() -> None:
    def unversioned(sample):
        return sample

    identity = sft_runtime_source_identity(
        SimpleNamespace(
            records=[{"id": 1}],
            online_transforms=[unversioned],
            media_snapshot_id="",
        )
    )

    assert identity.fingerprint
    assert identity.complete is False
    assert identity.incomplete_reasons[0].startswith("unversioned_transform:")
    assert identity.incomplete_reasons[1] == "missing_media_snapshot_id"


def test_varlen_batch_layout_rejects_an_empty_physical_batch() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        ShaftVarlenBatchLayout.build(
            contexts=(),
            input_ids=(),
            labels=(),
            mm_token_type_ids=(),
            loss_scales=(),
            ignore_index=-100,
            max_sequence_length=16,
        )


class _CostTokenizer:
    eos_token_id = 2
    name_or_path = "cost-tokenizer"
    shaft_cost_fingerprint = "test-cost-tokenizer-v1"

    def __call__(self, texts, add_special_tokens=False, return_attention_mask=False):
        _ = add_special_tokens, return_attention_mask
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [
                [20, 21] if text == "target" else [10, 99, 11, 12]
                for text in texts
            ]
        }


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

    def __init__(self) -> None:
        self.image_cost_calls = 0

    def processor_cost_semantics_signature(self, **kwargs):
        _ = kwargs
        return ("test-processor-cost-semantics-v1",)

    def estimate_processor_image_cost(self, **kwargs):
        self.image_cost_calls += 1
        assert kwargs["image_sizes"] == ((64, 64),)
        return ShaftProcessorCostEstimate(
            processed_image_tokens=4,
            vision_patches=16,
            exact=True,
        )

    def estimate_processor_token_layout(self, **kwargs):
        assert kwargs["rendered_token_ids"] == (10, 99, 11, 12)
        return ShaftProcessorTokenLayout(processed_boundaries=(0, 1, 5, 6, 7))


class _CostTemplate:
    def build_supervision_plan(self, **kwargs):
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


def _build_provider(
    tmp_path: Path,
    *,
    missing_second_image: bool = False,
    cache_size: int = 8,
    tokenizer=None,
):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(image_path)
    second_path = tmp_path / "missing.png" if missing_second_image else image_path
    plan = ShaftSamplePlan(
        {"dataset": 2},
        {"dataset": 1.0},
        strategy="concat",
        num_samples=2,
        shuffle=False,
        seed=17,
    )

    @planning_safe_online_transform(fingerprint="test-cost-transform-v1")
    def apply_draw_prompt(item):
        updated = dict(item)
        updated["user_prompt"] = f"draw-{item['_sample_context']['draw_id'] % 1}"
        return updated

    dataset = SFTDataset(
        {
            "dataset": [
                SFTRecord(
                    image_path=str(image_path),
                    target_text="target",
                    dataset_name="dataset",
                ),
                SFTRecord(
                    image_path=str(second_path),
                    target_text="target",
                    dataset_name="dataset",
                ),
            ]
        },
        sample_plan=plan,
        media_snapshot_id="test-media-v1",
        online_transforms=[apply_draw_prompt],
    )
    adapter = _CostModelAdapter()
    provider = ShaftSFTSampleCostProvider(
        dataset=dataset,
        model_adapter=adapter,
        template=_CostTemplate(),
        processor=_CostProcessor(),
        tokenizer=tokenizer or _CostTokenizer(),
        min_pixels=None,
        max_pixels=None,
        max_length=None,
        add_eos_token=True,
        loss_scale_name="default",
        cache_size=cache_size,
    )
    return plan, provider, adapter


@pytest.mark.parametrize(
    "kwargs",
    [
        {"llm_tokens": 0},
        {"llm_tokens": 2, "supervised_tokens": -1},
        {"llm_tokens": 2, "supervised_tokens": 3},
        {"llm_tokens": 2, "vision_patches": -1},
        {"llm_tokens": 2, "loss_weight_sum": float("nan")},
    ],
)
def test_sample_cost_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        ShaftSampleCost(**kwargs)


def test_sft_cost_provider_matches_supervision_and_processor_cost(tmp_path: Path) -> None:
    plan, provider, _ = _build_provider(tmp_path)

    cost = provider(plan.ref_at(0))

    assert cost == ShaftSampleCost(
        llm_tokens=10,
        supervised_tokens=8,
        vision_patches=16,
        loss_weight_sum=8.5,
        exact=True,
    )
    assert provider.fingerprint


def test_provider_construction_does_not_scan_future_image_headers(tmp_path: Path) -> None:
    plan, provider, _ = _build_provider(tmp_path, missing_second_image=True)

    assert provider(plan.ref_at(0)).exact is True
    with pytest.raises(FileNotFoundError):
        provider(plan.ref_at(1))


def test_provider_bounded_cache_reuses_equivalent_prompt_variant(tmp_path: Path) -> None:
    plan, provider, adapter = _build_provider(tmp_path, cache_size=2)

    first = provider(plan.ref_at(0))
    second = provider(plan.ref_at(0))

    assert second == first
    assert adapter.image_cost_calls == 1
    assert provider.cache_entry_counts == (1, 1)


@pytest.mark.parametrize(
    ("rank_zero", "expected_warning_count"),
    [(True, 1), (False, 0)],
)
def test_large_image_header_warning_is_emitted_only_by_rank_zero(
    tmp_path: Path,
    monkeypatch,
    caplog,
    rank_zero: bool,
    expected_warning_count: int,
) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 3_000)
    monkeypatch.setattr("shaft.data.cost.is_rank_zero", lambda: rank_zero)
    plan, provider, _ = _build_provider(tmp_path)

    with caplog.at_level("WARNING", logger="shaft.data.cost"):
        provider(plan.ref_at(0))

    messages = [
        record.getMessage()
        for record in caplog.records
        if "large image header observed" in record.getMessage()
    ]
    assert len(messages) == expected_warning_count


def test_cost_header_probe_does_not_swallow_unrelated_warnings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan, provider, _ = _build_provider(tmp_path)
    original_open = Image.open

    def _open_with_unrelated_warning(*args, **kwargs):
        warnings.warn("independent image backend warning", UserWarning, stacklevel=1)
        return original_open(*args, **kwargs)

    monkeypatch.setattr(Image, "open", _open_with_unrelated_warning)

    with pytest.warns(UserWarning, match="independent image backend warning"):
        provider(plan.ref_at(0))


def test_provider_rejects_tokenizer_without_stable_artifact_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shaft_cost_fingerprint"):
        _build_provider(tmp_path, tokenizer=_UndeclaredCostTokenizer())


def test_provider_rejects_undeclared_online_transform(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64)).save(image_path)
    plan = ShaftSamplePlan(
        {"dataset": 1},
        {"dataset": 1.0},
        strategy="concat",
        shuffle=False,
    )
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
        media_snapshot_id="test-media-v1",
        online_transforms=[lambda item: item],
    )

    with pytest.raises(ValueError, match="unsafe transforms"):
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
