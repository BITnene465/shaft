from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from shaft.model import (
    ProcessorPolicy,
    ShaftProcessedBatch,
    ShaftProcessorCostEstimate,
    build_model_meta,
)


def test_processor_policy_controls_pixel_budget_forwarding() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    captured = {}

    class _Processor:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {
                "input_ids": torch.tensor([[1]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
            }

    _ = model_adapter.build_processor_batch(
        processor=_Processor(),
        prompt_texts=["hello"],
        images=["img"],
        min_pixels=16,
        max_pixels=32,
    )
    assert "min_pixels" not in captured
    assert "max_pixels" not in captured
    assert captured["images_kwargs"] == {"min_pixels": 16, "max_pixels": 32}


def test_qwen_processor_policy_estimates_resized_image_tokens_and_patches() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            patch_size=16,
            merge_size=2,
            size=SimpleNamespace(shortest_edge=None, longest_edge=None),
            get_number_of_image_patches=lambda **kwargs: 16,
        )
    )

    estimate = model_adapter.estimate_processor_image_cost(
        processor=processor,
        image_sizes=((64, 64),),
        min_pixels=None,
        max_pixels=None,
    )

    assert estimate.processed_image_tokens == 4
    assert estimate.vision_patches == 16
    assert estimate.exact is True

    layout = model_adapter.estimate_processor_token_layout(
        processor=SimpleNamespace(image_token_id=99),
        tokenizer=SimpleNamespace(),
        rendered_token_ids=(10, 99, 11),
        image_costs=(estimate,),
    )
    assert layout.processed_boundaries == (0, 1, 5, 6)


def test_qwen_processor_policy_rejects_mismatched_image_costs() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    estimate = ShaftProcessorCostEstimate(
        processed_image_tokens=4,
        vision_patches=16,
        exact=True,
    )

    with pytest.raises(ValueError, match="placeholder count"):
        model_adapter.estimate_processor_token_layout(
            processor=SimpleNamespace(image_token_id=99),
            tokenizer=SimpleNamespace(),
            rendered_token_ids=(10, 11),
            image_costs=(estimate,),
        )


def test_processor_policy_can_disable_pixel_budget() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(
        model_name_or_path="models/Smoke-VLM"
    )
    captured = {}

    class _Processor:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {
                "input_ids": torch.tensor([[1]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
            }

    _ = model_adapter.build_processor_batch(
        processor=_Processor(),
        prompt_texts=["hello"],
        images=["img"],
        min_pixels=16,
        max_pixels=32,
    )
    assert "min_pixels" not in captured
    assert "max_pixels" not in captured
    assert "images_kwargs" not in captured


def test_processor_policy_temporarily_controls_padding_side() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    captured = {}

    class _Tokenizer:
        def __init__(self) -> None:
            self.padding_side = "right"

    tokenizer = _Tokenizer()

    class _Processor:
        def __init__(self, tokenizer_obj) -> None:
            self.tokenizer = tokenizer_obj

        def __call__(self, **kwargs):
            captured["padding_side_during_call"] = self.tokenizer.padding_side
            return {
                "input_ids": torch.tensor([[1]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
            }

    processor = _Processor(tokenizer)
    _ = model_adapter.build_processor_batch(
        processor=processor,
        tokenizer=tokenizer,
        prompt_texts=["hello"],
        images=["img"],
        min_pixels=16,
        max_pixels=32,
        padding_side="left",
    )
    assert captured["padding_side_during_call"] == "left"
    assert tokenizer.padding_side == "right"


def test_qwen_processor_policy_maps_expanded_multimodal_token_runs() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )

    layout = model_adapter.build_processor_token_layout(
        rendered_token_ids=(10, 11, 12),
        processed_batch=ShaftProcessedBatch(
            model_inputs={
                "input_ids": torch.tensor([[10, 10, 10, 11, 12]], dtype=torch.long),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long),
            },
            batch_size=1,
        ),
        row_index=0,
    )

    assert layout.processed_boundaries == (0, 3, 4, 5)
    assert layout.project_span(1, 2) == (3, 4)


@pytest.mark.parametrize(
    ("input_ids", "attention_mask", "mm_token_type_ids"),
    [
        ([10, 10, 11, 12, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]),
        ([0, 10, 10, 11, 12], [0, 1, 1, 1, 1], [0, 1, 1, 0, 0]),
    ],
    ids=("right-padding", "left-padding"),
)
def test_qwen_processor_policy_token_layout_ignores_padding(
    input_ids: list[int],
    attention_mask: list[int],
    mm_token_type_ids: list[int],
) -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )

    layout = model_adapter.build_processor_token_layout(
        rendered_token_ids=(10, 11, 12),
        processed_batch=ShaftProcessedBatch(
            model_inputs={
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
                "mm_token_type_ids": torch.tensor([mm_token_type_ids], dtype=torch.long),
            },
            batch_size=1,
        ),
        row_index=0,
    )

    assert layout.processed_boundaries == (0, 2, 3, 4)


def test_identity_processor_policy_requires_exact_token_layout() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(
        model_name_or_path="models/Smoke-VLM"
    )

    layout = model_adapter.build_processor_token_layout(
        rendered_token_ids=(10, 11, 12),
        processed_batch=ShaftProcessedBatch(
            model_inputs={
                "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            },
            batch_size=1,
        ),
        row_index=0,
    )
    assert layout.processed_boundaries == (0, 1, 2, 3)

    with pytest.raises(ValueError, match="register a model-specific processor policy"):
        model_adapter.build_processor_token_layout(
            rendered_token_ids=(10, 11, 12),
            processed_batch=ShaftProcessedBatch(
                model_inputs={
                    "input_ids": torch.tensor([[10, 99, 12]], dtype=torch.long),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long),
                },
                batch_size=1,
            ),
            row_index=0,
        )


def test_qwen_processor_policy_refuses_missing_multimodal_token_types() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )

    with pytest.raises(ValueError, match="register a model-specific processor policy"):
        model_adapter.build_processor_token_layout(
            rendered_token_ids=(10, 11),
            processed_batch=ShaftProcessedBatch(
                model_inputs={
                    "input_ids": torch.tensor([[10, 10, 11]], dtype=torch.long),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long),
                },
                batch_size=1,
            ),
            row_index=0,
        )


def test_processor_policy_owns_dpo_expansion_for_all_model_inputs() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    model_adapter = replace(
        model_adapter,
        processor_policy=replace(
            model_adapter.processor_policy,
            sample_aligned_model_input_names=("model_specific_mask",),
        ),
    )
    processed_batch = ShaftProcessedBatch(
        model_inputs={
            "input_ids": torch.tensor([[10], [11]], dtype=torch.long),
            "attention_mask": torch.ones((2, 1), dtype=torch.long),
            # Qwen flattens image patches rather than exposing a sample batch axis.
            "pixel_values": torch.arange(20, dtype=torch.float32).reshape(5, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 1, 1]], dtype=torch.long),
            "model_specific_mask": torch.tensor([[3], [4]], dtype=torch.long),
        },
        batch_size=2,
    )

    assembled = model_adapter.assemble_processor_training_inputs(
        processed_batch=processed_batch,
        sequence_inputs={
            "input_ids": torch.tensor([[10], [11], [10], [11]], dtype=torch.long),
            "attention_mask": torch.ones((4, 1), dtype=torch.long),
        },
        row_indices=(0, 1, 0, 1),
    )

    assert assembled["pixel_values"].shape == (10, 4)
    assert torch.equal(assembled["pixel_values"][:5], assembled["pixel_values"][5:])
    assert assembled["image_grid_thw"].tolist() == [
        [1, 2, 2],
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
    ]
    assert assembled["model_specific_mask"].flatten().tolist() == [3, 4, 3, 4]


def test_processor_policy_fails_on_unowned_sequence_aligned_output() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(
        model_name_or_path="models/Smoke-VLM"
    )
    processed_batch = ShaftProcessedBatch(
        model_inputs={
            "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        },
        batch_size=1,
    )

    with pytest.raises(ValueError, match="must explicitly assemble sequence-aligned"):
        model_adapter.assemble_processor_training_inputs(
            processed_batch=processed_batch,
            sequence_inputs={
                "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            },
            row_indices=(0,),
        )


def test_processor_policy_fails_on_undeclared_dpo_output() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    processed_batch = ShaftProcessedBatch(
        model_inputs={
            "input_ids": torch.tensor([[10], [11]], dtype=torch.long),
            "attention_mask": torch.ones((2, 1), dtype=torch.long),
            "unknown_flattened_state": torch.zeros((3, 4), dtype=torch.float32),
        },
        batch_size=2,
    )

    with pytest.raises(ValueError, match="does not declare the layout"):
        model_adapter.assemble_processor_training_inputs(
            processed_batch=processed_batch,
            sequence_inputs={
                "input_ids": torch.tensor([[10], [11], [10], [11]], dtype=torch.long),
                "attention_mask": torch.ones((4, 1), dtype=torch.long),
            },
            row_indices=(0, 1, 0, 1),
        )


def test_processor_policy_rejects_ambiguous_field_layout_declarations() -> None:
    with pytest.raises(ValueError, match="declared by both"):
        ProcessorPolicy(
            sample_aligned_model_input_names=("vision_state",),
            whole_batch_model_input_names=("vision_state",),
        )
