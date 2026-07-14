from __future__ import annotations

from dataclasses import replace
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest

from shaft.config import RuntimeConfig
from shaft.model import (
    build_model_meta,
    build_model_tokenizer_processor,
)


@pytest.mark.parametrize(
    "mutation",
    [
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "size",
        "image_token_id",
        "image_token",
        "min_pixels",
        "max_pixels",
        "estimator",
    ],
)
def test_qwen_processor_cost_signature_binds_policy_owned_processor_state(
    mutation: str,
) -> None:
    class ImageProcessor:
        def __init__(self) -> None:
            self.patch_size = 14
            self.temporal_patch_size = 2
            self.merge_size = 2
            self.size = {"shortest_edge": 56}

        def get_number_of_image_patches(self, *, height, width, images_kwargs):
            _ = height, width, images_kwargs
            return 16

    class Processor:
        def __init__(self) -> None:
            self.image_token_id = 151655
            self.image_token = "<|image_pad|>"
            self.image_processor = ImageProcessor()

    def alternative_patch_estimator(self, *, height, width, images_kwargs):
        _ = self, height, width, images_kwargs
        return 16

    adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    first_processor = Processor()
    first = adapter.processor_cost_semantics_signature(
        processor=first_processor,
        min_pixels=1024,
        max_pixels=4096,
    )
    second_processor = Processor()
    min_pixels = 1024
    max_pixels = 4096
    if mutation == "patch_size":
        second_processor.image_processor.patch_size = 16
    elif mutation == "temporal_patch_size":
        second_processor.image_processor.temporal_patch_size = 4
    elif mutation == "merge_size":
        second_processor.image_processor.merge_size = 4
    elif mutation == "size":
        second_processor.image_processor.size = {"shortest_edge": 112}
    elif mutation == "image_token_id":
        second_processor.image_token_id = 151656
    elif mutation == "image_token":
        second_processor.image_token = "<|different_image_pad|>"
    elif mutation == "min_pixels":
        min_pixels = 2048
    elif mutation == "max_pixels":
        max_pixels = 8192
    elif mutation == "estimator":
        second_processor.image_processor.get_number_of_image_patches = MethodType(
            alternative_patch_estimator,
            second_processor.image_processor,
        )
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)
    second = adapter.processor_cost_semantics_signature(
        processor=second_processor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    assert first != second


def test_qwen_exact_cost_rejects_unforwarded_pixel_budget() -> None:
    class ImageProcessor:
        patch_size = 14
        temporal_patch_size = 2
        merge_size = 2
        size = {"shortest_edge": 56}

        def get_number_of_image_patches(self, *, height, width, images_kwargs):
            _ = height, width, images_kwargs
            return 16

    class Processor:
        image_token_id = 151655
        image_processor = ImageProcessor()

    adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    disabled_policy = replace(
        adapter.processor_policy,
        supports_pixel_budget=False,
    )

    with pytest.raises(ValueError, match="does not forward pixel budgets"):
        disabled_policy.cost_semantics_signature(
            processor=Processor(),
            min_pixels=1024,
            max_pixels=4096,
        )
    assert disabled_policy.cost_semantics_signature(
        processor=Processor(),
        min_pixels=None,
        max_pixels=None,
    ) != adapter.processor_policy.cost_semantics_signature(
        processor=Processor(),
        min_pixels=None,
        max_pixels=None,
    )


def test_unknown_model_type_raises() -> None:
    config = RuntimeConfig()
    config.model.model_type = "unknown_model"
    with pytest.raises(KeyError):
        build_model_tokenizer_processor(config)


def test_builder_dispatches_registry() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    fake_artifacts = object()

    def build_fake_artifacts(
        self,
        cfg,
        *,
        model_meta,
        model_adapter,
        sequence_execution_contract=None,
    ):
        _ = self, cfg, model_meta, model_adapter, sequence_execution_contract
        return fake_artifacts

    fake_meta = SimpleNamespace(
        loader=type("Loader", (), {"build": build_fake_artifacts})(),
    )
    fake_adapter = SimpleNamespace(check_requires=lambda: None)
    fake_plan = SimpleNamespace(
        init_from_checkpoint=None,
        init_kind="base",
        effective_model_name_or_path=config.model.model_name_or_path,
        model_meta=fake_meta,
        model_adapter=fake_adapter,
    )
    with patch(
        "shaft.model.builder.resolve_model_plan",
        return_value=fake_plan,
    ) as mocked:
        out = build_model_tokenizer_processor(config)
    mocked.assert_called_once_with(config, init_from_checkpoint=None)
    assert out is fake_artifacts


def test_smoke_artifacts_expose_meta_and_template() -> None:
    cfg = RuntimeConfig()
    cfg.model.model_type = "smoke_vlm"
    artifacts = build_model_tokenizer_processor(cfg)
    assert artifacts.model_meta.model_type == "smoke_vlm"
    assert artifacts.model_adapter.model_type == "smoke_vlm"
    assert artifacts.model_info.model_type == "smoke_vlm"
    assert artifacts.model_info.model_dir == cfg.model.model_name_or_path
    assert artifacts.model_info.is_multimodal is True
    assert artifacts.model_adapter.default_target_modules() == ["all-linear"]
    assert artifacts.model_adapter.processor_policy.supports_pixel_budget is False
    assert artifacts.template.name == "smoke_vlm"
    assert artifacts.template.template_meta.template_type == "smoke_vlm"
