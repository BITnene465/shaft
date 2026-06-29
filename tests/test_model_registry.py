from __future__ import annotations

from unittest.mock import patch

import pytest

from shaft.config import RuntimeConfig
from shaft.model import (
    MODEL_REGISTRY,
    PEFT_POLICY_REGISTRY,
    PROCESSOR_POLICY_REGISTRY,
    build_model_tokenizer_processor,
)


def test_qwen3vl_registered() -> None:
    assert MODEL_REGISTRY.has("qwen3vl")


def test_qwen35vl_registered() -> None:
    assert MODEL_REGISTRY.has("qwen35vl")
    assert MODEL_REGISTRY.has("qwen36vl")


def test_builtin_model_policies_registered() -> None:
    assert PROCESSOR_POLICY_REGISTRY.has("pixel_budget")
    assert PROCESSOR_POLICY_REGISTRY.has("no_pixel_budget")
    assert PEFT_POLICY_REGISTRY.has("all_linear")


def test_unknown_model_type_raises() -> None:
    config = RuntimeConfig()
    config.model.model_type = "unknown_model"
    with pytest.raises(KeyError):
        build_model_tokenizer_processor(config)


def test_builder_dispatches_registry() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    fake_artifacts = object()
    fake_meta = type(
        "Meta",
        (),
        {
            "resolve_adapter": lambda self, *, model_name_or_path, template_type=None: type(
                "Adapter",
                (),
                {"check_requires": lambda self: None},
            )(),
            "loader": type(
                "Loader",
                (),
                {
                    "build": lambda self, cfg, *, model_meta, model_adapter: fake_artifacts,
                },
            )(),
        },
    )()
    with patch("shaft.model.builder.build_model_meta", return_value=fake_meta) as mocked:
        out = build_model_tokenizer_processor(config)
    mocked.assert_called_once_with("qwen3vl")
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
    assert artifacts.model_adapter.capabilities.supports_pixel_budget is False
    assert artifacts.template.name == "smoke_vlm"
    assert artifacts.template.template_meta.template_type == "smoke_vlm"
