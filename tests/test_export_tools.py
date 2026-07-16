from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from safetensors.torch import load_file, save_file
import torch

import shaft.model.builder as model_builder
from shaft.config import RuntimeConfig
from shaft.export import (
    infer_base_model_from_adapter,
    merge_peft_adapter as _merge_peft_adapter,
    validate_hf_artifact,
)
from shaft.model import (
    build_model_meta,
    build_model_tokenizer_processor,
    LoadedAdapterArtifacts,
    load_adapter_artifacts,
    resolve_model_plan,
)
from shaft.model.smoke_vlm import SmokeVLMModel
from shaft.training.checkpointing import ensure_hf_export_layout


def merge_peft_adapter(**kwargs):
    """Existing synthetic adapters intentionally exercise the explicit unsafe path."""

    kwargs.setdefault("allow_unverified_base_model", True)
    return _merge_peft_adapter(**kwargs)


def _build_smoke_lora_artifacts():
    cfg = RuntimeConfig()
    cfg.model.model_type = "smoke_vlm"
    cfg.model.model_name_or_path = "models/smoke-vlm"
    cfg.model.finetune.mode = "lora"
    cfg.model.finetune.target_modules = ["all-linear"]
    return build_model_tokenizer_processor(cfg)


def test_infer_base_model_from_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "models/qwen-base"}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert infer_base_model_from_adapter(adapter_dir) == "models/qwen-base"


def test_validate_hf_artifact_with_model_meta(tmp_path: Path) -> None:
    export_dir = tmp_path / "full"
    export_dir.mkdir()
    (export_dir / "config.json").write_text("{}", encoding="utf-8")
    (export_dir / "model.safetensors").write_bytes(b"ok")
    (export_dir / "smoke_tokenizer.json").write_text("{}", encoding="utf-8")
    (export_dir / "smoke_processor.json").write_text("{}", encoding="utf-8")
    layout = validate_hf_artifact(
        export_dir,
        finetune_mode="full",
        model_type="smoke_vlm",
        model_name_or_path="models/smoke-vlm",
    )
    assert layout.kind == "full"


def test_validate_hf_artifact_resolves_custom_qwen_moe_descriptor(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "custom-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    export_dir = tmp_path / "full-moe"
    export_dir.mkdir()
    (export_dir / "config.json").write_text("{}", encoding="utf-8")
    (export_dir / "model.safetensors").write_bytes(b"ok")

    with patch("shaft.export.hf.ensure_hf_export_layout") as validate_layout:
        validate_hf_artifact(
            export_dir,
            finetune_mode="full",
            model_type="qwen36vl",
            model_name_or_path=str(model_dir),
        )

    resolved_adapter = validate_layout.call_args.kwargs["model_meta"]
    assert resolved_adapter.group_name == "moe"


def test_merge_peft_adapter_exports_full_layout(tmp_path: Path) -> None:
    torch.manual_seed(17)
    artifacts = _build_smoke_lora_artifacts()
    for name, parameter in artifacts.model.named_parameters():
        if "lora_A" in name:
            parameter.data.fill_(0.125)
        elif "lora_B" in name:
            parameter.data.fill_(0.25)
    input_ids = torch.tensor([[1, 7, 11, 2]], dtype=torch.long)
    with torch.no_grad():
        expected_logits = artifacts.model(input_ids=input_ids).logits
    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)

    output_dir = tmp_path / "merged"
    torch.manual_seed(17)
    base_config = RuntimeConfig()
    base_config.model.model_type = "smoke_vlm"
    base_config.model.model_name_or_path = "models/smoke-vlm"
    base_plan = resolve_model_plan(base_config, require_immutable_artifact=True)
    metadata = SimpleNamespace(
        train_input_contract=SimpleNamespace(
            model_plan_fingerprint=base_plan.fingerprint,
        )
    )
    with patch(
        "shaft.export.hf.load_checkpoint_batching_metadata",
        return_value=metadata,
    ):
        result = merge_peft_adapter(
            model_type="smoke_vlm",
            adapter_path=adapter_dir,
            output_dir=output_dir,
            base_model_path="models/smoke-vlm",
            torch_dtype="float32",
            allow_unverified_base_model=False,
        )

    assert result.output_dir == output_dir
    ensure_hf_export_layout(
        output_dir,
        finetune_mode="full",
        model_meta=build_model_meta("smoke_vlm"),
    )
    assert (output_dir / "smoke_tokenizer.json").exists()
    assert (output_dir / "smoke_processor.json").exists()
    merged_model = SmokeVLMModel.from_pretrained(output_dir)
    with torch.no_grad():
        actual_logits = merged_model(input_ids=input_ids).logits
    assert torch.allclose(actual_logits, expected_logits, atol=1e-6, rtol=1e-6)


def test_merge_peft_adapter_rejects_adapter_without_base_provenance(
    tmp_path: Path,
) -> None:
    artifacts = _build_smoke_lora_artifacts()
    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)

    with pytest.raises(ValueError, match="cannot prove which base-model bytes"):
        _merge_peft_adapter(
            model_type="smoke_vlm",
            adapter_path=adapter_dir,
            output_dir=tmp_path / "merged",
            base_model_path="models/smoke-vlm",
            torch_dtype="float32",
        )


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_merge_peft_adapter_rejects_non_boolean_provenance_override(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(TypeError, match="allow_unverified_base_model must be a boolean"):
        _merge_peft_adapter(
            model_type="smoke_vlm",
            adapter_path=tmp_path / "missing-adapter",
            output_dir=tmp_path / "merged",
            base_model_path="models/smoke-vlm",
            allow_unverified_base_model=value,
        )


def test_merge_peft_adapter_rejects_mismatched_base_provenance(
    tmp_path: Path,
) -> None:
    artifacts = _build_smoke_lora_artifacts()
    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)
    metadata = SimpleNamespace(
        train_input_contract=SimpleNamespace(
            model_plan_fingerprint="different-base-plan",
        )
    )

    with (
        patch(
            "shaft.export.hf.load_checkpoint_batching_metadata",
            return_value=metadata,
        ),
        pytest.raises(ValueError, match="base-model identity differs"),
    ):
        _merge_peft_adapter(
            model_type="smoke_vlm",
            adapter_path=adapter_dir,
            output_dir=tmp_path / "merged",
            base_model_path="models/smoke-vlm",
            torch_dtype="float32",
        )


def test_merge_peft_adapter_falls_back_to_parent_run_provenance(
    tmp_path: Path,
) -> None:
    artifacts = _build_smoke_lora_artifacts()
    adapter_dir = tmp_path / "run" / "best"
    artifacts.model.save_pretrained(adapter_dir)
    base_config = RuntimeConfig()
    base_config.model.model_type = "smoke_vlm"
    base_config.model.model_name_or_path = "models/smoke-vlm"
    base_plan = resolve_model_plan(base_config, require_immutable_artifact=True)
    legacy_metadata = SimpleNamespace(train_input_contract=None)
    parent_metadata = SimpleNamespace(
        train_input_contract=SimpleNamespace(
            model_plan_fingerprint=base_plan.fingerprint,
        )
    )

    with (
        patch(
            "shaft.export.hf.load_checkpoint_batching_metadata",
            return_value=legacy_metadata,
        ),
        patch(
            "shaft.export.hf.load_batching_run_metadata",
            side_effect=[FileNotFoundError("no local metadata"), parent_metadata],
        ) as run_loader,
    ):
        result = _merge_peft_adapter(
            model_type="smoke_vlm",
            adapter_path=adapter_dir,
            output_dir=tmp_path / "merged",
            base_model_path="models/smoke-vlm",
            torch_dtype="float32",
            allow_unverified_base_model=False,
        )

    assert result.layout.kind == "full"
    assert [call.args[0] for call in run_loader.call_args_list] == [
        adapter_dir,
        adapter_dir.parent,
    ]


def test_merge_peft_adapter_rejects_non_empty_output_dir(tmp_path: Path) -> None:
    artifacts = _build_smoke_lora_artifacts()
    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)

    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    (output_dir / "already.txt").write_text("x", encoding="utf-8")

    try:
        merge_peft_adapter(
            model_type="smoke_vlm",
            adapter_path=adapter_dir,
            output_dir=output_dir,
            base_model_path="models/smoke-vlm",
        )
    except ValueError as exc:
        assert "must be empty" in str(exc)
    else:
        raise AssertionError("Expected non-empty output dir to be rejected.")


def test_merge_peft_adapter_reads_base_model_from_adapter(tmp_path: Path) -> None:
    artifacts = _build_smoke_lora_artifacts()
    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)
    payload = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    payload["base_model_name_or_path"] = "models/smoke-vlm"
    (adapter_dir / "adapter_config.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    output_dir = tmp_path / "merged"
    with patch(
        "shaft.export.hf.load_adapter_artifacts",
        wraps=load_adapter_artifacts,
    ) as mocked:
        merge_peft_adapter(
            model_type="smoke_vlm",
            adapter_path=adapter_dir,
            output_dir=output_dir,
            torch_dtype="float32",
        )
    mocked.assert_called_once()
    assert mocked.call_args.kwargs["adapter_path"] == str(adapter_dir)
    plan = mocked.call_args.kwargs["resolved_model_plan"]
    assert plan.init_kind == "adapter"
    assert plan.adapter_init is not None


def test_merge_peft_adapter_prefers_adapter_processing_assets(tmp_path: Path) -> None:
    artifacts = _build_smoke_lora_artifacts()
    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)
    payload = {"source": "adapter"}
    (adapter_dir / "smoke_tokenizer.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (adapter_dir / "smoke_processor.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    output_dir = tmp_path / "merged"
    merge_peft_adapter(
        model_type="smoke_vlm",
        adapter_path=adapter_dir,
        output_dir=output_dir,
        base_model_path="models/smoke-vlm",
        torch_dtype="float32",
    )

    assert json.loads((output_dir / "smoke_tokenizer.json").read_text(encoding="utf-8")) == payload
    assert json.loads((output_dir / "smoke_processor.json").read_text(encoding="utf-8")) == payload


def test_merge_peft_adapter_preserves_modules_to_save_weights(tmp_path: Path) -> None:
    cfg = RuntimeConfig()
    cfg.model.model_type = "smoke_vlm"
    cfg.model.model_name_or_path = "models/smoke-vlm"
    cfg.model.finetune.mode = "lora"
    cfg.model.finetune.target_modules = ["all-linear"]
    cfg.model.finetune.freeze.trainable_prefixes = ["lm_head"]
    artifacts = build_model_tokenizer_processor(cfg)
    for name, parameter in artifacts.model.named_parameters():
        if name.endswith("lm_head.modules_to_save.default.weight"):
            parameter.data.fill_(0.375)

    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)

    output_dir = tmp_path / "merged"
    merge_peft_adapter(
        model_type="smoke_vlm",
        adapter_path=adapter_dir,
        output_dir=output_dir,
        base_model_path="models/smoke-vlm",
        torch_dtype="float32",
    )

    merged_model = SmokeVLMModel.from_pretrained(output_dir)
    expected = torch.full_like(merged_model.lm_head.weight, 0.375)
    assert torch.allclose(merged_model.lm_head.weight, expected)


def test_merge_peft_adapter_rejects_dense_base_for_moe_adapter(
    tmp_path: Path,
) -> None:
    dense_dir = tmp_path / "dense"
    dense_dir.mkdir()
    (dense_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )
    moe_dir = tmp_path / "moe"
    moe_dir.mkdir()
    (moe_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(moe_dir),
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"placeholder")
    output_dir = tmp_path / "merged"

    with pytest.raises(ValueError, match="base variant differs"):
        merge_peft_adapter(
            model_type="qwen36vl",
            adapter_path=adapter_dir,
            output_dir=output_dir,
            base_model_path=str(dense_dir),
            local_files_only=True,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_merge_peft_adapter_rejects_inexact_state(
    tmp_path: Path,
    corruption: str,
) -> None:
    artifacts = _build_smoke_lora_artifacts()
    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)
    weight_path = adapter_dir / "adapter_model.safetensors"
    state = load_file(weight_path)
    first_key = next(iter(state))
    if corruption == "missing":
        state.pop(first_key)
    elif corruption == "unexpected":
        state["unexpected.weight"] = torch.zeros(1)
    else:
        state[first_key] = state[first_key][:-1]
    save_file(state, weight_path)
    output_dir = tmp_path / "merged"

    with pytest.raises(ValueError, match="does not exactly match"):
        merge_peft_adapter(
            model_type="smoke_vlm",
            adapter_path=adapter_dir,
            output_dir=output_dir,
            base_model_path="models/smoke-vlm",
            torch_dtype="float32",
        )

    assert not output_dir.exists()


def test_load_adapter_artifacts_rejects_config_change_after_plan(
    tmp_path: Path,
) -> None:
    artifacts = _build_smoke_lora_artifacts()
    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)
    config = RuntimeConfig()
    config.model.model_type = "smoke_vlm"
    config.model.model_name_or_path = "models/smoke-vlm"
    plan = resolve_model_plan(config, init_from_checkpoint=str(adapter_dir))
    config_path = adapter_dir / "adapter_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["r"] = int(payload["r"]) + 1
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="config changed after ResolvedModelPlan"):
        load_adapter_artifacts(
            config,
            adapter_path=str(adapter_dir),
            resolved_model_plan=plan,
        )


def test_load_adapter_artifacts_has_adapter_specific_result(tmp_path: Path) -> None:
    artifacts = _build_smoke_lora_artifacts()
    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)
    config = RuntimeConfig()
    config.model.model_type = "smoke_vlm"
    config.model.model_name_or_path = "models/smoke-vlm"

    loaded = load_adapter_artifacts(config, adapter_path=str(adapter_dir))

    assert isinstance(loaded, LoadedAdapterArtifacts)
    assert not hasattr(loaded, "finetune_plan")


def test_load_adapter_artifacts_accepts_verified_bin_state(tmp_path: Path) -> None:
    artifacts = _build_smoke_lora_artifacts()
    adapter_dir = tmp_path / "adapter-bin"
    artifacts.model.save_pretrained(adapter_dir, safe_serialization=False)
    assert (adapter_dir / "adapter_model.bin").is_file()
    config = RuntimeConfig()
    config.model.model_type = "smoke_vlm"
    config.model.model_name_or_path = "models/smoke-vlm"

    loaded = load_adapter_artifacts(config, adapter_path=str(adapter_dir))

    assert isinstance(loaded, LoadedAdapterArtifacts)


def test_load_adapter_artifacts_rejects_weight_change_during_base_build(
    tmp_path: Path,
) -> None:
    artifacts = _build_smoke_lora_artifacts()
    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)
    weight_path = adapter_dir / "adapter_model.safetensors"
    replacement = bytearray(weight_path.read_bytes())
    replacement[-1] ^= 1
    original_size = weight_path.stat().st_size
    config = RuntimeConfig()
    config.model.model_type = "smoke_vlm"
    config.model.model_name_or_path = "models/smoke-vlm"
    plan = resolve_model_plan(config, init_from_checkpoint=str(adapter_dir))
    real_loader = model_builder.invoke_model_loader

    def _load_then_replace(prepared):
        built = real_loader(prepared)
        weight_path.write_bytes(replacement)
        assert weight_path.stat().st_size == original_size
        return built

    with (
        patch.object(
            model_builder,
            "invoke_model_loader",
            side_effect=_load_then_replace,
        ),
        pytest.raises(ValueError, match="weights changed after ResolvedModelPlan"),
    ):
        load_adapter_artifacts(
            config,
            adapter_path=str(adapter_dir),
            resolved_model_plan=plan,
        )
