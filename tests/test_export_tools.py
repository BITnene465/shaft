from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from shaft.config import RuntimeConfig
from shaft.export import infer_base_model_from_adapter, merge_peft_adapter, validate_hf_artifact
from shaft.model import build_model_tokenizer_processor, build_model_meta
from shaft.training.checkpointing import ensure_hf_export_layout


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


def test_merge_peft_adapter_exports_full_layout(tmp_path: Path) -> None:
    artifacts = _build_smoke_lora_artifacts()
    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)

    output_dir = tmp_path / "merged"
    result = merge_peft_adapter(
        model_type="smoke_vlm",
        adapter_path=adapter_dir,
        output_dir=output_dir,
        base_model_path="models/smoke-vlm",
        torch_dtype="float32",
    )

    assert result.output_dir == output_dir
    ensure_hf_export_layout(
        output_dir,
        finetune_mode="full",
        model_meta=build_model_meta("smoke_vlm"),
    )
    assert (output_dir / "smoke_tokenizer.json").exists()
    assert (output_dir / "smoke_processor.json").exists()


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
    with patch("shaft.export.hf.build_model_tokenizer_processor", wraps=build_model_tokenizer_processor) as mocked:
        merge_peft_adapter(
            model_type="smoke_vlm",
            adapter_path=adapter_dir,
            output_dir=output_dir,
            torch_dtype="float32",
        )
    mocked.assert_called_once()


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
