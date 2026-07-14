from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from shaft.config import RuntimeConfig
from shaft.model import build_model_meta, build_model_tokenizer_processor
from shaft.model.builder import (
    _validate_hf_sharded_checkpoint_files,
    _validate_local_hf_config_model_type,
)


def test_model_loader_uses_effective_gradient_checkpointing_for_fsdp() -> None:
    cfg = RuntimeConfig()
    cfg.model.model_type = "smoke_vlm"
    cfg.train.gradient_checkpointing = True
    cfg.train.distributed.strategy = "fsdp"
    cfg.train.distributed.fsdp.activation_checkpointing = True

    artifacts = build_model_tokenizer_processor(cfg)

    assert getattr(artifacts.model.config, "use_cache", True) is True


def test_hf_sharded_checkpoint_validation_allows_complete_index(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"")
    (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    _validate_hf_sharded_checkpoint_files(model_dir)


def test_hf_sharded_checkpoint_validation_reports_incomplete_download(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"")
    temp_dir = model_dir / "._____temp"
    temp_dir.mkdir()
    (temp_dir / "model-00002-of-00002.safetensors").write_bytes(b"partial")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="download is likely incomplete"):
        _validate_hf_sharded_checkpoint_files(model_dir)


def test_builder_rejects_incomplete_hf_sharded_checkpoint_before_loader(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"")
    temp_dir = model_dir / "._____temp"
    temp_dir.mkdir()
    (temp_dir / "model-00002-of-00002.safetensors").write_bytes(b"partial")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    config = RuntimeConfig()
    config.model.model_type = "smoke_vlm"
    config.model.model_name_or_path = str(model_dir)

    with pytest.raises(FileNotFoundError, match="download is likely incomplete"):
        build_model_tokenizer_processor(config)


def test_builder_rejects_local_hf_config_model_type_mismatch_before_loader(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )

    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = str(model_dir)

    with pytest.raises(ValueError, match="does not match model.model_type='qwen36vl'"):
        build_model_tokenizer_processor(config)


def test_local_hf_config_model_type_validation_accepts_qwen36_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )

    _validate_local_hf_config_model_type(model_dir, model_meta=build_model_meta("qwen36vl"))


def test_init_from_full_checkpoint_overrides_model_path(tmp_path: Path) -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    init_ckpt = tmp_path / "full_ckpt"
    init_ckpt.mkdir(parents=True, exist_ok=True)
    (init_ckpt / "config.json").write_text("{}", encoding="utf-8")

    captured = {}

    class _Adapter:
        loader = None

        def __init__(self):
            self.loader = type("Loader", (), {"build": self._build})()

        def resolve_adapter(self, *, model_name_or_path, template_type=None):
            _ = template_type
            return type(
                "ResolvedAdapter",
                (),
                {
                    "check_requires": lambda self: None,
                    "model_name_or_path": model_name_or_path,
                },
            )()

        def _build(
            self,
            cfg,
            *,
            model_meta,
            model_adapter,
            sequence_execution_contract=None,
        ):
            captured["cfg"] = cfg
            captured["model_meta"] = model_meta
            captured["model_adapter"] = model_adapter
            captured["sequence_execution_contract"] = sequence_execution_contract
            return object()

    with patch("shaft.model.builder.build_model_meta", return_value=_Adapter()):
        _ = build_model_tokenizer_processor(config, init_from_checkpoint=str(init_ckpt))
    called_cfg = captured["cfg"]
    assert called_cfg is not config
    assert called_cfg.model.model_name_or_path == str(init_ckpt)
    assert captured["model_meta"] is not None
    assert captured["model_adapter"].model_name_or_path == str(init_ckpt)
