from __future__ import annotations

from pathlib import Path

import pytest

from shaft.webui.services import ShaftWebUIConfigService
from shaft.webui.types import ShaftSFTWebUIOverrides


def test_webui_config_service_resolves_relative_paths_and_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text("", encoding="utf-8")
    val_path.write_text("", encoding="utf-8")
    config_path.write_text(
        """
experiment:
  name: demo
  output_dir: outputs/demo
algorithm:
  name: sft
data:
  datasets:
    - dataset_name: ds
      train_path: train.jsonl
      val_path: val.jsonl
train:
  report_to: ["none"]
eval:
  enabled: true
""",
        encoding="utf-8",
    )
    service = ShaftWebUIConfigService()
    yaml_text = service.read_config_text(config_path)
    config, rendered = service.resolve_sft_config(
        config_path=config_path,
        yaml_text=yaml_text,
        overrides=ShaftSFTWebUIOverrides(
            run_id="web-run",
            learning_rate=2e-5,
            loss_scale="all",
            freeze_groups="vision_tower,aligner",
            use_cpu=True,
        ),
    )
    assert config.algorithm.name == "sft"
    assert config.experiment.run_id == "web-run"
    assert config.train.learning_rate == pytest.approx(2e-5)
    assert config.train.loss_scale == "all"
    assert config.train.use_cpu is True
    assert config.model.finetune.freeze.groups == ["vision_tower", "aligner"]
    assert config.data.datasets[0].train_paths == [str(train_path.resolve())]
    assert config.data.datasets[0].val_paths == [str(val_path.resolve())]
    assert "run_id: web-run" in rendered
    assert "loss_scale: all" in rendered


def test_webui_config_service_rejects_non_sft_config(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
algorithm:
  name: dpo
data:
  datasets:
    - dataset_name: ds
      source_type: jsonl_dpo
      train_path: train.jsonl
      val_path: val.jsonl
train:
  report_to: ["none"]
eval:
  enabled: true
""",
        encoding="utf-8",
    )
    service = ShaftWebUIConfigService()
    with pytest.raises(ValueError, match="只支持 SFT"):
        service.resolve_sft_config(
            config_path=config_path,
            yaml_text=service.read_config_text(config_path),
            overrides=ShaftSFTWebUIOverrides(),
        )
