from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import RuntimeConfig, load_config


def test_load_minimal_config(tmp_path: Path) -> None:
    payload = """
experiment:
  name: demo
data:
  datasets:
    - name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    cfg = load_config(config_path)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.experiment.name == "demo"
    assert len(cfg.data.datasets) == 1
    assert cfg.data.datasets[0].name == "ds1"
    assert cfg.model.finetune.mode == "full"
    assert cfg.model.attn_implementation is None


def test_normalization(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: SFT
data:
  mix_strategy: INTERLEAVE_OVER
  datasets:
    - name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  scheduler_name: auto
  lr_scheduler_type: LINEAR
model:
  finetune:
    mode: DORA
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg.algorithm.name == "sft"
    assert cfg.data.mix_strategy == "interleave_over"
    assert cfg.train.scheduler_name == "linear"
    assert cfg.model.finetune.mode == "dora"


def test_unknown_key_raises(tmp_path: Path) -> None:
    payload = """
experiment:
  name: demo
  unknown: true
data:
  datasets:
    - name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_path)


def test_algorithm_source_type_mismatch_raises(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: dpo
data:
  datasets:
    - name: ds1
      source_type: jsonl_sft
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_path)


def test_rlhf_numeric_validation_raises(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: ppo
data:
  datasets:
    - name: ds1
      source_type: jsonl_ppo
      train_path: train.jsonl
      val_path: val.jsonl
rlhf:
  ppo:
    cliprange: 1.5
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_path)


def test_load_config_resolves_dataset_refs_from_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        """
datasets:
  registry_ds:
    source_type: jsonl_sft
    train_path: registry/train.jsonl
    val_path: registry/val.jsonl
    weight: 2.0
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  registry_path: {registry_path.name}
  dataset_refs: [registry_ds]
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert len(cfg.data.datasets) == 1
    dataset = cfg.data.datasets[0]
    assert dataset.name == "registry_ds"
    assert dataset.weight == 2.0
    assert dataset.train_paths == [str((tmp_path / "registry" / "train.jsonl").resolve())]
    assert dataset.val_paths == [str((tmp_path / "registry" / "val.jsonl").resolve())]
    assert cfg.data.registry_path == str(registry_path.resolve())
    assert cfg.data.dataset_refs == ["registry_ds"]


def test_load_config_merges_registry_refs_and_inline_datasets(tmp_path: Path) -> None:
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        """
datasets:
  ds_from_registry:
    source_type: jsonl_sft
    train_path: train_a.jsonl
    val_path: val_a.jsonl
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  registry_path: {registry_path.name}
  dataset_refs: [ds_from_registry]
  datasets:
    - name: inline_ds
      train_path: inline_train.jsonl
      val_path: inline_val.jsonl
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert [dataset.name for dataset in cfg.data.datasets] == ["ds_from_registry", "inline_ds"]
    assert cfg.data.datasets[1].train_paths == [str((tmp_path / "inline_train.jsonl").resolve())]
    assert cfg.data.datasets[1].val_paths == [str((tmp_path / "inline_val.jsonl").resolve())]


def test_load_config_raises_for_missing_dataset_ref(tmp_path: Path) -> None:
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text("datasets: {}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  registry_path: {registry_path.name}
  dataset_refs: [missing_ds]
""",
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        load_config(config_path)
