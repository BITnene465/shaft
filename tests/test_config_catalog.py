from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import load_config


pytestmark = pytest.mark.component


def test_load_config_resolves_catalog_entries(tmp_path: Path) -> None:
    catalog_path = tmp_path / "datasets.yaml"
    catalog_path.write_text(
        """
datasets:
  registry_ds:
    source_type: jsonl_sft
    train_path: registry/train.jsonl
    val_path: registry/val.jsonl
    weight: 2.0
    help: demo
    tags: [base, train]
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
  catalog_path: {catalog_path.name}
  catalog_names: [registry_ds]
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert len(cfg.data.datasets) == 1
    dataset = cfg.data.datasets[0]
    assert dataset.dataset_name == "registry_ds"
    assert dataset.weight == 2.0
    assert dataset.help == "demo"
    assert dataset.tags == ["base", "train"]
    assert dataset.train_paths == [str((tmp_path / "registry" / "train.jsonl").resolve())]
    assert dataset.val_paths == [str((tmp_path / "registry" / "val.jsonl").resolve())]
    assert cfg.data.catalog_path == str(catalog_path.resolve())
    assert cfg.data.catalog_names == ["registry_ds"]


def test_load_config_merges_catalog_entries_and_inline_datasets(tmp_path: Path) -> None:
    catalog_path = tmp_path / "datasets.yaml"
    catalog_path.write_text(
        """
datasets:
  ds_from_catalog:
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
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
  catalog_path: {catalog_path.name}
  catalog_names: [ds_from_catalog]
  datasets:
    - dataset_name: inline_ds
      train_path: inline_train.jsonl
      val_path: inline_val.jsonl
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert [dataset.dataset_name for dataset in cfg.data.datasets] == [
        "ds_from_catalog",
        "inline_ds",
    ]
    assert cfg.data.datasets[1].train_paths == [str((tmp_path / "inline_train.jsonl").resolve())]
    assert cfg.data.datasets[1].val_paths == [str((tmp_path / "inline_val.jsonl").resolve())]


def test_load_config_raises_for_missing_catalog_entry(tmp_path: Path) -> None:
    catalog_path = tmp_path / "datasets.yaml"
    catalog_path.write_text("datasets: {}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
  catalog_path: {catalog_path.name}
  catalog_names: [missing_ds]
""",
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        load_config(config_path)
