from __future__ import annotations

import pytest

from shaft.config import DataConfig, DatasetSourceConfig
from shaft.data import ShaftDatasetMeta, build_dataset_metas


def test_dataset_meta_from_config_preserves_metadata_fields() -> None:
    source = DatasetSourceConfig(
        dataset_name="demo_ds",
        source_type="jsonl_sft",
        train_path="train.jsonl",
        val_paths=["val_a.jsonl", "val_b.jsonl"],
        weight=2.5,
        enabled=True,
        use_for_eval=False,
        offline_transforms=["dedup"],
        online_transforms=["identity"],
        help="demo",
        tags=["vision", "json"],
    )
    meta = ShaftDatasetMeta.from_config(source)
    assert meta.dataset_name == "demo_ds"
    assert meta.source_type == "jsonl_sft"
    assert meta.train_paths == ("train.jsonl",)
    assert meta.val_paths == ("val_a.jsonl", "val_b.jsonl")
    assert meta.weight == pytest.approx(2.5)
    assert meta.use_for_eval is False
    assert meta.offline_transforms == ("dedup",)
    assert meta.online_transforms == ("identity",)
    assert meta.help == "demo"
    assert meta.tags == ("vision", "json")


def test_build_dataset_metas_rejects_duplicate_dataset_names() -> None:
    data_config = DataConfig(
        datasets=[
            DatasetSourceConfig(dataset_name="dup", train_path="a.jsonl", val_path="av.jsonl"),
            DatasetSourceConfig(dataset_name="dup", train_path="b.jsonl", val_path="bv.jsonl"),
        ]
    )
    with pytest.raises(ValueError):
        build_dataset_metas(data_config)
