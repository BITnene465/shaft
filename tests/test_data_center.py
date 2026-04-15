from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from shaft.config import DataSourceConfig, RuntimeConfig
from shaft.data import DPODataset, SFTDataset, ShaftDataCenter
from shaft.data.transforms import ONLINE_TRANSFORM_REGISTRY

_MARK_DATASET_TRANSFORM = "mark_dataset_for_data_center_tests"


if not ONLINE_TRANSFORM_REGISTRY.has(_MARK_DATASET_TRANSFORM):

    @ONLINE_TRANSFORM_REGISTRY.register(_MARK_DATASET_TRANSFORM)
    def _mark_dataset_transform(sample: dict[str, object]) -> dict[str, object]:
        updated = dict(sample)
        extra = dict(updated.get("extra", {}))
        extra["marked_dataset"] = updated.get("dataset_id")
        updated["extra"] = extra
        return updated


def _write_image(path: Path) -> Path:
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(path)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_data_center_builds_sft_dataset_pair(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "img.png")
    train_a = _write_jsonl(
        tmp_path / "train_a.jsonl",
        [
            {"image_path": str(image), "target_text": "{\"a\":1}", "sample_id": "a1"},
            {"image_path": str(image), "target_text": "{\"a\":2}", "sample_id": "a2"},
        ],
    )
    train_b = _write_jsonl(
        tmp_path / "train_b.jsonl",
        [{"image_path": str(image), "target_text": "{\"b\":1}", "sample_id": "b1"}],
    )
    val_a = _write_jsonl(
        tmp_path / "val_a.jsonl",
        [{"image_path": str(image), "target_text": "{\"va\":1}", "sample_id": "va1"}],
    )
    val_b = _write_jsonl(
        tmp_path / "val_b.jsonl",
        [{"image_path": str(image), "target_text": "{\"vb\":1}", "sample_id": "vb1"}],
    )

    config = RuntimeConfig()
    config.experiment.seed = 7
    config.data.mix_strategy = "concat"
    config.data.shuffle = False
    config.data.datasets = [
        DataSourceConfig(
            name="ds_a",
            train_path=str(train_a),
            val_path=str(val_a),
            online_transforms=[_MARK_DATASET_TRANSFORM],
        ),
        DataSourceConfig(
            name="ds_b",
            train_path=str(train_b),
            val_path=str(val_b),
        ),
    ]

    center = ShaftDataCenter(config.data, seed=config.experiment.seed)
    train_dataset, val_dataset = center.build_dataset_pair(SFTDataset)

    assert len(train_dataset) == 3
    assert len(val_dataset) == 2
    sample_a = train_dataset[0]
    sample_b = train_dataset[2]
    assert sample_a["dataset_id"] == "ds_a"
    assert sample_a["extra"]["marked_dataset"] == "ds_a"
    assert sample_b["dataset_id"] == "ds_b"
    assert "marked_dataset" not in sample_b["extra"]


def test_data_center_builds_dpo_dataset_pair(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "img.png")
    train_path = _write_jsonl(
        tmp_path / "train_dpo.jsonl",
        [
            {
                "image_path": str(image),
                "chosen_text": "{\"ok\":1}",
                "rejected_text": "{\"ok\":0}",
                "sample_id": "d1",
            }
        ],
    )
    val_path = _write_jsonl(
        tmp_path / "val_dpo.jsonl",
        [
            {
                "image_path": str(image),
                "chosen_text": "{\"ok\":2}",
                "rejected_text": "{\"ok\":1}",
                "sample_id": "d2",
            }
        ],
    )

    config = RuntimeConfig()
    config.algorithm.name = "dpo"
    config.data.datasets = [
        DataSourceConfig(
            name="dpo_ds",
            source_type="jsonl_dpo",
            train_path=str(train_path),
            val_path=str(val_path),
        )
    ]

    center = ShaftDataCenter(config.data, seed=config.experiment.seed)
    train_dataset, val_dataset = center.build_dataset_pair(DPODataset)

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    sample = train_dataset[0]
    assert sample["dataset_id"] == "dpo_ds"
    assert sample["chosen_text"] == "{\"ok\":1}"
    assert sample["rejected_text"] == "{\"ok\":0}"
