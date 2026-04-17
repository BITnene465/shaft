from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from shaft.config import DatasetSourceConfig, RuntimeConfig
from shaft.data import DPODataset, SFTDataset, ShaftDataCenter, ShaftMixedIndexSampler
from shaft.data.transforms import ONLINE_TRANSFORM_REGISTRY

_MARK_DATASET_TRANSFORM = "mark_dataset_for_data_center_tests"


if not ONLINE_TRANSFORM_REGISTRY.has(_MARK_DATASET_TRANSFORM):

    @ONLINE_TRANSFORM_REGISTRY.register(_MARK_DATASET_TRANSFORM)
    def _mark_dataset_transform(sample: dict[str, object]) -> dict[str, object]:
        updated = dict(sample)
        extra = dict(updated.get("extra", {}))
        extra["marked_dataset"] = updated.get("dataset_name")
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
        DatasetSourceConfig(
            dataset_name="ds_a",
            train_path=str(train_a),
            val_path=str(val_a),
            online_transforms=[_MARK_DATASET_TRANSFORM],
        ),
        DatasetSourceConfig(
            dataset_name="ds_b",
            train_path=str(train_b),
            val_path=str(val_b),
        ),
    ]

    center = ShaftDataCenter(config.data, seed=config.experiment.seed)
    dataset_bundle = center.build_dataset_bundle(SFTDataset)
    train_dataset = dataset_bundle.train_dataset
    val_dataset = dataset_bundle.eval_dataset

    assert len(train_dataset) == 3
    assert len(val_dataset) == 2
    sample_a = train_dataset[0]
    sample_b = train_dataset[2]
    assert sample_a["dataset_name"] == "ds_a"
    assert sample_a["extra"]["marked_dataset"] == "ds_a"
    assert sample_b["dataset_name"] == "ds_b"
    assert "marked_dataset" not in sample_b["extra"]
    assert isinstance(train_dataset.records, dict)
    assert isinstance(dataset_bundle.train_sampler, ShaftMixedIndexSampler)
    assert dataset_bundle.train_sampler.refresh_mode == "static"


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
        DatasetSourceConfig(
            dataset_name="dpo_ds",
            source_type="jsonl_dpo",
            train_path=str(train_path),
            val_path=str(val_path),
        )
    ]

    center = ShaftDataCenter(config.data, seed=config.experiment.seed)
    dataset_bundle = center.build_dataset_bundle(DPODataset)
    train_dataset = dataset_bundle.train_dataset
    val_dataset = dataset_bundle.eval_dataset

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    sample = train_dataset[0]
    assert sample["dataset_name"] == "dpo_ds"
    assert sample["chosen_text"] == "{\"ok\":1}"
    assert sample["rejected_text"] == "{\"ok\":0}"


def test_data_center_skips_val_for_train_only_dataset(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "img.png")
    train_a = _write_jsonl(
        tmp_path / "train_a.jsonl",
        [{"image_path": str(image), "target_text": "{\"a\":1}", "sample_id": "a1"}],
    )
    train_b = _write_jsonl(
        tmp_path / "train_b.jsonl",
        [{"image_path": str(image), "target_text": "{\"b\":1}", "sample_id": "b1"}],
    )
    val_a = _write_jsonl(
        tmp_path / "val_a.jsonl",
        [{"image_path": str(image), "target_text": "{\"va\":1}", "sample_id": "va1"}],
    )

    config = RuntimeConfig()
    config.data.mix_strategy = "concat"
    config.data.shuffle = False
    config.data.datasets = [
        DatasetSourceConfig(
            dataset_name="eval_ds",
            train_path=str(train_a),
            val_path=str(val_a),
            use_for_eval=True,
        ),
        DatasetSourceConfig(
            dataset_name="train_only_ds",
            train_path=str(train_b),
            use_for_eval=False,
        ),
    ]

    center = ShaftDataCenter(config.data, seed=config.experiment.seed)
    dataset_bundle = center.build_dataset_bundle(SFTDataset)
    train_dataset = dataset_bundle.train_dataset
    val_dataset = dataset_bundle.eval_dataset

    assert len(train_dataset) == 2
    assert len(val_dataset) == 1
    assert train_dataset[0]["dataset_name"] == "eval_ds"
    assert train_dataset[1]["dataset_name"] == "train_only_ds"
    assert val_dataset[0]["dataset_name"] == "eval_ds"


def test_data_center_epoch_refresh_builds_train_sampler(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "img.png")
    train_a = _write_jsonl(
        tmp_path / "train_a.jsonl",
        [{"image_path": str(image), "target_text": "{\"a\":1}", "sample_id": f"a{i}"} for i in range(4)],
    )
    train_b = _write_jsonl(
        tmp_path / "train_b.jsonl",
        [{"image_path": str(image), "target_text": "{\"b\":1}", "sample_id": f"b{i}"} for i in range(4)],
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
    config.experiment.seed = 5
    config.data.mix_strategy = "concat"
    config.data.mix_refresh = "epoch_refresh"
    config.data.shuffle = True
    config.data.datasets = [
        DatasetSourceConfig(dataset_name="ds_a", train_path=str(train_a), val_path=str(val_a)),
        DatasetSourceConfig(dataset_name="ds_b", train_path=str(train_b), val_path=str(val_b)),
    ]

    center = ShaftDataCenter(config.data, seed=config.experiment.seed)
    dataset_bundle = center.build_dataset_bundle(SFTDataset)
    train_dataset = dataset_bundle.train_dataset

    assert isinstance(dataset_bundle.train_sampler, ShaftMixedIndexSampler)
    assert dataset_bundle.train_sampler.refresh_mode == "epoch_refresh"
    first = list(dataset_bundle.train_sampler.current_indices)
    first_sample = train_dataset[0]["sample_id"]
    dataset_bundle.train_sampler.set_epoch(1)
    second = list(dataset_bundle.train_sampler.current_indices)
    second_sample = train_dataset[0]["sample_id"]
    assert first != second
    assert first_sample != second_sample
