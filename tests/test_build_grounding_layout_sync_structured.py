from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image
import pytest


def _load_module():
    script_path = Path("scripts/tasks/build_grounding_layout_sync_structured.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "build_grounding_layout_sync_structured",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_source(root: Path, stem: str) -> None:
    (root / "img").mkdir(parents=True, exist_ok=True)
    (root / "gt_standard").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), "white").save(root / "img" / f"{stem}.png")
    payload = {
        "size": [100, 80],
        "layout": [
            {"type": "shape", "bbox": [0, 0, 100, 80]},
            {"type": "shape", "bbox": [10, 10, 50, 40]},
            {"type": "icon", "bbox": [55, 10, 70, 25]},
            {"type": "image", "bbox": [10, 45, 45, 75]},
            {"type": "arrow", "bbox": [50, 50, 90, 70]},
            {"type": "text", "bbox": [0, 0, 20, 10]},
            {"type": "line", "bbox": [4, 4, 4, 20]},
        ],
    }
    (root / "gt_standard" / f"{stem}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_builds_train_only_clean_full_image_rows(tmp_path: Path) -> None:
    module = _load_module()
    source_root = tmp_path / "regulated"
    output_root = tmp_path / "grounding_layout_sync"
    _write_source(source_root, "train_001")
    _write_source(source_root, "val_001")
    split_file = source_root / "custom-train.txt"
    split_file.write_text("train_001\n", encoding="utf-8")
    (source_root / "val.txt").write_text("val_001\n", encoding="utf-8")

    summary = module.build_dataset(
        dataset_root=source_root,
        output_root=output_root,
        split_file=split_file,
        workers=1,
        clean=True,
    )

    rows = [
        json.loads(line)
        for line in (output_root / "structured" / "train.jsonl").read_text().splitlines()
    ]
    assert summary == {
        "rows": 1,
        "empty_rows": 0,
        "label_counts": {"shape": 1, "icon": 1, "image": 1, "line": 1},
        "dropped_counts": {
            "background_shape": 1,
            "unsupported_label:text": 1,
            "invalid_bbox": 1,
        },
        "excluded_val_sources": 1,
    }
    assert len(rows) == 1
    row = rows[0]
    assert row["source_sample_id"] == "train_001"
    assert row["extra"]["view_type"] == "full_image"
    assert row["extra"]["pixel_augmentation"] == {"name": "none"}
    assert [instance["label"] for instance in row["instances"]] == [
        "shape",
        "icon",
        "image",
        "line",
    ]
    referenced_image = (output_root / "structured" / row["image_path"]).resolve()
    assert referenced_image == (source_root / "img" / "train_001.png").resolve()
    assert (output_root / "structured" / "val.jsonl").read_text() == ""
    readme = (output_root / "README.md").read_text()
    assert "no resize, crop, blur, noise, padding" in readme
    assert "custom-train.txt" in readme


def test_rejects_overlapping_input_and_output_before_clean(tmp_path: Path) -> None:
    module = _load_module()
    source_root = tmp_path / "regulated"
    _write_source(source_root, "train_001")
    (source_root / "train.txt").write_text("train_001\n", encoding="utf-8")
    sentinel = source_root / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be disjoint"):
        module.build_dataset(
            dataset_root=source_root,
            output_root=source_root,
            workers=1,
            clean=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep\n"
