from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image


def _load_module():
    script_path = Path("scripts/tasks/build_point_line_structured.py").resolve()
    spec = importlib.util.spec_from_file_location("build_point_line_structured", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_point_line_builds_seeded_padding_crops_for_arrow_and_line(tmp_path: Path) -> None:
    module = _load_module()

    raw_root = tmp_path / "raw"
    (raw_root / "json").mkdir(parents=True)
    (raw_root / "images").mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(raw_root / "images" / "sample.png")
    (raw_root / "json" / "sample.json").write_text(
        json.dumps(
            {
                "image_path": "images/sample.png",
                "image_width": 100,
                "image_height": 80,
                "annotation": {"layers": ["layout", "arrow"]},
                "instances": [
                    {
                        "label": "arrow",
                        "bbox": [10, 20, 60, 25],
                        "linestrip": [[10, 22], [60, 22]],
                    },
                    {
                        "label": "line",
                        "bbox": [30, 40, 35, 70],
                        "linestrip": [[32, 40], [32, 70]],
                    },
                    {"label": "shape", "bbox": [1, 2, 3, 4]},
                ],
            }
        ),
        encoding="utf-8",
    )
    split_path = raw_root / "splits" / "train.txt"
    split_path.parent.mkdir()
    split_path.write_text("json/sample.json\n", encoding="utf-8")

    result = module.build_split(
        split_path=split_path,
        output_path=tmp_path / "point_line" / "structured" / "train.jsonl",
        config=module.BuildConfig(
            raw_root=raw_root,
            split="train",
            image_output_dir=tmp_path / "point_line" / "images" / "train",
            padding_min=0.2,
            padding_max=0.5,
            val_padding=0.35,
            seed=42,
            min_crop_size=4,
        ),
        workers=1,
    )

    assert len(result.rows) == 2
    assert result.skipped_count == 0
    assert {row["instances"][0]["label"] for row in result.rows} == {"line"}
    assert {row["extra"]["source_label"] for row in result.rows} == {"arrow", "line"}
    assert {row["extra"]["augmentation"]["name"] for row in result.rows} == {
        "bbox_padding_crop"
    }
    for row in result.rows:
        assert 0.2 <= row["extra"]["padding_ratio"] <= 0.5
    for row in result.rows:
        image_path = tmp_path / "point_line" / "structured" / row["image_path"]
        assert image_path.resolve().exists()
