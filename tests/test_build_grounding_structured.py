from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    script_path = Path("scripts/tasks/build_grounding_structured.py").resolve()
    spec = importlib.util.spec_from_file_location("build_grounding_structured", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_read_split_accepts_vlm_json_manifest(tmp_path: Path) -> None:
    module = _load_module()
    split_path = tmp_path / "vlm.test.json"
    split_path.write_text(
        json.dumps(
            {
                "schema": "vlm_data.test_split.v2",
                "name": "vlm.test",
                "task": "vlm",
                "split": "test",
                "items": [
                    {"id": "sample_a", "image_path": "images/sample_a.png"},
                    {"image_path": "images/sample_b.jpg"},
                    {"json_path": "part1/json/custom.json", "image_path": "images/ignored.png"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert module._read_split(split_path) == [
        "json/sample_a.json",
        "json/sample_b.json",
        "part1/json/custom.json",
    ]


def test_build_grounding_layout_supports_unified_raw_and_new_augments(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "json").mkdir(parents=True)
    (raw_root / "images").mkdir(parents=True)
    image_path = raw_root / "images" / "sample.png"

    from PIL import Image

    Image.new("RGB", (320, 240), "white").save(image_path)
    payload = {
        "image_width": 320,
        "image_height": 240,
        "instances": [
            {"label": "shape", "bbox": [20, 20, 70, 70]},
            {"label": "icon", "bbox": [90, 20, 120, 60]},
            {"label": "image", "bbox": [150, 30, 220, 100]},
            {"label": "arrow", "bbox": [30, 120, 160, 130]},
            {"label": "line", "bbox": [180, 150, 270, 160]},
        ],
        "background": True,
    }
    (raw_root / "json" / "sample.json").write_text(json.dumps(payload), encoding="utf-8")
    train_split = tmp_path / "train.txt"
    val_split = tmp_path / "val.txt"
    train_split.write_text("json/sample.json\n", encoding="utf-8")
    val_split.write_text("", encoding="utf-8")

    output_root = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "scripts/tasks/build_grounding_structured.py",
            "--raw-root",
            str(raw_root),
            "--output-root",
            str(output_root),
            "--train-split",
            str(train_split),
            "--val-split",
            str(val_split),
            "--task",
            "grounding_layout",
            "--workers",
            "1",
            "--candidate-count",
            "8",
            "--negative-candidate-count",
            "8",
            "--density-crop-ratio",
            "0.3",
            "--negative-ratio",
            "0.0",
            "--blur-ratio",
            "1.0",
            "--padded-full-ratio",
            "1.0",
            "--clean",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [
        json.loads(line)
        for line in (output_root / "grounding_layout" / "structured" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    view_types = {row["extra"]["view_type"] for row in rows}
    assert "full_image" in view_types
    assert "random_padded_full" in view_types
    assert view_types & {"blur_full", "blur_crop"}
    labels = {
        instance["label"]
        for row in rows
        for instance in row["instances"]
    }
    assert {"shape", "icon", "image", "line"}.issubset(labels)


def test_negative_crop_only_requires_gt_disjoint() -> None:
    module = _load_module()
    rng = module.random.Random(123)
    instances = [
        module.SourceInstance(index=0, label="shape", bbox=(80, 80, 120, 120)),
    ]

    crop = module._select_negative_crop(
        instances,
        image_width=400,
        image_height=300,
        rng=rng,
        candidate_count=64,
    )
    assert crop is not None
    assert not any(module._bbox_intersects(instance.bbox, crop) for instance in instances)
