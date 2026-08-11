from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


SCRIPT = Path("scripts/tasks/prepare_gt_standard_v5_7.py")


def _sharp(point: list[int]) -> dict:
    return {"type": "sharp", "point": point}


def _card() -> dict:
    return {
        "type": "shape",
        "bbox": [20, 20, 180, 120],
        "parameters": {
            "shape_type": "card",
            "border": {"type": "uniform", "style": "solid", "color": "#111111"},
            "fill": [
                {"type": "uniform", "color": "#FFFFFF"},
                {"type": "complex"},
            ],
            "corners": [
                _sharp([20, 20]),
                _sharp([180, 20]),
                _sharp([180, 120]),
                _sharp([20, 120]),
            ],
            "splits": [
                {
                    "type": "uniform",
                    "style": "dash",
                    "color": "#222222",
                    "split_corners": [_sharp([20, 60]), _sharp([180, 60])],
                }
            ],
            "effect": {"type": "none"},
        },
    }


def _line() -> dict:
    return {
        "type": "line",
        "bbox": [10, 130, 190, 180],
        "parameters": {
            "line_type": "straight",
            "line_style": "path",
            "is_single": False,
            "points": [
                [[10, 150], [100, 150], [190, 140]],
                [[100, 150], [150, 180]],
            ],
            "dash_style": "solid",
            "begin_arrow": "none",
            "end_arrow": "triangle",
            "fill": {"type": "uniform", "color": "#333333"},
            "border": {"type": "none"},
            "corner_style": "sharp",
        },
    }


def _oval_image() -> dict:
    return {
        "type": "image",
        "bbox": [20, 20, 80, 80],
        "parameters": {
            "image_type": "N/A",
            "clip_shape": "oval",
            "corners": [],
            "border": {"type": "none"},
            "effect": {"type": "exist"},
        },
    }


def test_audits_and_builds_deterministic_v5_7_selection_manifests(tmp_path: Path) -> None:
    dataset_root = tmp_path / "v9"
    (dataset_root / "img").mkdir(parents=True)
    (dataset_root / "gt_standard").mkdir()
    Image.new("RGB", (200, 200), "white").save(dataset_root / "img/00000.png")
    payload = {
        "size": [200, 200],
        "background": "none",
        "layout": [
            _card(),
            _line(),
            _oval_image(),
            {"type": "icon", "bbox": [1, 1, 10, 10]},
        ],
    }
    (dataset_root / "gt_standard/00000.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (dataset_root / "train.txt").write_text("00000\n", encoding="utf-8")
    (dataset_root / "val.txt").write_text("", encoding="utf-8")
    output_root = tmp_path / "selection"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--workers",
            "1",
            "--shape-target",
            "10",
            "--line-target",
            "10",
            "--line-points-target",
            "10",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    printed = json.loads(completed.stdout)
    assert printed["audit"]["fatal_rows"] == 0
    assert printed["audit"]["instance_issue_counts"] == {}
    assert printed["selection"]["rows"] == {"shape": 1, "line": 1, "line_points": 1}
    assert (output_root / "clean_train.txt").read_text(encoding="utf-8") == "00000\n"
    shape = json.loads((output_root / "shape/train.jsonl").read_text(encoding="utf-8"))
    line = json.loads((output_root / "line/train.jsonl").read_text(encoding="utf-8"))
    points = json.loads((output_root / "line_points/train.jsonl").read_text(encoding="utf-8"))
    assert shape["extra"]["source_instance_index"] == 0
    assert line["extra"]["source_instance_index"] == 1
    assert points == line
