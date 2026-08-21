from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

from PIL import Image


SCRIPT = Path("scripts/tasks/prepare_gt_standard_v5_7.py")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("prepare_gt_standard_v5_7", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_rejects_train_val_overlap_before_audit(tmp_path: Path) -> None:
    dataset_root = tmp_path / "v9"
    (dataset_root / "img").mkdir(parents=True)
    (dataset_root / "gt_standard").mkdir()
    (dataset_root / "train.txt").write_text("00000\n", encoding="utf-8")
    (dataset_root / "val.txt").write_text("00000\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "selection"),
            "--workers",
            "1",
            "--audit-only",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "train/val source overlap" in completed.stderr


def test_line_validator_reports_non_scalar_enums_instead_of_crashing() -> None:
    module = _load_script_module()
    parameters = _line()["parameters"]
    parameters.update(
        {
            "line_type": ["straight"],
            "line_style": ["path"],
            "dash_style": ["solid"],
            "begin_arrow": ["none"],
            "end_arrow": ["triangle"],
            "corner_style": ["sharp"],
        }
    )

    issues = module._validate_line(parameters, width=200, height=200)

    assert any(issue.startswith("line.line_type:") for issue in issues)
    assert any(issue.startswith("line.line_style:") for issue in issues)
    assert any(issue.startswith("line.dash_style:") for issue in issues)
    assert any(issue.startswith("line.begin_arrow:") for issue in issues)
    assert any(issue.startswith("line.end_arrow:") for issue in issues)
    assert any(issue.startswith("line.corner_style:") for issue in issues)


def test_line_validator_rejects_zero_length_and_consecutive_duplicate_points() -> None:
    module = _load_script_module()
    parameters = _line()["parameters"]
    parameters["points"] = [
        [[10, 150], [100, 150], [100, 150], [190, 140]],
        [[100, 150], [100, 150]],
    ]

    issues = module._validate_line(parameters, width=200, height=200)

    assert "line.points[0].consecutive_duplicate" in issues
    assert "line.points[1].degenerate" in issues
    assert "line.points[1].consecutive_duplicate" in issues


def test_v5_8_selection_keeps_rare_shapes_and_caps_dominant_heads() -> None:
    module = _load_script_module()
    shape_strata = {
        "rectangle|uniform:solid|uniform|none|round0|splits0": 1_000,
        "oval|uniform:solid|uniform|none|round0|splits0": 300,
        "step|uniform:solid|uniform|none|round0|splits0": 3,
    }

    quotas, policy = module._shape_quotas_v5_8(
        shape_strata,
        target=100,
        keep_all_threshold=10,
        max_rectangle_fraction=0.20,
    )

    assert quotas["step|uniform:solid|uniform|none|round0|splits0"] == 3
    assert sum(quotas.values()) == 100
    assert policy["selected_by_shape_type"]["rectangle"] <= 20
    assert policy["keep_all_shape_types"] == ["step"]


def test_v5_8_line_selection_keeps_all_multi_branch_and_caps_simple_lines() -> None:
    module = _load_script_module()
    line_strata = {
        "straight|path|segments1|solid|none|none|uniform|none:-|sharp": 300,
        "curved|path|segments1|dash|none|triangle|uniform|none:-|-": 200,
        "straight|path|segments2|solid|none|triangle|uniform|none:-|sharp": 50,
        "straight|path|segments3|dash|none|stealth|uniform|none:-|round": 10,
    }

    quotas, policy = module._line_quotas_v5_8(
        line_strata,
        target=150,
        max_single_segment_fraction=0.50,
    )

    assert quotas[
        "straight|path|segments2|solid|none|triangle|uniform|none:-|sharp"
    ] == 50
    assert quotas[
        "straight|path|segments3|dash|none|stealth|uniform|none:-|round"
    ] == 10
    assert policy["selected_multi_branch_rows"] == 60
    assert policy["selected_single_segment_rows"] == 60
    assert policy["selected_single_segment_fraction"] == 0.5


def test_v5_8_line_point_selection_protects_rare_full_attribute_strata() -> None:
    module = _load_script_module()
    rare_two = "straight|path|segments2|dash|circle|circle|uniform|none:-|round"
    rare_many = "straight|path|segments9|solid|none|none|uniform|none:-|sharp"
    common = "straight|path|segments2|solid|none|triangle|uniform|none:-|sharp"

    quotas, policy = module._line_point_quotas_v5_8(
        {rare_two: 2, rare_many: 1, common: 100},
        target=10,
        keep_all_stratum_threshold=3,
    )

    assert quotas[rare_two] == 2
    assert quotas[rare_many] == 1
    assert sum(quotas.values()) == 10
    assert policy["protected_rows"] == 3
    assert policy["profile"] == "v5.8_multi_branch_rarity_first"
