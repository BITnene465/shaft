from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = Path("scripts/tasks/build_sft_from_structured.py").resolve()
    spec = importlib.util.spec_from_file_location("build_sft_from_structured", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_output_row_uses_qwen_grounding_schema(tmp_path: Path) -> None:
    module = _load_module()

    task_root = tmp_path / "grounding_layout"
    image_dir = task_root / "images" / "train"
    image_dir.mkdir(parents=True)
    (image_dir / "sample.png").write_bytes(b"fake")

    structured_path = task_root / "structured" / "train.jsonl"
    output_path = task_root / "sft" / "train.jsonl"
    config = module.ConvertConfig(
        task=module.TaskSpec("grounding_layout", "grounding", "unused.yaml"),
        prompt=module.PromptConfig(
            prompt_id="unit.qwen_bbox",
            system_prompt="Return JSON.",
            user_prompt="",
            metadata={"output_schema": "qwen_bbox_2d_list"},
        ),
        structured_path=structured_path,
        output_path=output_path,
        num_bins=1000,
    )
    record = {
        "sample_id": "sample_001",
        "source_sample_id": "raw_001",
        "source_type": "full_image",
        "image_path": "../images/train/sample.png",
        "image_width": 1000,
        "image_height": 1000,
        "instances": [
            {"label": "line", "bbox": [0, 100, 999, 120]},
            {"label": "shape", "bbox": [100, 100, 300, 300]},
            {"label": "image", "bbox": [500, 500, 750, 750]},
        ],
        "extra": {"split": "train"},
    }

    row = module._build_output_row((1, json.dumps(record), config))

    assert row["dataset_name"] == "grounding_layout"
    assert row["image_path"] == "../images/train/sample.png"
    assert row["system_prompt"] == ""
    assert row["user_prompt"] == ""
    assert json.loads(row["target_text"]) == [
        {"bbox_2d": [0, 100, 999, 120], "label": "line"},
        {"bbox_2d": [100, 100, 300, 300], "label": "shape"},
        {"bbox_2d": [500, 500, 750, 750], "label": "image"},
    ]
    assert row["extra"]["output_schema"] == "qwen_bbox_2d_list"
    assert row["extra"]["sort_policy"] == {
        "type": "row_bucket_top_area_v3",
        "coordinate_space": "bbox_2d",
        "row_anchor": "y1",
        "row_bucket_size_2d": 20,
        "order": ("row_bucket", "x1", "y1", "-area", "x2", "y2", "label"),
        "tie_break": "qwen_bbox_2d",
    }


def test_grounding_canonical_order_uses_y1_bucket_and_area_tie() -> None:
    module = _load_module()

    target, sort_policy = module._build_grounding_target(
        [
            {"label": "shape", "bbox": [20, 100, 220, 260]},
            {"label": "icon", "bbox": [20, 100, 120, 180]},
            {"label": "line", "bbox": [10, 110, 900, 118]},
            {"label": "image", "bbox": [5, 121, 50, 160]},
            {"label": "shape", "bbox": [4, 80, 40, 95]},
        ],
        image_width=1000,
        image_height=1000,
        num_bins=1000,
    )

    assert sort_policy["type"] == "row_bucket_top_area_v3"
    assert sort_policy["row_bucket_size_2d"] == 20
    assert target == [
        {"bbox_2d": [4, 80, 40, 95], "label": "shape"},
        {"bbox_2d": [10, 110, 900, 118], "label": "line"},
        {"bbox_2d": [20, 100, 220, 260], "label": "shape"},
        {"bbox_2d": [20, 100, 120, 180], "label": "icon"},
        {"bbox_2d": [5, 121, 50, 160], "label": "image"},
    ]


def test_build_output_row_uses_qwen_point_line_schema(tmp_path: Path) -> None:
    module = _load_module()

    task_root = tmp_path / "point_line"
    image_dir = task_root / "images" / "train"
    image_dir.mkdir(parents=True)
    (image_dir / "line.png").write_bytes(b"fake")

    structured_path = task_root / "structured" / "train.jsonl"
    output_path = task_root / "sft" / "train.jsonl"
    config = module.ConvertConfig(
        task=module.TaskSpec("point_line", "point_line", "unused.yaml"),
        prompt=module.PromptConfig(
            prompt_id="unit.line_points",
            system_prompt="Return JSON.",
            user_prompt="",
            metadata={"output_schema": "qwen_line_points_segments_object"},
        ),
        structured_path=structured_path,
        output_path=output_path,
        num_bins=1000,
    )
    record = {
        "sample_id": "line_001",
        "source_sample_id": "raw_line_001",
        "source_type": "crop",
        "image_path": "../images/train/line.png",
        "image_width": 1000,
        "image_height": 500,
        "instances": [
            {
                "label": "line",
                "bbox": [228, 246, 810, 246],
                "linestrip": [
                    [[228, 246], [400, 246], [400, 300]],
                    [[400, 246], [810, 246]],
                ],
            }
        ],
        "extra": {"split": "train"},
    }

    row = module._build_output_row((1, json.dumps(record), config))

    assert row["dataset_name"] == "point_line"
    assert json.loads(row["target_text"]) == {
        "type": "line",
        "parameters": {
            "is_single": False,
            "points": [
                [[228, 492], [400, 492], [400, 601]],
                [[400, 492], [810, 492]],
            ],
        },
    }
    assert row["extra"]["target_policy"] == {
        "type": "line_points_segments",
        "coordinate_space": "parameters.points",
        "order": "source_linestrip_segment_order",
    }
