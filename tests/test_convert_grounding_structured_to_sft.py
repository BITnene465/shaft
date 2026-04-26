from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = Path("scripts/tasks/convert_grounding_structured_to_sft.py").resolve()
    spec = importlib.util.spec_from_file_location("convert_grounding_structured_to_sft", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_row_major_module():
    script_path = Path("scripts/tasks/convert_grounding_structured_to_sft_row_major.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "convert_grounding_structured_to_sft_row_major",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_convert_structured_jsonl_to_sft_sorts_by_log_bucket_then_coordinates(tmp_path):
    module = _load_module()

    prompt_path = tmp_path / "prompt.yaml"
    prompt_path.write_text(
        """
metadata:
  id: test.prompt
prompt:
  system_prompt: ""
  user_prompt: Locate arrows.
""".strip()
        + "\n",
        encoding="utf-8",
    )

    image_dir = tmp_path / "images" / "train"
    image_dir.mkdir(parents=True)
    (image_dir / "sample.png").write_bytes(b"fake")

    structured_dir = tmp_path / "structured"
    structured_dir.mkdir()
    structured_path = structured_dir / "train.jsonl"
    row = {
        "sample_id": "sample_001",
        "source_sample_id": "sample_001",
        "source_type": "full_image",
        "image_path": "../images/train/sample.png",
        "image_width": 1000,
        "image_height": 1000,
        "instances": [
            {"label": "double_arrow", "bbox": [300, 200, 500, 400]},
            {"label": "single_arrow", "bbox": [50, 50, 450, 450]},
            {"label": "single_arrow", "bbox": [50, 100, 240, 310]},
        ],
        "extra": {"view_type": "identity"},
    }
    structured_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    output_path = tmp_path / "sft" / "train.jsonl"
    count = module.convert_structured_jsonl_to_sft(
        structured_path=structured_path,
        output_path=output_path,
        dataset_name="grounding_arrow_train",
        prompt_config_path=prompt_path,
        num_bins=1000,
        bucket_base=2.0,
    )

    assert count == 1
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    converted = json.loads(lines[0])
    assert converted["dataset_name"] == "grounding_arrow_train"
    assert converted["image_path"] == "../images/train/sample.png"
    assert converted["system_prompt"] == ""
    assert converted["user_prompt"] == "Locate arrows."

    target = json.loads(converted["target_text"])
    assert target == [
        {"label": "single_arrow", "bbox_2d": [50, 50, 450, 450]},
        {"label": "single_arrow", "bbox_2d": [50, 100, 240, 310]},
        {"label": "double_arrow", "bbox_2d": [300, 200, 500, 400]},
    ]
    assert converted["extra"]["prompt_id"] == "test.prompt"
    assert converted["extra"]["sort_policy"] == {
        "type": "log_area_bucket",
        "bucket_base": 2.0,
        "within_bucket": "bbox_coordinate",
    }


def test_convert_structured_jsonl_to_sft_rebases_image_path_for_sibling_output(tmp_path):
    module = _load_module()

    prompt_path = tmp_path / "prompt.yaml"
    prompt_path.write_text(
        """
metadata:
  id: test.prompt
prompt:
  system_prompt: ""
  user_prompt: Locate arrows.
""".strip()
        + "\n",
        encoding="utf-8",
    )

    root_dir = tmp_path / "dataset"
    images_dir = root_dir / "images" / "train"
    images_dir.mkdir(parents=True)
    (images_dir / "sample.png").write_bytes(b"fake")

    structured_dir = root_dir / "structured"
    structured_dir.mkdir()
    structured_path = structured_dir / "train.jsonl"
    row = {
        "sample_id": "sample_001",
        "source_sample_id": "sample_001",
        "source_type": "full_image",
        "image_path": "images/train/sample.png",
        "image_width": 744,
        "image_height": 992,
        "instances": [],
        "extra": {"view_type": "identity"},
    }
    structured_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    output_path = root_dir / "sft" / "train.jsonl"
    module.convert_structured_jsonl_to_sft(
        structured_path=structured_path,
        output_path=output_path,
        dataset_name="grounding_arrow_syn",
        prompt_config_path=prompt_path,
        num_bins=1000,
        bucket_base=2.0,
    )

    converted = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert converted["image_path"] == "../images/train/sample.png"
    assert converted["target_text"] == "[]"


def test_convert_structured_jsonl_to_sft_row_major_groups_rows_by_center_then_sorts_left_to_right(tmp_path):
    module = _load_row_major_module()

    prompt_path = tmp_path / "prompt.yaml"
    prompt_path.write_text(
        """
metadata:
  id: test.prompt
prompt:
  system_prompt: ""
  user_prompt: Locate objects.
""".strip()
        + "\n",
        encoding="utf-8",
    )

    image_dir = tmp_path / "images" / "train"
    image_dir.mkdir(parents=True)
    (image_dir / "sample.png").write_bytes(b"fake")

    structured_dir = tmp_path / "structured"
    structured_dir.mkdir()
    structured_path = structured_dir / "train.jsonl"
    row = {
        "sample_id": "sample_001",
        "source_sample_id": "sample_001",
        "source_type": "full_image",
        "image_path": "../images/train/sample.png",
        "image_width": 1000,
        "image_height": 1000,
        "instances": [
            {"label": "icon", "bbox": [300, 80, 420, 220]},
            {"label": "icon", "bbox": [120, 100, 240, 200]},
            {"label": "icon", "bbox": [80, 500, 220, 700]},
        ],
        "extra": {"view_type": "identity"},
    }
    structured_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    output_path = tmp_path / "sft" / "train.jsonl"
    count = module.convert_structured_jsonl_to_sft_row_major(
        structured_path=structured_path,
        output_path=output_path,
        dataset_name="grounding_layout",
        prompt_config_path=prompt_path,
        num_bins=1000,
    )

    assert count == 1
    converted = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert converted["dataset_name"] == "grounding_layout"
    target = json.loads(converted["target_text"])
    assert target == [
        {"label": "icon", "bbox_2d": [120, 100, 240, 200]},
        {"label": "icon", "bbox_2d": [300, 80, 420, 220]},
        {"label": "icon", "bbox_2d": [80, 500, 220, 700]},
    ]
    sort_policy = converted["extra"]["sort_policy"]
    assert sort_policy["type"] == "row_bucket_center_v2"
    assert sort_policy["coordinate_space"] == "bbox_2d"
    assert sort_policy["row_anchor"] == "y_center"
    assert sort_policy["row_bucket_size_2d"] == 70
    assert sort_policy["order"] == ["row_bucket", "x1", "y1", "y2", "x2", "label"]
    assert sort_policy["tie_break"] == "source_bbox_float"
