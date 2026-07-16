from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from shaft.codec.coordinates import quantize_qwen_bbox, quantize_qwen_point


def _write_selection(
    path: Path,
    *,
    sample_id: str,
    label: str,
    bbox: list[float],
    parameters: dict,
    source_json: str,
    source_image: str,
    instance_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "sample_id": sample_id,
        "image_path": "../legacy-crop.png",
        "instances": [{"label": label, "bbox": [0, 0, 10, 10], "parameters": parameters}],
        "extra": {
            "source_json": source_json,
            "source_image": source_image,
            "source_instance_index": instance_index,
            "source_bbox": bbox,
        },
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _run_builder(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/tasks/build_region_reconstruction_sft.py", *args],
        cwd=Path.cwd(),
        check=check,
        capture_output=True,
        text=True,
    )


def test_builds_full_image_bbox_conditioned_region_tasks(tmp_path: Path) -> None:
    synthetic_root = tmp_path / "synthetic"
    (synthetic_root / "img").mkdir(parents=True)
    (synthetic_root / "gt_standard").mkdir()
    Image.new("RGB", (100, 100), "white").save(synthetic_root / "img/00001.png")
    shape_bbox = [10.0, 20.0, 50.0, 60.0]
    line_bbox = [20.0, 10.0, 80.0, 100.0]
    source_line_bbox = [20.0, 10.0, 80.0, 110.0]
    (synthetic_root / "gt_standard/00001.json").write_text(
        json.dumps(
            {
                "size": [100, 100],
                "layout": [
                    {
                        "type": "shape",
                        "bbox": shape_bbox,
                        "parameters": {
                            "shape_type": "rectangle",
                            "corners": [
                                {"type": "sharp", "point": [10, 20]},
                                {"type": "sharp", "point": [50, 60]},
                            ],
                        },
                    },
                    {
                        "type": "line",
                        "bbox": source_line_bbox,
                        "parameters": {
                            "line_type": "straight",
                            "line_style": "path",
                            "is_single": True,
                            "points": [[[20, 20], [80, 100]]],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    raw_root = tmp_path / "raw"
    (raw_root / "images").mkdir(parents=True)
    (raw_root / "json").mkdir()
    Image.new("RGB", (200, 100), "white").save(raw_root / "images/prod.png")
    selections = tmp_path / "selections"
    _write_selection(
        selections / "shape.jsonl",
        sample_id="00001__shape_0000",
        label="shape",
        bbox=shape_bbox,
        parameters={"shape_type": "rectangle"},
        source_json="gt_standard/00001.json",
        source_image="img/00001.png",
        instance_index=0,
    )
    _write_selection(
        selections / "line.jsonl",
        sample_id="00001__line_0001",
        label="line",
        bbox=line_bbox,
        parameters={"line_type": "straight"},
        source_json="gt_standard/00001.json",
        source_image="img/00001.png",
        instance_index=1,
    )
    image_bbox = [20.0, 10.0, 180.0, 90.0]
    (raw_root / "json/prod.json").write_text(
        json.dumps(
            {
                "image_width": 200,
                "image_height": 100,
                "instances": [
                    {
                        "label": "image",
                        "bbox": image_bbox,
                        "extra": {"parameters": {"image_type": "diagram"}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_selection(
        selections / "image.jsonl",
        sample_id="prod__image_0000",
        label="image",
        bbox=image_bbox,
        parameters={"image_type": "photo"},
        source_json="json/prod.json",
        source_image="images/prod.png",
        instance_index=0,
    )
    output_root = tmp_path / "output"
    _run_builder(
        "--synthetic-root",
        str(synthetic_root),
        "--raw-root",
        str(raw_root),
        "--output-root",
        str(output_root),
        "--shape-selection",
        str(selections / "shape.jsonl"),
        "--line-selection",
        str(selections / "line.jsonl"),
        "--image-selection",
        str(selections / "image.jsonl"),
        "--workers",
        "1",
        "--clean",
    )

    expectations = {
        "shape_region_reconstruction": (shape_bbox, 100, 100, "shape"),
        "line_region_reconstruction": (line_bbox, 100, 100, "line"),
        "image_region_reconstruction": (image_bbox, 200, 100, "image"),
    }
    for task, (bbox, width, height, label) in expectations.items():
        task_root = output_root / task
        structured = json.loads((task_root / "structured/train.jsonl").read_text().strip())
        sft = json.loads((task_root / "sft/train.jsonl").read_text().strip())
        expected_bbox = quantize_qwen_bbox(bbox, width=width, height=height)
        assert structured["image_width"] == width
        assert structured["image_height"] == height
        assert structured["instances"][0]["bbox"] == bbox
        assert structured["extra"]["view_type"] == "full_image_bbox_conditioned"
        assert structured["extra"]["augmentation"] == {"name": "none"}
        assert structured["extra"]["prompt_coordinate_space"] == "qwen_0_999_full_image"
        assert structured["extra"]["target_coordinate_space"] == "qwen_0_999_full_image"
        assert sft["dataset_name"] == task
        assert sft["prompt_args"] == {"bbox_2d": expected_bbox}
        assert sft["system_prompt"] == sft["user_prompt"] == ""
        assert json.loads(sft["target_text"])["type"] == label
        assert (task_root / "sft" / sft["image_path"]).resolve().is_file()
        assert not (task_root / "images").exists()

    shape_target = json.loads(
        (output_root / "shape_region_reconstruction/sft/train.jsonl").read_text()
    )["target_text"]
    assert json.loads(shape_target)["parameters"]["corners"] == [
        {"type": "sharp", "point": [101, 202]},
        {"type": "sharp", "point": [505, 605]},
    ]
    line_target = json.loads(
        (output_root / "line_region_reconstruction/sft/train.jsonl").read_text()
    )["target_text"]
    assert json.loads(line_target)["parameters"]["points"] == [[[202, 202], [807, 999]]]
    image_target = json.loads(
        (output_root / "image_region_reconstruction/sft/train.jsonl").read_text()
    )["target_text"]
    assert json.loads(image_target)["parameters"] == {"image_type": "diagram"}


def test_same_basename_sources_are_built_independently(tmp_path: Path) -> None:
    synthetic_root = tmp_path / "synthetic"
    selections = tmp_path / "line.jsonl"
    bbox = [10.0, 10.0, 90.0, 90.0]
    expected_points: dict[str, list[list[list[int]]]] = {}

    for directory, sample_id, points in (
        ("part_a", "line-a", [[[10, 10], [20, 20]]]),
        ("part_b", "line-b", [[[70, 70], [90, 90]]]),
    ):
        source_dir = synthetic_root / directory
        source_dir.mkdir(parents=True)
        Image.new("RGB", (100, 100), "white").save(source_dir / "shared.png")
        (source_dir / "shared.json").write_text(
            json.dumps(
                {
                    "size": [100, 100],
                    "layout": [
                        {
                            "type": "line",
                            "bbox": bbox,
                            "parameters": {"points": points},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        _write_selection(
            selections,
            sample_id=sample_id,
            label="line",
            bbox=bbox,
            parameters={"line_type": "straight"},
            source_json=f"{directory}/shared.json",
            source_image=f"{directory}/shared.png",
            instance_index=0,
        )
        expected_points[sample_id] = [
            [quantize_qwen_point(point, width=100, height=100) for point in segment]
            for segment in points
        ]

    output_root = tmp_path / "output"
    _run_builder(
        "--synthetic-root",
        str(synthetic_root),
        "--output-root",
        str(output_root),
        "--line-selection",
        str(selections),
        "--tasks",
        "line_region_reconstruction",
        "--workers",
        "1",
        "--clean",
    )

    rows = [
        json.loads(line)
        for line in (output_root / "line_region_reconstruction/sft/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["sample_id"] for row in rows} == set(expected_points)
    for row in rows:
        target = json.loads(row["target_text"])
        assert target["parameters"]["points"] == expected_points[row["sample_id"]]


def test_worker_failure_keeps_previous_outputs_and_rejects_invalid_segments(
    tmp_path: Path,
) -> None:
    synthetic_root = tmp_path / "synthetic"
    synthetic_root.mkdir()
    Image.new("RGB", (100, 100), "white").save(synthetic_root / "line.png")
    bbox = [10.0, 10.0, 90.0, 90.0]
    (synthetic_root / "line.json").write_text(
        json.dumps(
            {
                "size": [100, 100],
                "layout": [
                    {
                        "type": "line",
                        "bbox": bbox,
                        "parameters": {"points": [[[10, 10], [90, 90]], {"invalid": "segment"}]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    selection_path = tmp_path / "line.jsonl"
    _write_selection(
        selection_path,
        sample_id="invalid-line",
        label="line",
        bbox=bbox,
        parameters={"line_type": "straight"},
        source_json="line.json",
        source_image="line.png",
        instance_index=0,
    )

    output_root = tmp_path / "output"
    task_root = output_root / "line_region_reconstruction"
    for kind in ("structured", "sft"):
        (task_root / kind).mkdir(parents=True, exist_ok=True)
        (task_root / kind / "train.jsonl").write_text("sentinel\n", encoding="utf-8")

    result = _run_builder(
        "--synthetic-root",
        str(synthetic_root),
        "--output-root",
        str(output_root),
        "--line-selection",
        str(selection_path),
        "--tasks",
        "line_region_reconstruction",
        "--workers",
        "1",
        "--clean",
        check=False,
    )

    assert result.returncode != 0
    assert "Invalid line point segment" in result.stderr
    assert (task_root / "structured/train.jsonl").read_text(encoding="utf-8") == "sentinel\n"
    assert (task_root / "sft/train.jsonl").read_text(encoding="utf-8") == "sentinel\n"
    assert not list(output_root.glob(".line_region_reconstruction.staging.*"))
