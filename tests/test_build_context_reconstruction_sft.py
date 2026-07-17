from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image
import pytest

from shaft.codec.coordinates import quantize_qwen_bbox, quantize_qwen_point


SCRIPT = Path("scripts/tasks/build_context_reconstruction_sft.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("build_context_reconstruction_sft", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_prompt(path: Path, *, task: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
metadata:
  id: shaft.{task}.prompt_pool.v5.3
  version: v5.3
  task: {task}
  target_labels: [{label}]
arguments:
  proposal_bbox_2d:
    type: bbox_2d_0_999
prompts:
  - id: main
    system_prompt: JSON only.
    user_prompt_template: "Target {{{{ proposal_bbox_2d | json }}}} in contextual crop."
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_selection(
    path: Path,
    *,
    sample_id: str,
    label: str,
    bbox: list[float],
    source_json: str,
    source_image: str,
    instance_index: int,
    parameters: dict | None = None,
    weak_label: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "sample_id": sample_id,
        "image_path": "../full-image.png",
        "image_width": 1,
        "image_height": 1,
        "instances": [{"label": label, "bbox": bbox, "parameters": parameters or {}}],
        "extra": {
            "source_json": source_json,
            "source_image": source_image,
            "source_instance_index": instance_index,
            "source_bbox": bbox,
        },
    }
    if weak_label is not None:
        row["extra"]["weak_label"] = weak_label
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _run_builder(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=Path.cwd(),
        check=check,
        capture_output=True,
        text=True,
    )


def _read_one(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    return rows[0]


def test_context_view_is_deterministic_asymmetric_and_contains_gt() -> None:
    module = _load_module()
    first = module._sample_context_view(
        source_bbox=(80.0, 90.0, 180.0, 210.0),
        image_width=400,
        image_height=320,
        task="shape_context_reconstruction",
        sample_id="shape-1",
        seed=42,
        max_aspect_ratio=60.0,
    )
    second = module._sample_context_view(
        source_bbox=(80.0, 90.0, 180.0, 210.0),
        image_width=400,
        image_height=320,
        task="shape_context_reconstruction",
        sample_id="shape-1",
        seed=42,
        max_aspect_ratio=60.0,
    )

    assert first == second
    assert len(set(first.requested_padding_ratios)) > 1
    left, top, right, bottom = first.crop_box
    assert left <= 80.0 < 180.0 <= right
    assert top <= 90.0 < 210.0 <= bottom
    assert 0 <= left < right <= 400
    assert 0 <= top < bottom <= 320
    assert first.gt_coverage == 1.0
    assert 0.0 < first.proposal_iou <= 1.0
    width, height = right - left, bottom - top
    assert max(width / height, height / width) <= 60.0


def test_zero_limit_is_rejected_before_output_creation(tmp_path: Path) -> None:
    completed = _run_builder(
        "--output-root",
        str(tmp_path / "output"),
        "--limit",
        "0",
        check=False,
    )

    assert completed.returncode == 2
    assert "limit must be positive" in completed.stderr
    assert not (tmp_path / "output").exists()


def test_synthetic_realism_plan_is_deterministic_stacked_and_size_preserving() -> None:
    module = _load_module()
    plans = [
        module._sample_synthetic_pixel_augmentation(
            task="line_context_reconstruction",
            sample_id=f"line-{index}",
            seed=42,
            target_short_span=320,
            image_width=96,
            image_height=64,
        )
        for index in range(64)
    ]

    assert all(plan["profile"] == "synthetic_realism_v1" for plan in plans)
    assert all(1 <= len(plan["operations"]) <= 3 for plan in plans)
    assert any(len(plan["operations"]) > 1 for plan in plans)
    assert plans[0] == module._sample_synthetic_pixel_augmentation(
        task="line_context_reconstruction",
        sample_id="line-0",
        seed=42,
        target_short_span=320,
        image_width=96,
        image_height=64,
    )

    tiny = module._sample_synthetic_pixel_augmentation(
        task="line_context_reconstruction",
        sample_id="tiny-line",
        seed=42,
        target_short_span=40,
        image_width=96,
        image_height=64,
    )
    assert tiny["severity"] == "mild"
    assert len(tiny["operations"]) == 1

    image = Image.new("RGB", (96, 64))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = ((x * 17) % 256, (y * 29) % 256, ((x + y) * 13) % 256)
    augmented = module._apply_synthetic_pixel_augmentation(image, plans[0])
    try:
        assert augmented.size == image.size
        assert augmented.tobytes() != image.tobytes()
    finally:
        augmented.close()
        image.close()


def test_context_manifest_can_recover_base_selection_without_reusing_target(tmp_path: Path) -> None:
    module = _load_module()
    selection_path = tmp_path / "context.jsonl"
    selection_path.write_text(
        json.dumps(
            {
                "sample_id": "source__shape_0003__context_00",
                "instances": [
                    {
                        "label": "shape",
                        "bbox": [1, 2, 3, 4],
                        "parameters": {"shape_type": "derived-must-not-be-read"},
                    }
                ],
                "extra": {
                    "view_type": "context_crop_bbox_conditioned",
                    "source_json": "gt_standard/source.json",
                    "source_image": "img/source.png",
                    "source_instance_index": 3,
                    "source_bbox": [10, 20, 80, 90],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = module.TaskSpec(
        name="shape_context_reconstruction",
        label="shape",
        selection_path=selection_path,
        source_root=tmp_path,
        prompt_path=tmp_path / "prompt.yaml",
        source_kind="synthetic",
    )

    selections, excluded = module._load_selections(spec, excluded_ids=set(), limit=None)

    assert excluded == 0
    assert len(selections) == 1
    assert selections[0].sample_id == "source__shape_0003"
    assert selections[0].source_bbox == (10.0, 20.0, 80.0, 90.0)


def test_real_weak_selection_requires_api_provenance(tmp_path: Path) -> None:
    module = _load_module()
    selection_path = tmp_path / "weak.jsonl"
    _write_selection(
        selection_path,
        sample_id="source__shape_0000",
        label="shape",
        bbox=[10.0, 20.0, 80.0, 90.0],
        source_json="json/source.json",
        source_image="images/source.png",
        instance_index=0,
        parameters={
            "shape_type": "rectangle",
            "border": {"type": "none"},
            "fill": {"type": "none"},
            "effect": {"type": "none"},
        },
    )
    spec = module.TaskSpec(
        name="shape_context_attributes",
        label="shape",
        selection_path=selection_path,
        source_root=tmp_path,
        prompt_path=tmp_path / "prompt.yaml",
        source_kind="real_weak",
    )

    with pytest.raises(ValueError, match="Missing real weak provenance"):
        module._load_selections(spec, excluded_ids=set(), limit=None)


def test_builds_context_crop_tasks_with_crop_local_prompt_and_targets(tmp_path: Path) -> None:
    synthetic_root = tmp_path / "synthetic"
    (synthetic_root / "img").mkdir(parents=True)
    (synthetic_root / "gt_standard").mkdir()
    Image.new("RGB", (200, 160), "white").save(synthetic_root / "img/00001.png")
    shape_bbox = [60.0, 30.0, 130.0, 100.0]
    line_bbox = [20.0, 110.0, 180.0, 140.0]
    shape_parameters = {
        "shape_type": "rectangle",
        "corners": [
            {"type": "sharp", "point": [60, 30]},
            {"type": "sharp", "point": [130, 30]},
            {"type": "sharp", "point": [130, 100]},
            {"type": "sharp", "point": [60, 100]},
        ],
    }
    line_parameters = {
        "line_type": "straight",
        "line_style": "path",
        "is_single": False,
        "points": [
            [[20, 125], [100, 125], [180, 115]],
            [[100, 125], [150, 140]],
        ],
        "begin_arrow": "none",
        "end_arrow": "triangle",
        "dash_style": "solid",
        "fill_color": "#112233",
    }
    (synthetic_root / "gt_standard/00001.json").write_text(
        json.dumps(
            {
                "size": [200, 160],
                "background": "none",
                "layout": [
                    {"type": "shape", "bbox": shape_bbox, "parameters": shape_parameters},
                    {"type": "line", "bbox": line_bbox, "parameters": line_parameters},
                ],
            }
        ),
        encoding="utf-8",
    )

    raw_root = tmp_path / "raw"
    (raw_root / "images").mkdir(parents=True)
    (raw_root / "json").mkdir()
    Image.new("RGB", (240, 120), "white").save(raw_root / "images/prod.png")
    image_bbox = [40.0, 20.0, 200.0, 100.0]
    (raw_root / "json/prod.json").write_text(
        json.dumps(
            {
                "image_width": 240,
                "image_height": 120,
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
    Image.new("RGB", (240, 120), "white").save(raw_root / "images/heldout.png")
    (raw_root / "json/heldout.json").write_text(
        json.dumps(
            {
                "image_width": 240,
                "image_height": 120,
                "instances": [
                    {
                        "label": "image",
                        "bbox": image_bbox,
                        "extra": {"parameters": {"image_type": "photo"}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    selections = tmp_path / "selections"
    _write_selection(
        selections / "shape.jsonl",
        sample_id="00001__shape_0000",
        label="shape",
        bbox=shape_bbox,
        source_json="gt_standard/00001.json",
        source_image="img/00001.png",
        instance_index=0,
    )
    _write_selection(
        selections / "line.jsonl",
        sample_id="00001__line_0001",
        label="line",
        bbox=line_bbox,
        source_json="gt_standard/00001.json",
        source_image="img/00001.png",
        instance_index=1,
    )
    _write_selection(
        selections / "image.jsonl",
        sample_id="prod__image_0000",
        label="image",
        bbox=image_bbox,
        source_json="json/prod.json",
        source_image="images/prod.png",
        instance_index=0,
    )
    _write_selection(
        selections / "image.jsonl",
        sample_id="heldout__image_0000",
        label="image",
        bbox=image_bbox,
        source_json="json/heldout.json",
        source_image="images/heldout.png",
        instance_index=0,
    )
    prompt_root = tmp_path / "prompts"
    for name, label in (("shape", "shape"), ("line", "line"), ("image", "image")):
        _write_prompt(
            prompt_root / f"{name}.yaml",
            task=f"{name}_context_reconstruction",
            label=label,
        )
    excluded = tmp_path / "test.json"
    excluded.write_text(json.dumps({"items": [{"id": "heldout"}]}), encoding="utf-8")

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
        "--shape-prompt-pool",
        str(prompt_root / "shape.yaml"),
        "--line-prompt-pool",
        str(prompt_root / "line.yaml"),
        "--image-prompt-pool",
        str(prompt_root / "image.yaml"),
        "--exclude-manifest",
        str(excluded),
        "--workers",
        "1",
        "--clean",
    )

    targets = {
        "shape_context_reconstruction": ("shape", shape_parameters),
        "line_context_reconstruction": ("line", line_parameters),
        "image_context_reconstruction": ("image", {"image_type": "diagram"}),
    }
    for task, (label, _) in targets.items():
        task_root = output_root / task
        assert task_root.stat().st_mode & 0o777 == 0o755
        structured = _read_one(task_root / "structured/train.jsonl")
        sft = _read_one(task_root / "sft/train.jsonl")
        assert structured["sample_id"].endswith("__context_00")
        assert sft["sample_id"] == structured["sample_id"]
        assert sft["dataset_name"] == task
        assert sft["system_prompt"] == sft["user_prompt"] == ""
        assert set(sft["prompt_args"]) == {"proposal_bbox_2d"}
        assert json.loads(sft["target_text"])["type"] == label
        assert structured["extra"]["view_type"] == "context_crop_bbox_conditioned"
        assert structured["extra"]["prompt_coordinate_space"] == "qwen_0_999_context_crop"
        assert structured["extra"]["target_coordinate_space"] == "qwen_0_999_context_crop"
        assert structured["extra"]["gt_coverage"] == 1.0
        assert len(structured["extra"]["requested_padding_ratios"]) == 4
        pixel_augmentation = structured["extra"]["pixel_augmentation"]
        if task == "image_context_reconstruction":
            assert pixel_augmentation == {"profile": "none", "operations": []}
        else:
            assert pixel_augmentation["profile"] == "synthetic_realism_v1"
            assert 1 <= len(pixel_augmentation["operations"]) <= 3
            assert pixel_augmentation["dimensions_unchanged"] is True
        image_path = (task_root / "structured" / structured["image_path"]).resolve()
        assert image_path.is_file()
        with Image.open(image_path) as image:
            assert image.size == (structured["image_width"], structured["image_height"])
        assert (task_root / "structured/val.jsonl").read_text() == ""
        assert (task_root / "sft/val.jsonl").read_text() == ""
        readme = (task_root / "README.md").read_text(encoding="utf-8")
        assert "Rebuild selection snapshot: `selection/train.jsonl`" in readme
        assert "- Selection manifest:" not in readme
        maintained_selection = _read_one(task_root / "selection/train.jsonl")
        assert maintained_selection["sample_id"] == structured["sample_id"].removesuffix(
            "__context_00"
        )
        assert structured["extra"]["selection_manifest"] == "../selection/train.jsonl"
        summary = json.loads((task_root / "build_summary.json").read_text(encoding="utf-8"))
        assert summary["excluded_test_rows"] == int(task == "image_context_reconstruction")

        crop_left, crop_top, crop_right, crop_bottom = structured["extra"]["crop_box"]
        crop_width, crop_height = crop_right - crop_left, crop_bottom - crop_top
        proposal = structured["extra"]["proposal_bbox"]
        local_proposal = [
            proposal[0] - crop_left,
            proposal[1] - crop_top,
            proposal[2] - crop_left,
            proposal[3] - crop_top,
        ]
        assert sft["prompt_args"]["proposal_bbox_2d"] == quantize_qwen_bbox(
            local_proposal,
            width=crop_width,
            height=crop_height,
        )

    shape_structured = _read_one(
        output_root / "shape_context_reconstruction/structured/train.jsonl"
    )
    shape_target = json.loads(
        _read_one(output_root / "shape_context_reconstruction/sft/train.jsonl")["target_text"]
    )
    left, top, right, bottom = shape_structured["extra"]["crop_box"]
    expected_shape_points = [
        quantize_qwen_point(
            [point["point"][0] - left, point["point"][1] - top],
            width=right - left,
            height=bottom - top,
        )
        for point in shape_parameters["corners"]
    ]
    assert [corner["point"] for corner in shape_target["parameters"]["corners"]] == (
        expected_shape_points
    )

    line_structured = _read_one(output_root / "line_context_reconstruction/structured/train.jsonl")
    line_target = json.loads(
        _read_one(output_root / "line_context_reconstruction/sft/train.jsonl")["target_text"]
    )
    left, top, right, bottom = line_structured["extra"]["crop_box"]
    expected_segments = [
        [
            quantize_qwen_point(
                [point[0] - left, point[1] - top],
                width=right - left,
                height=bottom - top,
            )
            for point in segment
        ]
        for segment in line_parameters["points"]
    ]
    assert line_target["parameters"]["points"] == expected_segments


def test_builds_archived_line_context_point_subset_from_clean_full_image(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive2"
    point_selection = archive_root / "point_arrow/structured/train.jsonl"
    point_selection.parent.mkdir(parents=True)
    full_manifest = archive_root / "grounding_layout/structured/train.jsonl"
    full_manifest.parent.mkdir(parents=True)
    image_root = archive_root / "grounding_layout/images/train"
    image_root.mkdir(parents=True)

    rows = (
        ("part1/json/source.json", "source__arrow_0002", [30, 40, 180, 120]),
        ("part1/json/heldout.json", "heldout__arrow_0000", [20, 30, 100, 90]),
    )
    point_lines: list[str] = []
    full_lines: list[str] = []
    for source_json, sample_id, source_bbox in rows:
        image_name = f"{Path(source_json).stem}__full.png"
        Image.new("RGB", (240, 160), "white").save(image_root / image_name)
        source_linestrip = [
            [source_bbox[0], source_bbox[1] + 20],
            [100, source_bbox[1] + 20],
            [source_bbox[2], source_bbox[3] - 10],
        ]
        point_lines.append(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "image_path": f"../images/train/{sample_id}.png",
                    "instances": [
                        {
                            "label": "arrow",
                            "bbox": [0, 0, 10, 10],
                            "linestrip": [[1, 1], [9, 9]],
                        }
                    ],
                    "extra": {
                        "task": "point_arrow",
                        "source_json": source_json,
                        "source_image": source_json.replace("/json/", "/images/").replace(
                            ".json", ".png"
                        ),
                        "source_instance_index": 2,
                        "source_bbox": source_bbox,
                        "source_linestrip": source_linestrip,
                        "crop_box": [0, 0, 200, 140],
                    },
                }
            )
        )
        full_lines.append(
            json.dumps(
                {
                    "sample_id": f"{Path(source_json).stem}__full",
                    "image_path": f"../images/train/{image_name}",
                    "image_width": 240,
                    "image_height": 160,
                    "instances": [],
                    "extra": {
                        "task": "grounding_layout",
                        "view_type": "full_image",
                        "source_json": source_json,
                    },
                }
            )
        )
    point_selection.write_text("\n".join(point_lines) + "\n", encoding="utf-8")
    full_manifest.write_text("\n".join(full_lines) + "\n", encoding="utf-8")

    synthetic_root = tmp_path / "synthetic"
    (synthetic_root / "img").mkdir(parents=True)
    (synthetic_root / "gt_standard").mkdir()
    Image.new("RGB", (240, 160), "white").save(synthetic_root / "img/fork.png")
    multi_bbox = [30.0, 30.0, 210.0, 140.0]
    single_bbox = [20.0, 10.0, 220.0, 20.0]
    multi_parameters = {
        "line_type": "straight",
        "line_style": "path",
        "is_single": False,
        "points": [
            [[30, 80], [110, 80], [200, 30]],
            [[110, 80], [200, 130]],
        ],
        "begin_arrow": "none",
        "end_arrow": "triangle",
        "dash_style": "solid",
        "fill_color": "#000000",
    }
    single_parameters = {
        **multi_parameters,
        "is_single": True,
        "points": [[[20, 15], [220, 15]]],
    }
    synthetic_parameters = [
        multi_parameters,
        {
            **multi_parameters,
            "points": [
                [[30, 80], [110, 80], [200, 30]],
                [[110, 80], [200, 80]],
                [[110, 80], [200, 130]],
            ],
        },
        {
            **multi_parameters,
            "points": [
                [[30, 80], [110, 80], [200, 30]],
                [[110, 80], [200, 65]],
                [[110, 80], [200, 95]],
                [[110, 80], [200, 130]],
            ],
        },
        {**multi_parameters},
        single_parameters,
    ]
    (synthetic_root / "gt_standard/fork.json").write_text(
        json.dumps(
            {
                "size": [240, 160],
                "layout": [
                    {
                        "type": "line",
                        "bbox": single_bbox if index == 4 else multi_bbox,
                        "parameters": parameters,
                    }
                    for index, parameters in enumerate(synthetic_parameters)
                ],
            }
        ),
        encoding="utf-8",
    )
    synthetic_selection = tmp_path / "synthetic_multi.jsonl"
    for index in range(len(synthetic_parameters)):
        _write_selection(
            synthetic_selection,
            sample_id=f"fork__line_{index:04d}",
            label="line",
            bbox=single_bbox if index == 4 else multi_bbox,
            source_json="gt_standard/fork.json",
            source_image="img/fork.png",
            instance_index=index,
        )

    prompt = tmp_path / "line_context_points.yaml"
    _write_prompt(prompt, task="line_context_points", label="line")
    excluded = tmp_path / "test.json"
    excluded.write_text(json.dumps({"items": [{"id": "heldout"}]}), encoding="utf-8")
    output_root = tmp_path / "output"

    _run_builder(
        "--archive-root",
        str(archive_root),
        "--synthetic-root",
        str(synthetic_root),
        "--output-root",
        str(output_root),
        "--line-point-selection",
        str(point_selection),
        "--line-point-full-image-manifest",
        str(full_manifest),
        "--line-point-synthetic-selection",
        str(synthetic_selection),
        "--line-point-synthetic-limit",
        "3",
        "--line-point-prompt-pool",
        str(prompt),
        "--exclude-manifest",
        str(excluded),
        "--tasks",
        "line_context_points",
        "--workers",
        "1",
        "--clean",
    )

    task_root = output_root / "line_context_points"
    structured_rows = [
        json.loads(line) for line in (task_root / "structured/train.jsonl").read_text().splitlines()
    ]
    sft_rows = [
        json.loads(line) for line in (task_root / "sft/train.jsonl").read_text().splitlines()
    ]
    assert len(structured_rows) == len(sft_rows) == 4
    paired = list(zip(structured_rows, sft_rows, strict=True))
    archived_pairs = [
        pair for pair in paired if pair[1]["extra"]["source_type"] == "archived_real_context_points"
    ]
    synthetic_pairs = [
        pair
        for pair in paired
        if pair[1]["extra"]["source_type"] == "synthetic_gt_standard_context_points"
    ]
    assert len(archived_pairs) == 1
    assert len(synthetic_pairs) == 3
    structured, sft = archived_pairs[0]
    target = json.loads(sft["target_text"])
    assert target["type"] == "line"
    assert set(target["parameters"]) == {"is_single", "points"}
    assert target["parameters"]["is_single"] is True
    assert len(target["parameters"]["points"]) == 1
    assert len(target["parameters"]["points"][0]) == 3
    assert all(
        0 <= coordinate <= 999
        for point in target["parameters"]["points"][0]
        for coordinate in point
    )
    assert sft["extra"]["source_type"] == "archived_real_context_points"
    assert structured["extra"]["pixel_augmentation"] == {"profile": "none", "operations": []}
    assert structured["extra"]["geometry_coverage"] == 1.0
    assert structured["extra"]["distractor_count"] is None
    assert structured["extra"]["archive_provenance"]["source_point_order"] == "arrow_tail_to_head"
    selected_segment_counts = []
    for synthetic_structured, synthetic_sft in synthetic_pairs:
        synthetic_target = json.loads(synthetic_sft["target_text"])
        assert synthetic_target == {
            "type": "line",
            "parameters": {
                "is_single": False,
                "points": synthetic_structured["instances"][0]["parameters"]["points"],
            },
        }
        selected_segment_counts.append(len(synthetic_target["parameters"]["points"]))
        assert synthetic_structured["extra"]["pixel_augmentation"]["profile"] == (
            "synthetic_realism_v1"
        )
    assert sorted(selected_segment_counts) == [2, 3, 4]
    maintained_rows = [
        json.loads(line) for line in (task_root / "selection/train.jsonl").read_text().splitlines()
    ]
    assert len(maintained_rows) == 4
    assert all(row["instances"][0]["label"] == "line" for row in maintained_rows)
    summary = json.loads((task_root / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["rows"] == 4
    assert summary["excluded_test_rows"] == 1
    assert summary["counts"]["line_is_single_false"] == 3
    assert summary["counts"]["selection_source_archived_point"] == 1
    assert summary["counts"]["selection_source_synthetic_point_multi"] == 3
    assert summary["counts"]["synthetic_multi_eligible"] == 4
    assert summary["counts"]["synthetic_multi_selected"] == 3
    assert summary["counts"]["synthetic_multi_dropped_by_cap"] == 1
    for segment_count in (2, 3, 4):
        assert summary["counts"][f"synthetic_multi_selected_segments_{segment_count}"] == 1
    assert summary["counts"]["synthetic_multi_rejected_single"] == 1


def test_worker_failure_keeps_previous_task_and_removes_staging(tmp_path: Path) -> None:
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
                        "parameters": {
                            "line_type": "straight",
                            "points": [[[10, 10], [90, 90]], {"invalid": "segment"}],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "line.jsonl"
    _write_selection(
        selection,
        sample_id="invalid-line",
        label="line",
        bbox=bbox,
        source_json="line.json",
        source_image="line.png",
        instance_index=0,
    )
    prompt = tmp_path / "line.yaml"
    _write_prompt(prompt, task="line_context_reconstruction", label="line")
    output_root = tmp_path / "output"
    task_root = output_root / "line_context_reconstruction"
    for kind in ("structured", "sft"):
        (task_root / kind).mkdir(parents=True, exist_ok=True)
        (task_root / kind / "train.jsonl").write_text("sentinel\n", encoding="utf-8")

    result = _run_builder(
        "--synthetic-root",
        str(synthetic_root),
        "--output-root",
        str(output_root),
        "--line-selection",
        str(selection),
        "--line-prompt-pool",
        str(prompt),
        "--exclude-manifest",
        "",
        "--tasks",
        "line_context_reconstruction",
        "--workers",
        "1",
        "--clean",
        check=False,
    )

    assert result.returncode != 0
    assert "Invalid line point segment" in result.stderr
    assert (task_root / "structured/train.jsonl").read_text() == "sentinel\n"
    assert (task_root / "sft/train.jsonl").read_text() == "sentinel\n"
    assert not list(output_root.glob(".line_context_reconstruction.staging.*"))


def test_builds_real_shape_context_attribute_subset_without_geometry(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "images").mkdir(parents=True)
    (raw_root / "json").mkdir()
    Image.new("RGB", (320, 240), "white").save(raw_root / "images/prod.png")
    bbox = [80.0, 50.0, 240.0, 190.0]
    (raw_root / "json/prod.json").write_text(
        json.dumps(
            {
                "image_width": 320,
                "image_height": 240,
                "instances": [{"label": "shape", "bbox": bbox}],
            }
        ),
        encoding="utf-8",
    )
    parameters = {
        "shape_type": "rectangle",
        "border": {"type": "uniform", "style": "solid", "color": "#112233"},
        "fill": {"type": "solid", "color": "#DDEEFF"},
        "effect": {"type": "none"},
    }
    selection = tmp_path / "shape_attributes.jsonl"
    _write_selection(
        selection,
        sample_id="prod__shape_0000",
        label="shape",
        bbox=bbox,
        source_json="json/prod.json",
        source_image="images/prod.png",
        instance_index=0,
        parameters=parameters,
        weak_label={
            "source": "api",
            "task": "shape_context_attributes",
            "schema_version": "v1",
            "model_id": "test-model",
            "batch_id": "batch-1",
            "created_at_utc": "2026-07-17T00:00:00+00:00",
        },
    )
    prompt = tmp_path / "shape_attributes.yaml"
    _write_prompt(prompt, task="shape_context_attributes", label="shape")
    output_root = tmp_path / "output"

    _run_builder(
        "--raw-root",
        str(raw_root),
        "--output-root",
        str(output_root),
        "--shape-attribute-selection",
        str(selection),
        "--shape-attribute-prompt-pool",
        str(prompt),
        "--exclude-manifest",
        "",
        "--tasks",
        "shape_context_attributes",
        "--workers",
        "1",
        "--seed",
        "123",
        "--clean",
    )

    task_root = output_root / "shape_context_attributes"
    structured = _read_one(task_root / "structured/train.jsonl")
    sft = _read_one(task_root / "sft/train.jsonl")
    assert sft["dataset_name"] == "shape_context_attributes"
    assert json.loads(sft["target_text"]) == {"type": "shape", "parameters": parameters}
    assert sft["extra"]["source_type"] == "api_weak_real_context"
    assert set(sft["prompt_args"]) == {"proposal_bbox_2d"}
    assert len(structured["extra"]["requested_padding_ratios"]) == 4
    assert len(set(structured["extra"]["requested_padding_ratios"])) > 1
    assert structured["extra"]["pixel_augmentation"] == {"profile": "none", "operations": []}
    assert not {"corners", "body_corners", "body_bbox", "tail"}.intersection(
        json.loads(sft["target_text"])["parameters"]
    )
    maintained = _read_one(task_root / "selection/train.jsonl")
    assert maintained["instances"][0]["parameters"] == parameters


def test_real_shape_context_attribute_rejects_schema_drift(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "images").mkdir(parents=True)
    (raw_root / "json").mkdir()
    Image.new("RGB", (320, 240), "white").save(raw_root / "images/prod.png")
    bbox = [80.0, 50.0, 240.0, 190.0]
    (raw_root / "json/prod.json").write_text(
        json.dumps(
            {
                "image_width": 320,
                "image_height": 240,
                "instances": [{"label": "shape", "bbox": bbox}],
            }
        ),
        encoding="utf-8",
    )
    parameters = {
        "shape_type": "rectangle",
        "border": {
            "type": "uniform",
            "style": "solid",
            "color": "#112233",
            "color2": "#445566",
        },
        "fill": {"type": "none"},
        "effect": {"type": "none"},
    }
    selection = tmp_path / "shape_attributes.jsonl"
    _write_selection(
        selection,
        sample_id="prod__shape_0000",
        label="shape",
        bbox=bbox,
        source_json="json/prod.json",
        source_image="images/prod.png",
        instance_index=0,
        parameters=parameters,
        weak_label={
            "source": "api",
            "task": "shape_context_attributes",
            "schema_version": "v1",
            "model_id": "test-model",
            "batch_id": "batch-1",
            "created_at_utc": "2026-07-17T00:00:00+00:00",
        },
    )
    prompt = tmp_path / "shape_attributes.yaml"
    _write_prompt(prompt, task="shape_context_attributes", label="shape")
    output_root = tmp_path / "output"

    completed = _run_builder(
        "--raw-root",
        str(raw_root),
        "--output-root",
        str(output_root),
        "--shape-attribute-selection",
        str(selection),
        "--shape-attribute-prompt-pool",
        str(prompt),
        "--exclude-manifest",
        "",
        "--tasks",
        "shape_context_attributes",
        "--workers",
        "1",
        "--clean",
        check=False,
    )

    assert completed.returncode != 0
    assert "Invalid real weak parameters" in completed.stderr
    assert "border:unexpected_fields:color2" in completed.stderr
    assert not (output_root / "shape_context_attributes").exists()
    assert not list(output_root.glob(".shape_context_attributes.staging.*"))


def test_shape_context_attribute_stratification_caps_rectangle_deterministically() -> None:
    module = _load_module()

    def selection(index: int, shape_type: str) -> object:
        return module.Selection(
            sample_id=f"sample_{index:02d}",
            stem=f"source_{index:02d}",
            instance_index=0,
            source_bbox=(0.0, 0.0, 10.0, 10.0),
            source_image=f"images/source_{index:02d}.png",
            source_json=f"json/source_{index:02d}.json",
            parameters={
                "shape_type": shape_type,
                "border": {"type": "none"},
                "fill": {"type": "none"},
                "effect": {"type": "none"},
            },
        )

    rows = [selection(index, "rectangle") for index in range(8)] + [
        selection(8, "oval"),
        selection(9, "other"),
        selection(10, "diamond"),
        selection(11, "triangle"),
    ]

    first, first_counts = module._stratify_shape_attribute_selections(
        rows,
        max_rectangle_fraction=0.5,
        seed=42,
    )
    second, second_counts = module._stratify_shape_attribute_selections(
        rows,
        max_rectangle_fraction=0.5,
        seed=42,
    )

    assert [row.sample_id for row in first] == [row.sample_id for row in second]
    assert first_counts == second_counts
    assert len(first) == 8
    assert sum(row.parameters["shape_type"] == "rectangle" for row in first) == 4
    assert {row.parameters["shape_type"] for row in first if row.parameters} >= {
        "oval",
        "other",
        "diamond",
        "triangle",
    }
    assert first_counts["sampling_dropped_rectangle"] == 4
