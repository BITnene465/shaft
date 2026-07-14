from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
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

    Image.new("RGB", (1024, 768), "white").save(image_path)
    payload = {
        "image_width": 1024,
        "image_height": 768,
        "instances": [
            {"label": "shape", "bbox": [64, 64, 256, 256]},
            {"label": "icon", "bbox": [320, 64, 448, 224]},
            {"label": "image", "bbox": [512, 96, 768, 320]},
            {"label": "arrow", "bbox": [96, 448, 512, 480]},
            {"label": "line", "bbox": [576, 512, 896, 544]},
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
            "1.0",
            "--negative-ratio",
            "0.0",
            "--degraded-resize-ratio",
            "1.0",
            "--padded-full-ratio",
            "1.0",
            "--clean-resize-views",
            "2.0",
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
    assert "continuous_resize_full" in view_types
    assert "random_padded_full" in view_types
    assert "degraded_resize_full" in view_types
    assert not view_types & {"blur_full", "blur_crop"}
    readme = (output_root / "grounding_layout" / "README.md").read_text(encoding="utf-8")
    assert "Train All-Row Pixel Bands" in readme
    assert "Train Clean Resize Pixel Bands" in readme
    labels = {
        instance["label"]
        for row in rows
        for instance in row["instances"]
    }
    assert {"shape", "icon", "image", "line"}.issubset(labels)
    resized = next(row for row in rows if row["extra"]["view_type"] == "continuous_resize_full")
    assert resized["image_width"] % 32 == 0
    assert resized["image_height"] % 32 == 0
    assert resized["image_width"] <= 1024 * 2
    assert resized["image_height"] <= 768 * 2
    assert resized["extra"]["spatial_augmentation"]["kernel"] in {
        "bicubic",
        "lanczos",
        "area",
    }
    padded = next(row for row in rows if row["extra"]["view_type"] == "random_padded_full")
    padding = padded["extra"]["spatial_augmentation"]["padding"]
    assert padding["left"] + padding["right"] > 0
    assert padding["top"] + padding["bottom"] > 0


def test_multiscale_plan_respects_alignment_separation_and_low_band_l3_rule() -> None:
    module = _load_module()
    config = module.BuildConfig(
        raw_root=Path("data/raw"),
        task_name="grounding_layout",
        split="train",
        output_root=Path("unused"),
        image_output_dir=Path("unused"),
        seed=42,
        candidate_count=8,
        negative_candidate_count=8,
        negative_ratio=0.03,
        density_crop_ratio=0.25,
        blur_ratio=0.0,
        padded_full_ratio=0.1,
        padding_min_ratio=0.05,
        padding_max_ratio=0.25,
        augmentation_profile="layout_multiscale_v1",
        min_pixels=200_704,
        max_pixels=4_000_000,
        processor_factor=32,
        clean_resize_views=2.9,
        degraded_resize_ratio=1.2,
    )
    metas = [
        module.SourceMeta(f"json/sample_{index:03d}.json", 2048, 1536, index + 1, True)
        for index in range(40)
    ]

    plans = module._build_multiscale_plans(metas, config=config)
    repeated_plans = module._build_multiscale_plans(metas, config=config)
    half_degraded_plans = module._build_multiscale_plans(
        metas,
        config=replace(config, degraded_resize_ratio=0.5),
    )

    assert plans == repeated_plans
    assert sum(plan.make_padded for plan in plans.values()) == 4
    assert sum(len(plan.resize_plans) for plan in plans.values()) == 116
    assert sum(len(plan.degradation_plans) for plan in plans.values()) == 48
    assert sum(len(plan.degradation_plans) for plan in half_degraded_plans.values()) == 20
    for plan in plans.values():
        pixels = [resize.actual_pixels for resize in plan.resize_plans]
        assert all(resize.width % 32 == 0 for resize in plan.resize_plans)
        assert all(resize.height % 32 == 0 for resize in plan.resize_plans)
        assert all(
            max(left, right) / min(left, right) >= 1.35
            for index, left in enumerate(pixels)
            for right in pixels[index + 1 :]
        )
        for degradation in plan.degradation_plans:
            resize = plan.resize_plans[degradation.resize_index]
            assert not (degradation.severity == "L3" and resize.pixel_band == "0.2-0.5M")


def test_scaled_instances_clamp_floating_point_edges_to_canvas() -> None:
    module = _load_module()
    instances = [
        module.SourceInstance(index=0, label="shape", bbox=(0.0, 0.0, 966.0, 130.0)),
    ]

    scaled = module._scale_instances(
        instances,
        scale_x=1312 / 966,
        scale_y=192 / 130,
        max_x=1312,
        max_y=192,
    )

    assert scaled == [{"label": "shape", "bbox": [0.0, 0.0, 1312.0, 192.0]}]


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
