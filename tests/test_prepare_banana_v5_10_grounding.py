import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image
import pytest

pytest.importorskip("imagehash", reason="Offline grounding audit requires ImageHash")


def load():
    path = Path("scripts/tasks/prepare_banana_v5_10_grounding.py")
    spec = importlib.util.spec_from_file_location("grounding_recipe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recipe():
    return json.loads(Path("configs/data/preparation/banana_v5_10_grounding.json").read_text())


def test_jpeg_plan_reproducible_and_covers_range():
    m = load()
    r = recipe()
    plans = [m.jpeg_plan(f"json/{i}.json", r) for i in range(5000)]
    assert plans == [m.jpeg_plan(f"json/{i}.json", r) for i in range(5000)]
    assert {p["quality"] for p in plans} == set(range(40, 96))
    assert {p["subsampling"] for p in plans} == {2}
    for low, high, weight in r["jpeg_quality_bands"]:
        actual = sum(low <= p["quality"] <= high for p in plans) / len(plans)
        assert abs(actual - weight) < 0.025


def test_test_gate_handles_cross_id_content_and_near_match():
    m = load()
    source = {"id": "train", "sha256": "one", "phash": "0000000000000000"}
    tests = [{"id": "test", "sha256": "two", "phash": "000000000000003f"}]
    assert m.test_matches(source, tests, 6) == ["test"]
    assert m.test_matches(source, tests, 5) == []
    tests[0]["sha256"] = "one"
    assert m.test_matches(source, tests, 0) == ["test"]


def test_jpeg_preserves_clean_twin_geometry_and_source(tmp_path):
    m = load()
    (tmp_path / "images/train").mkdir(parents=True)
    (tmp_path / "structured").mkdir()
    image = tmp_path / "images/train/a.png"
    Image.new("RGB", (64, 32), (134, 201, 22)).save(image)
    before = image.read_bytes()
    row = {
        "sample_id": "a",
        "image_path": "../images/train/a.png",
        "image_width": 64,
        "image_height": 32,
        "instances": [{"label": "shape", "bbox": [1, 2, 60, 30]}],
        "extra": {"source_json": "json/a.json", "view_type": "full_image"},
    }
    result = m.jpeg_row((row, tmp_path, recipe()))
    assert result["instances"] == row["instances"]
    assert result["extra"]["clean_twin_sample_id"] == "a"
    assert image.read_bytes() == before
    assert "pixel_augmentation" not in row["extra"]
    with Image.open(tmp_path / "images/train/a__jpeg.png") as im:
        assert im.size == (64, 32)


def test_mild_degradation_config_consumed():
    path = Path("scripts/tasks/build_grounding_structured.py")
    spec = importlib.util.spec_from_file_location("grounding_builder", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    config = m.BuildConfig(
        raw_root=Path("unused"),
        task_name="grounding_layout",
        split="train",
        output_root=Path("unused"),
        image_output_dir=Path("unused"),
        seed=465,
        candidate_count=32,
        negative_candidate_count=48,
        negative_ratio=0.03,
        density_crop_ratio=0.15,
        blur_ratio=0,
        padded_full_ratio=0.1,
        padding_min_ratio=0.05,
        padding_max_ratio=0.25,
        augmentation_profile="layout_multiscale_v1",
        min_pixels=1000000,
        max_pixels=2000000,
        processor_factor=32,
        clean_resize_views=0.9,
        degraded_resize_ratio=0.25,
        degradation_max_severity="L1",
    )
    metas = [m.SourceMeta(f"json/{i}.json", 2000, 1600, 10, True) for i in range(40)]
    plans = m._build_multiscale_plans(metas, config=config)
    degraded = [d for plan in plans.values() for d in plan.degradation_plans]
    assert len(degraded) == 10
    assert {d.severity for d in degraded} == {"L1"}
    assert {d.family for d in degraded} == {"gaussian_blur", "gaussian_noise"}


def test_canonical_test_resolves_from_explicit_eval_directory(tmp_path):
    from types import SimpleNamespace

    m = load()
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "json").mkdir()
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    Image.new("RGB", (20, 10), "white").save(evaluation / "ppt_0001.png")
    Image.new("RGB", (20, 10), "white").save(raw / "images/train.png")
    (raw / "json/train.json").write_text(
        json.dumps({"size": [20, 10], "layout": [{"type": "shape", "bbox": [0, 0, 20, 10]}]})
    )
    manifest = tmp_path / "test.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "ppt_0001",
                        "image_path": "images/ppt_0001.png",
                        "width": 20,
                        "height": 10,
                    }
                ]
            }
        )
    )
    task = tmp_path / "task"
    m.select(
        SimpleNamespace(
            raw_root=raw,
            test_image_dir=[evaluation],
            test_manifest=manifest,
            test_raw_root=raw,
            workers=1,
            limit=0,
        ),
        recipe(),
        task,
    )
    metadata = json.loads((task / "selection/metadata.json").read_text())
    assert metadata["excluded_test_candidates"] == 1
    assert metadata["selected_sources"] == 0
