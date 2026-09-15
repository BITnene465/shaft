from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/tasks"))
spec = importlib.util.spec_from_file_location(
    "line_points_v510", ROOT / "scripts/tasks/prepare_banana_v5_10_line_points.py"
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_synthetic_selection_is_deterministic_and_multi_only():
    rows = [
        {"stem": str(i), "index": 0, "group": "multi_path", "stratum": str(i % 3)}
        for i in range(90)
    ] + [{"stem": "single", "index": 0, "group": "single_simple", "stratum": "s"}]
    chosen = M.choose_synthetic(rows, 15, 465)
    assert len(chosen) == 15
    assert chosen == M.choose_synthetic(list(reversed(rows)), 15, 465)
    assert {r["group"] for r in chosen} == {"multi_path"}


def test_jpeg_twin_keeps_geometry_and_clean_pixels(tmp_path):
    root = tmp_path
    (root / "structured").mkdir()
    source = root / "images/train/aa/a.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (80, 60), "white").save(source)
    before = source.read_bytes()
    row = {
        "sample_id": "a",
        "image_path": "../images/train/aa/a.png",
        "image_width": 80,
        "image_height": 60,
        "instances": [{"parameters": {"points": []}}],
        "extra": {"pixel_augmentation": {"profile": "none", "operations": []}},
    }
    sft = {
        "sample_id": "a",
        "image_path": "../../../images/train/aa/a.png",
        "target_text": "test",
        "extra": {"structured_extra": row["extra"]},
    }
    recipe = json.loads(
        (ROOT / "configs/data/preparation/banana_v5_10_line_points.json").read_text()
    )
    twin, twin_sft = M.jpeg_twin((root, row, sft, recipe))
    assert twin["instances"] == row["instances"]
    assert twin_sft["target_text"] == sft["target_text"]
    assert source.read_bytes() == before
    assert twin["extra"]["clean_sample_id"] == "a"
    op = twin["extra"]["pixel_augmentation"]["operations"][0]
    assert 60 <= op["quality"] <= 90 and op["subsampling"] == 0
    with Image.open(root / "structured" / twin["image_path"]) as image:
        assert image.size == (80, 60)
    generated = (root / "structured" / twin["image_path"]).read_bytes()
    assert M.jpeg_twin((root, row, sft, recipe)) == (twin, twin_sft)
    assert (root / "structured" / twin["image_path"]).read_bytes() == generated


def test_real_selection_does_not_edit_or_infer_missing_points(tmp_path):
    (tmp_path / "json").mkdir()
    (tmp_path / "images").mkdir()
    Image.new("RGB", (100, 100)).save(tmp_path / "images/a.png")
    path = tmp_path / "json/a.json"
    path.write_text(
        json.dumps(
            {
                "size": [100, 100],
                "layout": [
                    {
                        "type": "line",
                        "bbox": [1, 1, 90, 90],
                        "parameters": {"points": [[[5, 5], [80, 80]]]},
                    },
                    {"type": "line", "bbox": [1, 1, 90, 90], "parameters": {}},
                ],
            }
        )
    )
    frozen = {
        "id": "a",
        "json": "a.json",
        "image": "a.png",
        "json_sha256": M.S.sha(path),
        "sha256": M.S.sha(tmp_path / "images/a.png"),
    }
    before = path.read_bytes()
    rows, counts, rejected = M.scan_real((tmp_path, frozen))
    assert len(rows) == 1 and counts["empty_points"] == 1 and not rejected
    assert "parameters" not in rows[0] and path.read_bytes() == before
    stage = tmp_path / "output"
    stage.mkdir()
    task = M.B.TaskSpec(
        "line_context_points",
        "line",
        tmp_path / "selection.jsonl",
        tmp_path,
        ROOT / "configs/prompts/pools/line_context_reconstruction.v5.8.yaml",
        "real_point",
        source_dataset_id="fixture",
    )
    config = M.B.WorkerConfig(
        task, stage, "line", None, 465, 32, 60, 1, ("points",), True, None, True
    )
    built, rejected = M.build_source(
        (
            tmp_path,
            rows,
            config,
            {"json/a.json": frozen["json_sha256"], "images/a.png": frozen["sha256"]},
        )
    )
    assert len(built) == 1 and not rejected
    structured, sft = built[0]
    target = json.loads(sft["target_text"])
    assert set(target["parameters"]) == {"is_single", "points"}
    assert structured["extra"]["pixel_augmentation"]["profile"] == "none"
    assert len(target["parameters"]["points"][0]) == 2
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        M.scan_real((tmp_path, frozen))
