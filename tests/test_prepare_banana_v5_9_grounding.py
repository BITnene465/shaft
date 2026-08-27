from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image


def _load_module():
    script = Path("scripts/tasks/prepare_banana_v5_9_grounding.py").resolve()
    spec = importlib.util.spec_from_file_location("prepare_banana_v5_9_grounding", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color=color).save(path)


def _write_annotation(
    path: Path,
    *,
    layout: list[dict[str, object]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "size": [16, 12],
        "layout": layout
        or [
            {"type": "shape", "bbox": [1, 1, 5, 5]},
            {"type": "icon", "bbox": [6, 1, 10, 5]},
            {"type": "line", "bbox": [1, 7, 14, 10]},
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_prepare_v5_9_grounding_builds_id_safe_increment(tmp_path: Path) -> None:
    module = _load_module()
    base = tmp_path / "base"
    incoming_json = tmp_path / "incoming_json"
    incoming_images = tmp_path / "incoming_images"

    _write_annotation(base / "json/base.json")
    _write_image(base / "images/base.png", color=(1, 2, 3))
    (base / "splits").mkdir(parents=True)
    (base / "splits/grounding_layout.train.txt").write_text("base\n")

    _write_annotation(incoming_json / "good.json")
    _write_image(incoming_images / "good.png", color=(4, 5, 6))

    # ID-only test exclusion: identical pixels to a test image are still allowed.
    _write_annotation(incoming_json / "same_pixels_new_id.json")
    _write_image(incoming_images / "same_pixels_new_id.png", color=(7, 8, 9))
    _write_image(tmp_path / "real_v1/test_only.png", color=(7, 8, 9))

    _write_annotation(incoming_json / "test_only.json")
    _write_image(incoming_images / "test_only.png", color=(10, 11, 12))

    _write_annotation(
        incoming_json / "bad_bbox.json",
        layout=[
            {"type": "shape", "bbox": [1, 1, 1, 5]},
            {"type": "icon", "bbox": [6, 1, 10, 5]},
            {"type": "line", "bbox": [1, 7, 14, 10]},
        ],
    )
    _write_image(incoming_images / "bad_bbox.png", color=(13, 14, 15))

    for sample_id in ("duplicate_a", "duplicate_b"):
        _write_annotation(incoming_json / f"{sample_id}.json")
        _write_image(incoming_images / f"{sample_id}.png", color=(16, 17, 18))

    args = argparse.Namespace(
        base_raw_root=str(base),
        base_train_split=str(base / "splits/grounding_layout.train.txt"),
        incoming_json_dir=str(incoming_json),
        incoming_image_root=str(incoming_images),
        real_v1_image_dir=str(tmp_path / "real_v1"),
        real_v2_image_dir=str(tmp_path / "real_v2"),
        output_root=str(tmp_path / "output"),
        workers=2,
        clean=False,
    )

    summary = module.build(args)

    assert summary["status"] == "passed"
    assert summary["base_train"] == 1
    assert summary["incoming_total"] == 6
    assert summary["selected_new"] == 2
    assert summary["total_train"] == 3
    assert summary["test_gate"]["overlap"] == 0
    assert summary["test_gate"]["mode"] == "id_only"
    assert summary["exclusions"] == {
        "duplicate_image_quarantine": 2,
        "invalid_target_bbox": 1,
        "test_id": 1,
    }
    train_ids = (
        Path(args.output_root) / "splits/grounding_layout.train.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert train_ids == [
        "json/base.json",
        "json/good.json",
        "json/same_pixels_new_id.json",
    ]
    assert (Path(args.output_root) / "json/same_pixels_new_id.json").is_file()
    assert not (Path(args.output_root) / "json/test_only.json").exists()


def test_prepare_v5_9_grounding_preserves_existing_output_on_failure(
    tmp_path: Path,
) -> None:
    module = _load_module()
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    args = argparse.Namespace(
        base_raw_root=str(tmp_path / "missing"),
        base_train_split=str(tmp_path / "missing/train.txt"),
        incoming_json_dir=str(tmp_path / "missing/json"),
        incoming_image_root=str(tmp_path / "missing/images"),
        real_v1_image_dir=str(tmp_path / "missing/real_v1"),
        real_v2_image_dir=str(tmp_path / "missing/real_v2"),
        output_root=str(output),
        workers=2,
        clean=True,
    )

    try:
        module.build(args)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing input must fail")
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
