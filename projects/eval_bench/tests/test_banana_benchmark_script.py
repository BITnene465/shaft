from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from eval_bench.artifacts import BenchmarkArtifacts


def test_banana_benchmark_adds_point_arrow_crop_split(tmp_path: Path) -> None:
    module = _load_banana_benchmark_script()
    artifacts = BenchmarkArtifacts(tmp_path / "store", "banana_val")
    artifacts.ensure()
    manifest = {
        "benchmark_id": "banana_val",
        "tasks": ["detection"],
        "root": str(artifacts.data_dir),
        "split": "suite",
        "manifest_path": str(artifacts.split_path("suite")),
        "sample_count": 0,
        "split_manifests": {"suite": str(artifacts.split_path("suite"))},
        "sample_counts": {"suite": 0},
        "labels": ["shape"],
        "metadata": {"slices": {}, "source_manifest_paths": {}},
    }
    artifacts.split_path("suite").write_text("", encoding="utf-8")
    artifacts.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    crop_image = tmp_path / "point_arrow" / "images" / "val" / "crop.png"
    crop_image.parent.mkdir(parents=True)
    crop_image.write_bytes(b"png")
    structured = tmp_path / "point_arrow" / "structured" / "val.jsonl"
    structured.parent.mkdir(parents=True)
    structured.write_text(
        json.dumps(
            {
                "sample_id": "sample__arrow_0001",
                "image_path": "../images/val/crop.png",
                "image_width": 80,
                "image_height": 40,
                "instances": [
                    {
                        "label": "arrow",
                        "bbox": [10, 10, 70, 30],
                        "linestrip": [[12, 20], [68, 20]],
                    }
                ],
                "extra": {"source_json": "part1/json/a.json"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    updated = module.add_point_arrow_crop_split(
        store_root=tmp_path / "store",
        benchmark_id="banana_val",
        structured_path=structured,
    ).to_dict()

    assert updated["tasks"] == ["detection", "keypoint"]
    assert updated["split_manifests"]["point_arrow"] == str(artifacts.split_path("point_arrow"))
    assert updated["sample_counts"]["point_arrow"] == 1
    assert updated["metadata"]["slices"]["point_arrow"]["view_type"] == "arrow_crop"
    assert artifacts.split_path("point_arrow").read_text(encoding="utf-8") == (
        "point_arrow/json/sample__arrow_0001.json\n"
    )
    gt_payload = json.loads(
        (artifacts.data_dir / "point_arrow" / "json" / "sample__arrow_0001.json").read_text(
            encoding="utf-8"
        )
    )
    assert gt_payload["image_path"] == "point_arrow/images/sample__arrow_0001.png"
    assert gt_payload["extra"]["view_type"] == "arrow_crop"
    assert (artifacts.data_dir / "point_arrow" / "images" / "sample__arrow_0001.png").exists()


def _load_banana_benchmark_script():
    script_path = Path("scripts/tasks/create_banana_v2_4_eval_benchmark.py").resolve()
    spec = importlib.util.spec_from_file_location("create_banana_v2_4_eval_benchmark", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
