from __future__ import annotations

import json
from pathlib import Path

from eval_bench.artifacts import BenchmarkArtifacts
from eval_bench.benchmark import (
    BenchmarkSliceSpec,
    create_benchmark_from_raw_data,
    create_benchmark_suite_from_raw_data,
    resolve_benchmark_split_path,
)


def test_create_benchmark_copies_raw_data_validation_subset(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_data"
    (raw_root / "part1" / "images").mkdir(parents=True)
    (raw_root / "part1" / "json").mkdir(parents=True)
    image_path = raw_root / "part1" / "images" / "a.png"
    image_path.write_bytes(b"fake-image")
    raw_json = raw_root / "part1" / "json" / "a.json"
    raw_json.write_text(
        json.dumps(
            {
                "schema": "shaft.raw_data.v1",
                "image_path": "part1/images/a.png",
                "image_width": 10,
                "image_height": 10,
                "instances": [{"label": "icon", "bbox": [1, 2, 3, 4]}],
            }
        ),
        encoding="utf-8",
    )
    split_path = raw_root / "splits" / "layout_val.txt"
    split_path.parent.mkdir()
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")

    manifest = create_benchmark_from_raw_data(
        store_root=tmp_path / "store",
        benchmark_id="layout_val_v1",
        tasks=["detection", "keypoint"],
        source_root=raw_root,
        source_manifest=split_path,
        split="val",
        layers=["layout"],
    )

    artifacts = BenchmarkArtifacts(tmp_path / "store", "layout_val_v1")
    assert manifest.sample_count == 1
    assert Path(manifest.root) == artifacts.data_dir
    assert (artifacts.data_dir / "part1" / "json" / "a.json").exists()
    assert (artifacts.data_dir / "part1" / "images" / "a.png").exists()
    assert artifacts.split_path("val").read_text(encoding="utf-8") == "part1/json/a.json\n"
    benchmark_payload = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert benchmark_payload["tasks"] == ["detection", "keypoint"]
    assert benchmark_payload["layers"] == ["layout"]
    assert benchmark_payload["labels"] == ["icon"]


def test_create_benchmark_suite_writes_named_splits(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_data"
    (raw_root / "part1" / "images").mkdir(parents=True)
    (raw_root / "part1" / "json").mkdir(parents=True)
    for stem, label in (("a", "arrow"), ("b", "shape")):
        (raw_root / "part1" / "images" / f"{stem}.png").write_bytes(b"fake-image")
        (raw_root / "part1" / "json" / f"{stem}.json").write_text(
            json.dumps(
                {
                    "schema": "shaft.raw_data.v1",
                    "image_path": f"part1/images/{stem}.png",
                    "instances": [{"label": label, "bbox": [1, 2, 3, 4]}],
                }
            ),
            encoding="utf-8",
        )

    manifest = create_benchmark_suite_from_raw_data(
        store_root=tmp_path / "store",
        benchmark_id="banana_val",
        source_root=raw_root,
        split="suite",
        default_slice="grounding_arrow",
        flatten=True,
        slices=[
            BenchmarkSliceSpec(
                split="grounding_arrow",
                tasks=["detection"],
                entries=["part1/json/a.json"],
                layers=["arrow"],
                target_labels=["arrow"],
            ),
            BenchmarkSliceSpec(
                split="grounding_shape",
                tasks=["detection"],
                entries=["part1/json/b.json"],
                layers=["layout"],
                target_labels=["shape"],
            ),
        ],
    )

    artifacts = BenchmarkArtifacts(tmp_path / "store", "banana_val")
    payload = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest.sample_count == 2
    assert payload["split"] == "suite"
    assert payload["manifest_path"] == str(artifacts.split_path("suite"))
    assert payload["sample_counts"] == {"grounding_arrow": 1, "grounding_shape": 1, "suite": 2}
    assert sorted(payload["split_manifests"]) == ["grounding_arrow", "grounding_shape", "suite"]
    assert resolve_benchmark_split_path(payload, split="suite") == artifacts.split_path("suite")
    assert artifacts.split_path("suite").read_text(encoding="utf-8") == (
        "json/part1__a.json\njson/part1__b.json\n"
    )
    assert artifacts.split_path("grounding_arrow").read_text(encoding="utf-8") == (
        "json/part1__a.json\n"
    )
    assert artifacts.split_path("grounding_shape").read_text(encoding="utf-8") == (
        "json/part1__b.json\n"
    )
    assert (artifacts.data_dir / "json" / "part1__a.json").exists()
    assert (artifacts.data_dir / "images" / "part1__a.png").exists()
    flattened_payload = json.loads(
        (artifacts.data_dir / "json" / "part1__a.json").read_text(encoding="utf-8")
    )
    assert flattened_payload["image_path"] == "images/part1__a.png"
    assert flattened_payload["extra"]["source_json"] == "part1/json/a.json"
