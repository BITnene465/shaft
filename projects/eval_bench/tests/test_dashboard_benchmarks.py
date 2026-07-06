from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval_bench.dashboard import create_app
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_dashboard_creates_benchmark_copy_from_raw_data(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_data"
    source_manifest = raw_root / "splits" / "layout_val.txt"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text("part1/json/a.json\n", encoding="utf-8")
    (raw_root / "part1" / "images").mkdir(parents=True)
    (raw_root / "part1" / "images" / "a.png").write_bytes(b"image")
    _write_json(
        raw_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "image_width": 100,
            "image_height": 50,
            "instances": [
                {"label": "icon", "bbox": [1, 2, 10, 20]},
                {"label": "arrow", "bbox": [30, 30, 40, 40]},
            ],
        },
    )

    app = create_app(store_root=tmp_path / "store", frontend_dist=tmp_path / "dist")
    client = TestClient(app)
    response = client.post(
        "/api/benchmarks",
        json={
            "benchmark_id": "multitask_val_v1",
            "source_root": str(raw_root),
            "source_manifest": str(source_manifest),
            "split": "val",
            "tasks": ["detection", "keypoint"],
            "layers": ["layout", "arrow"],
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["benchmark_id"] == "multitask_val_v1"
    assert payload["sample_count"] == 1
    assert payload["labels"] == ["arrow", "icon"]
    assert payload["layers"] == ["layout", "arrow"]
    assert payload["source_raw_root"] == str(raw_root)
    assert payload["source_manifest_path"] == str(source_manifest)
    assert payload["split_manifests"] == {"val": payload["manifest_path"]}
    assert payload["sample_counts"] == {"val": 1}
    assert payload["metadata"] == {}
    assert client.get("/api/state").json()["benchmark_count"] == 1
    copied_sample = client.get("/api/benchmarks/multitask_val_v1/samples/0").json()
    assert [item["label"] for item in copied_sample["gt_instances"]] == ["icon", "arrow"]

    conflict = client.post(
        "/api/benchmarks",
        json={
            "benchmark_id": "multitask_val_v1",
            "source_root": str(raw_root),
            "source_manifest": str(source_manifest),
            "split": "val",
            "tasks": ["detection"],
        },
    )
    assert conflict.status_code == 409


def test_dashboard_creates_benchmark_suite_from_raw_data(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_data"
    split_dir = raw_root / "splits"
    split_dir.mkdir(parents=True)
    (raw_root / "part1" / "images").mkdir(parents=True)
    for stem, label in (("arrow", "arrow"), ("shape", "shape")):
        (raw_root / "part1" / "images" / f"{stem}.png").write_bytes(b"image")
        _write_json(
            raw_root / "part1" / "json" / f"{stem}.json",
            {
                "image_path": f"part1/images/{stem}.png",
                "image_width": 100,
                "image_height": 50,
                "instances": [{"label": label, "bbox": [1, 2, 10, 20]}],
            },
        )
    (split_dir / "grounding_arrow.txt").write_text(
        "part1/json/arrow.json\n",
        encoding="utf-8",
    )
    (split_dir / "grounding_shape.txt").write_text(
        "part1/json/shape.json\n",
        encoding="utf-8",
    )

    app = create_app(store_root=tmp_path / "store", frontend_dist=tmp_path / "dist")
    client = TestClient(app)
    response = client.post(
        "/api/benchmarks",
        json={
            "benchmark_id": "grounding_suite",
            "source_root": str(raw_root),
            "split": "suite",
            "default_slice": "grounding_arrow",
            "metadata": {"owner": "eval-team"},
            "slices": [
                {
                    "split": "grounding_arrow",
                    "source_manifest": str(split_dir / "grounding_arrow.txt"),
                    "tasks": ["detection"],
                    "layers": ["arrow"],
                    "target_labels": ["arrow"],
                },
                {
                    "split": "grounding_shape",
                    "source_manifest": str(split_dir / "grounding_shape.txt"),
                    "tasks": ["detection"],
                    "layers": ["layout"],
                    "target_labels": ["shape"],
                },
            ],
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["benchmark_id"] == "grounding_suite"
    assert payload["split"] == "suite"
    assert payload["sample_count"] == 2
    assert payload["sample_counts"] == {
        "grounding_arrow": 1,
        "grounding_shape": 1,
        "suite": 2,
    }
    assert payload["source_raw_root"] == str(raw_root)
    assert payload["source_manifest_path"] == str(split_dir / "grounding_arrow.txt")
    assert payload["metadata"]["owner"] == "eval-team"
    assert set(payload["metadata"]["source_manifest_paths"]) == {
        "grounding_arrow",
        "grounding_shape",
    }
    assert payload["metadata"]["slices"]["grounding_arrow"]["target_labels"] == ["arrow"]
    assert set(payload["split_manifests"]) == {"grounding_arrow", "grounding_shape", "suite"}
    shape_page = client.get(
        "/api/benchmarks/grounding_suite/samples",
        params={"split": "grounding_shape"},
    ).json()
    assert shape_page["total"] == 1
    assert shape_page["samples"][0]["image"] == "part1/images/shape.png"


def test_dashboard_exposes_benchmark_sample_detail_and_image(tmp_path: Path) -> None:
    data_root = tmp_path / "benchmarks" / "multitask_val_v1" / "data"
    split_manifest = tmp_path / "benchmarks" / "multitask_val_v1" / "splits" / "val.txt"
    split_manifest.parent.mkdir(parents=True)
    split_manifest.write_text("part1/json/a.json\n", encoding="utf-8")
    (data_root / "part1" / "images").mkdir(parents=True)
    (data_root / "part1" / "images" / "a.png").write_bytes(b"image")
    _write_json(
        tmp_path / "benchmarks" / "multitask_val_v1" / "benchmark.json",
        {
            "benchmark_id": "multitask_val_v1",
            "tasks": ["detection", "keypoint"],
            "layers": ["layout", "arrow"],
            "split": "val",
            "sample_count": 1,
            "root": str(data_root),
            "manifest_path": str(split_manifest),
        },
    )
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "image_width": 100,
            "image_height": 50,
            "instances": [{"label": "icon", "bbox": [1, 2, 10, 20]}],
        },
    )

    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    samples = client.get("/api/benchmarks/multitask_val_v1/samples").json()["samples"]
    sample_page = client.get("/api/benchmarks/multitask_val_v1/samples").json()
    assert sample_page["total"] == 1
    assert sample_page["labels"] == ["icon"]
    filtered_benchmark_samples = client.get(
        "/api/benchmarks/multitask_val_v1/samples",
        params={"label": "icon"},
    ).json()
    assert filtered_benchmark_samples["total"] == 1
    assert filtered_benchmark_samples["filters"] == {
        "benchmark_id": "multitask_val_v1",
        "label": "icon",
    }
    assert (
        client.get("/api/benchmarks/multitask_val_v1/samples", params={"label": "arrow"}).json()[
            "total"
        ]
        == 0
    )
    assert samples[0]["instance_count"] == 1
    assert samples[0]["labels"] == ["icon"]
    assert samples[0]["image_url"] == "/api/benchmarks/multitask_val_v1/samples/0/image"
    assert (
        samples[0]["image_preview_url"]
        == "/api/benchmarks/multitask_val_v1/samples/0/image/preview?max_side=1800"
    )
    assert (
        samples[0]["image_tile_url_template"]
        == "/api/benchmarks/multitask_val_v1/samples/0/image/tiles/{level}/{x}/{y}"
    )
    assert samples[0]["image_tile_size"] == 512

    detail = client.get("/api/benchmarks/multitask_val_v1/samples/0").json()
    assert detail["gt_instances"][0]["bbox"] == [1.0, 2.0, 10.0, 20.0]
    assert client.get("/api/benchmarks/multitask_val_v1/samples/0/image").content == b"image"

    preview = client.get("/api/settings/preview-sample").json()
    assert preview["benchmark_id"] == "multitask_val_v1"
    assert preview["sample"]["image_url"] == "/api/benchmarks/multitask_val_v1/samples/0/image"
    assert (
        preview["sample"]["image_preview_url"]
        == "/api/benchmarks/multitask_val_v1/samples/0/image/preview?max_side=1800"
    )
    assert preview["gt_instances"][0]["label"] == "icon"


def test_settings_preview_sample_falls_back_when_store_has_no_benchmark(tmp_path: Path) -> None:
    app = create_app(store_root=tmp_path / "store", frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    response = client.get("/api/settings/preview-sample")

    assert response.status_code == 200
    preview = response.json()
    assert preview["benchmark_id"] == "settings_preview"
    assert preview["sample"]["image_url"] == "/static/settings_preview.svg"
    assert preview["sample"]["image_width"] == 960
    assert preview["sample"]["image_height"] == 600
    assert preview["sample"]["labels"] == ["arrow", "icon"]
    assert [item["label"] for item in preview["gt_instances"]] == ["arrow", "icon"]
    assert preview["raw_payload"]["source"] == "settings_preview_fallback"


def test_dashboard_uses_named_benchmark_split_for_samples_and_facets(tmp_path: Path) -> None:
    data_root = tmp_path / "benchmarks" / "suite_bench" / "data"
    split_dir = tmp_path / "benchmarks" / "suite_bench" / "splits"
    suite_manifest = split_dir / "suite.txt"
    arrow_manifest = split_dir / "grounding_arrow.txt"
    shape_manifest = split_dir / "shape split.txt"
    split_dir.mkdir(parents=True)
    suite_manifest.write_text("part1/json/a.json\npart1/json/b.json\n", encoding="utf-8")
    arrow_manifest.write_text("part1/json/a.json\n", encoding="utf-8")
    shape_manifest.write_text("part1/json/b.json\n", encoding="utf-8")
    (data_root / "part1" / "images").mkdir(parents=True)
    (data_root / "part1" / "images" / "a.png").write_bytes(b"arrow")
    (data_root / "part1" / "images" / "b.png").write_bytes(b"shape")
    _write_json(
        tmp_path / "benchmarks" / "suite_bench" / "benchmark.json",
        {
            "benchmark_id": "suite_bench",
            "tasks": ["detection"],
            "layers": ["layout"],
            "split": "suite",
            "sample_count": 2,
            "root": str(data_root),
            "manifest_path": str(suite_manifest),
            "split_manifests": {
                "suite": str(suite_manifest),
                "grounding_arrow": str(arrow_manifest),
                "shape split": str(shape_manifest),
            },
            "sample_counts": {"suite": 2, "grounding_arrow": 1, "shape split": 1},
        },
    )
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [{"label": "arrow", "bbox": [1, 2, 3, 4]}],
        },
    )
    _write_json(
        data_root / "part1" / "json" / "b.json",
        {
            "image_path": "part1/images/b.png",
            "instances": [{"label": "shape", "bbox": [1, 2, 3, 4]}],
        },
    )

    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    benchmarks = client.get("/api/benchmarks", params={"split": "shape split"}).json()
    assert benchmarks["total"] == 1
    assert {item["value"] for item in benchmarks["facets"]["splits"]} == {
        "grounding_arrow",
        "shape split",
        "suite",
    }

    page = client.get(
        "/api/benchmarks/suite_bench/samples",
        params={"split": "shape split"},
    ).json()
    assert page["total"] == 1
    assert page["filters"]["split"] == "shape split"
    assert page["samples"][0]["image"] == "part1/images/b.png"
    assert (
        page["samples"][0]["image_url"]
        == "/api/benchmarks/suite_bench/samples/0/image?split=shape+split"
    )
    detail = client.get(
        "/api/benchmarks/suite_bench/samples/0",
        params={"split": "shape split"},
    ).json()
    assert detail["gt_instances"][0]["label"] == "shape"


def test_dashboard_serves_image_preview_proxy_and_pyramid_tiles(tmp_path: Path) -> None:
    from PIL import Image

    data_root = tmp_path / "benchmarks" / "multitask_val_v1" / "data"
    split_manifest = tmp_path / "benchmarks" / "multitask_val_v1" / "splits" / "val.txt"
    split_manifest.parent.mkdir(parents=True)
    split_manifest.write_text("part1/json/a.json\n", encoding="utf-8")
    image_path = data_root / "part1" / "images" / "a.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (640, 320), (20, 60, 90)).save(image_path)
    _write_json(
        tmp_path / "benchmarks" / "multitask_val_v1" / "benchmark.json",
        {
            "benchmark_id": "multitask_val_v1",
            "tasks": ["detection"],
            "layers": ["layout"],
            "split": "val",
            "sample_count": 1,
            "root": str(data_root),
            "manifest_path": str(split_manifest),
        },
    )
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "image_width": 640,
            "image_height": 320,
            "instances": [{"label": "icon", "bbox": [1, 2, 10, 20]}],
        },
    )

    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    preview_response = client.get(
        "/api/benchmarks/multitask_val_v1/samples/0/image/preview",
        params={"max_side": 256, "quality": 70},
    )
    assert preview_response.status_code == 200
    assert preview_response.headers["content-type"].startswith("image/jpeg")
    with Image.open(io.BytesIO(preview_response.content)) as preview_image:
        assert max(preview_image.size) <= 256

    tile_response = client.get("/api/benchmarks/multitask_val_v1/samples/0/image/tiles/1/0/0")
    assert tile_response.status_code == 200
    assert tile_response.headers["content-type"].startswith("image/jpeg")
    with Image.open(io.BytesIO(tile_response.content)) as tile_image:
        assert max(tile_image.size) <= 512

    assert (
        client.get("/api/benchmarks/multitask_val_v1/samples/0/image/tiles/0/99/0").status_code
        == 404
    )
    assert (tmp_path / "cache" / "image_proxy").exists()
