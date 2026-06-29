from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval_bench.dashboard import create_app
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_dashboard_encodes_sample_image_owner_ids(tmp_path: Path) -> None:
    benchmark_id = "bench space#1"
    encoded_benchmark_id = "bench%20space%231"
    run_id = "run space#1"
    encoded_run_id = "run%20space%231"
    data_root = tmp_path / "benchmarks" / benchmark_id / "data"
    split_manifest = tmp_path / "benchmarks" / benchmark_id / "splits" / "val.txt"
    split_manifest.parent.mkdir(parents=True)
    split_manifest.write_text("part1/json/a.json\n", encoding="utf-8")
    (data_root / "part1" / "images").mkdir(parents=True)
    (data_root / "part1" / "images" / "a.png").write_bytes(b"image")
    _write_json(
        tmp_path / "benchmarks" / benchmark_id / "benchmark.json",
        {
            "benchmark_id": benchmark_id,
            "tasks": ["detection"],
            "labels": ["icon"],
            "split": "val",
            "sample_count": 1,
            "root": str(data_root),
            "manifest_path": str(split_manifest),
            "created_at": "2026-05-09T00:00:00Z",
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
    _write_json(
        tmp_path / "runs" / run_id / "run.json",
        {
            "run_id": run_id,
            "status": "succeeded",
            "created_at": "2026-05-09T00:10:00Z",
            "model": {"model_id": "model-a", "path": "outputs/model-a/best"},
            "benchmark": {
                "benchmark_id": benchmark_id,
                "root": str(data_root),
                "split": "val",
                "tasks": ["detection"],
                "manifest_path": str(split_manifest),
            },
            "spec": {"task": "detection"},
        },
    )
    _write_json(
        tmp_path / "runs" / run_id / "predictions" / "part1" / "json" / "a.json",
        {"image": "part1/images/a.png", "instances": [], "metadata": {}},
    )

    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    run_sample = client.get(f"/api/runs/{encoded_run_id}/samples").json()["samples"][0]
    assert run_sample["image_url"] == f"/api/runs/{encoded_run_id}/samples/0/image"
    assert (
        run_sample["image_preview_url"]
        == f"/api/runs/{encoded_run_id}/samples/0/image/preview?max_side=1800"
    )
    assert (
        run_sample["image_tile_url_template"]
        == f"/api/runs/{encoded_run_id}/samples/0/image/tiles/{{level}}/{{x}}/{{y}}"
    )
    assert client.get(run_sample["image_url"]).content == b"image"

    benchmark_sample = client.get(f"/api/benchmarks/{encoded_benchmark_id}/samples").json()[
        "samples"
    ][0]
    assert (
        benchmark_sample["image_url"] == f"/api/benchmarks/{encoded_benchmark_id}/samples/0/image"
    )
    assert (
        benchmark_sample["image_preview_url"]
        == f"/api/benchmarks/{encoded_benchmark_id}/samples/0/image/preview?max_side=1800"
    )
    assert (
        benchmark_sample["image_tile_url_template"]
        == f"/api/benchmarks/{encoded_benchmark_id}/samples/0/image/tiles/{{level}}/{{x}}/{{y}}"
    )
    assert client.get(benchmark_sample["image_url"]).content == b"image"
