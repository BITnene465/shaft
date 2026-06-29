from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval_bench.dashboard import create_app
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_dashboard_exposes_run_sample_detail_and_image(tmp_path: Path) -> None:
    data_root = tmp_path / "benchmarks" / "multitask_val_v1" / "data"
    split_manifest = tmp_path / "benchmarks" / "multitask_val_v1" / "splits" / "val.txt"
    split_manifest.parent.mkdir(parents=True)
    split_manifest.write_text("part1/json/a.json\n", encoding="utf-8")
    (data_root / "part1" / "images").mkdir(parents=True)
    (data_root / "part1" / "images" / "a.png").write_bytes(b"image")
    _write_json(
        data_root / "part1" / "json" / "a.json",
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
    _write_json(
        tmp_path / "runs" / "run1" / "run.json",
        {
            "run_id": "run1",
            "status": "succeeded",
            "created_at": "2026-05-09T00:10:00Z",
            "model": {"model_id": "model-a", "path": "outputs/model-a/best"},
            "benchmark": {
                "benchmark_id": "multitask_val_v1",
                "root": str(data_root),
                "split": "val",
                "tasks": ["detection", "keypoint"],
                "manifest_path": str(split_manifest),
            },
            "spec": {"task": "detection", "target_labels": ["icon"]},
        },
    )
    _write_json(
        tmp_path / "runs" / "run1" / "predictions" / "part1" / "json" / "a.json",
        {
            "image": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [2, 3, 11, 21]},
                {"label": "arrow", "bbox": [31, 31, 41, 41]},
            ],
            "metadata": {},
        },
    )
    _write_json(
        tmp_path / "runs" / "run1" / "reports" / "metrics.json",
        {
            "run_id": "run1",
            "task": "detection",
            "samples": [
                {
                    "index": 0,
                    "json_path": "part1/json/a.json",
                    "image": "part1/images/a.png",
                    "gt_instance_count": 2,
                    "pred_instance_count": 2,
                    "matched_count": 2,
                    "false_negative_count": 0,
                    "false_positive_count": 0,
                    "mean_iou": 1.0,
                    "matches": [
                        {"label": "icon", "gt_index": 0, "pred_index": 0, "iou": 1.0},
                        {"label": "arrow", "gt_index": 1, "pred_index": 1, "iou": 1.0},
                    ],
                    "false_negatives": [],
                    "false_positives": [],
                    "labels": {
                        "icon": {
                            "gt_count": 1,
                            "pred_count": 1,
                            "matched_count": 1,
                            "false_negative_count": 0,
                            "false_positive_count": 0,
                            "mean_iou": 1.0,
                        },
                        "arrow": {
                            "gt_count": 1,
                            "pred_count": 1,
                            "matched_count": 1,
                            "false_negative_count": 0,
                            "false_positive_count": 0,
                            "mean_iou": 1.0,
                        },
                    },
                }
            ],
        },
    )

    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    samples = client.get("/api/runs/run1/samples").json()["samples"]
    sample_page = client.get("/api/runs/run1/samples").json()
    assert sample_page["total"] == 1
    assert sample_page["labels"] == ["icon"]
    filtered_run_samples = client.get(
        "/api/runs/run1/samples",
        params={"label": "icon"},
    ).json()
    assert filtered_run_samples["total"] == 1
    assert filtered_run_samples["filters"] == {
        "run_id": "run1",
        "label": "icon",
        "error_filter": "all",
    }
    assert client.get("/api/runs/run1/samples", params={"label": "arrow"}).json()["total"] == 0
    assert samples[0]["gt_instance_count"] == 1
    assert samples[0]["pred_instance_count"] == 1
    assert samples[0]["labels"] == ["icon"]
    assert samples[0]["diagnostics"]["matched_count"] == 1
    assert list(samples[0]["diagnostics"]["labels"]) == ["icon"]
    assert samples[0]["image_url"] == "/api/runs/run1/samples/0/image"
    assert samples[0]["image_preview_url"] == "/api/runs/run1/samples/0/image/preview?max_side=1800"
    assert (
        samples[0]["image_tile_url_template"]
        == "/api/runs/run1/samples/0/image/tiles/{level}/{x}/{y}"
    )
    assert samples[0]["image_tile_size"] == 512

    detail = client.get("/api/runs/run1/samples/0").json()
    assert detail["gt_instances"][0]["bbox"] == [1.0, 2.0, 10.0, 20.0]
    assert [item["label"] for item in detail["gt_instances"]] == ["icon"]
    assert [item["label"] for item in detail["raw_payload"]["instances"]] == ["icon"]
    assert detail["pred_instances"][0]["bbox"] == [2.0, 3.0, 11.0, 21.0]
    assert [item["label"] for item in detail["pred_instances"]] == ["icon"]
    assert [item["label"] for item in detail["prediction_payload"]["instances"]] == ["icon"]
    assert detail["diagnostics"]["matched_count"] == 1
    assert detail["diagnostics"]["matches"] == [
        {"label": "icon", "gt_index": 0, "pred_index": 0, "iou": 1.0}
    ]
    assert client.get("/api/runs/run1/samples/0/image").content == b"image"
