from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval_bench.dashboard import create_app
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_dashboard_exposes_composite_layout_arrow_sample_view(tmp_path: Path) -> None:
    data_root = tmp_path / "benchmarks" / "suite1" / "data"
    split_dir = tmp_path / "benchmarks" / "suite1" / "splits"
    layout_manifest = split_dir / "grounding_layout.txt"
    arrow_manifest = split_dir / "grounding_arrow.txt"
    split_dir.mkdir(parents=True)
    layout_manifest.write_text("part1/json/a.json\npart1/json/b.json\n", encoding="utf-8")
    arrow_manifest.write_text("part1/json/a.json\n", encoding="utf-8")
    (data_root / "part1" / "images").mkdir(parents=True)
    (data_root / "part1" / "images" / "a.png").write_bytes(b"image")
    (data_root / "part1" / "images" / "b.png").write_bytes(b"image")
    for stem in ("a", "b"):
        _write_json(
            data_root / "part1" / "json" / f"{stem}.json",
            {
                "image_path": f"part1/images/{stem}.png",
                "image_width": 100,
                "image_height": 50,
                "instances": [
                    {"label": "icon", "bbox": [1, 2, 10, 20]},
                    {"label": "arrow", "bbox": [30, 30, 40, 40]},
                ],
            },
        )
    for run_id, split, label, manifest_path in (
        ("layout_run", "grounding_layout", "icon", layout_manifest),
        ("arrow_run", "grounding_arrow", "arrow", arrow_manifest),
    ):
        _write_json(
            tmp_path / "runs" / run_id / "run.json",
            {
                "run_id": run_id,
                "status": "succeeded",
                "created_at": "2026-05-09T00:10:00Z",
                "model": {"model_id": "model-a", "path": "outputs/model-a/best"},
                "benchmark": {
                    "benchmark_id": "suite1",
                    "root": str(data_root),
                    "split": split,
                    "tasks": ["detection"],
                    "manifest_path": str(manifest_path),
                },
                "spec": {"task": "detection", "target_labels": [label]},
            },
        )
        _write_json(
            tmp_path / "runs" / run_id / "predictions" / "part1" / "json" / "a.json",
            {
                "image": "part1/images/a.png",
                "instances": [{"label": label, "bbox": [2, 3, 11, 21]}],
                "metadata": {},
            },
        )
        if run_id == "layout_run":
            _write_json(
                tmp_path / "runs" / run_id / "predictions" / "part1" / "json" / "b.json",
                {
                    "image": "part1/images/b.png",
                    "instances": [{"label": label, "bbox": [2, 3, 11, 21]}],
                    "metadata": {},
                },
            )
        _write_json(
            tmp_path / "runs" / run_id / "reports" / "metrics.json",
            {
                "run_id": run_id,
                "task": "detection",
                "samples": [
                    {
                        "index": 0,
                        "json_path": "part1/json/a.json",
                        "image": "part1/images/a.png",
                        "gt_instance_count": 1,
                        "pred_instance_count": 1,
                        "matched_count": 1,
                        "false_negative_count": 0,
                        "false_positive_count": 0,
                        "mean_iou": 1.0,
                        "matches": [{"label": label, "gt_index": 0, "pred_index": 0, "iou": 1.0}],
                        "false_negatives": [],
                        "false_positives": [],
                        "labels": {
                            label: {
                                "gt_count": 1,
                                "pred_count": 1,
                                "matched_count": 1,
                                "false_negative_count": 0,
                                "false_positive_count": 0,
                                "mean_iou": 1.0,
                            }
                        },
                    }
                ]
                + (
                    [
                        {
                            "index": 1,
                            "json_path": "part1/json/b.json",
                            "image": "part1/images/b.png",
                            "gt_instance_count": 1,
                            "pred_instance_count": 1,
                            "matched_count": 1,
                            "false_negative_count": 0,
                            "false_positive_count": 0,
                            "mean_iou": 1.0,
                            "matches": [
                                {"label": label, "gt_index": 0, "pred_index": 0, "iou": 1.0}
                            ],
                            "false_negatives": [],
                            "false_positives": [],
                            "labels": {},
                        }
                    ]
                    if run_id == "layout_run"
                    else []
                ),
            },
        )

    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    detail = client.get(
        "/api/composite-samples",
        params={
            "sample_index": 0,
            "layout_run_id": "layout_run",
            "arrow_run_id": "arrow_run",
        },
    ).json()

    assert detail["kind"] == "composite_sample_view"
    assert detail["image_count"] == 2
    assert detail["image_key"] == "part1/images/a.png"
    assert detail["layer_options"] == ["layout", "arrow"]
    assert detail["view_modes"] == ["gt", "prediction", "diff"]
    assert detail["diagnostics"]["warnings"] == []
    assert detail["diagnostics"]["per_layer"]["layout"]["matched_count"] == 1
    assert [layer["layer"] for layer in detail["layers"]] == ["layout", "arrow"]
    assert detail["layers"][0]["benchmark_split"] == "grounding_layout"
    assert detail["layers"][0]["sample"]["image_url"] == "/api/runs/layout_run/samples/0/image"
    assert [item["label"] for item in detail["layers"][0]["gt_instances"]] == ["icon"]
    assert [item["label"] for item in detail["layers"][1]["gt_instances"]] == ["arrow"]

    generic_detail = client.get(
        "/api/composite-samples",
        params=[
            ("sample_index", "0"),
            ("layer_run", "layout:layout_run"),
            ("layer_run", "arrow_overlay:arrow_run"),
        ],
    ).json()
    assert generic_detail["layer_options"] == ["layout", "arrow_overlay"]
    assert [layer["run_id"] for layer in generic_detail["layers"]] == ["layout_run", "arrow_run"]

    union_detail = client.get(
        "/api/composite-samples",
        params=[
            ("sample_index", "1"),
            ("layer_run", "layout:layout_run"),
            ("layer_run", "arrow:arrow_run"),
        ],
    ).json()
    assert union_detail["image_key"] == "part1/images/b.png"
    assert [layer["layer"] for layer in union_detail["layers"]] == ["layout"]
    assert {item["layer"]: item["status"] for item in union_detail["layer_statuses"]} == {
        "layout": "ready",
        "arrow": "image_missing",
    }

    missing_layer = client.get(
        "/api/composite-samples",
        params={"sample_index": 0, "layout_run_id": "layout_run"},
    )
    assert missing_layer.status_code == 400
