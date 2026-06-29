from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval_bench.dashboard import create_app
from eval_bench.database import EvalBenchDatabase
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_dashboard_imports_prediction_snapshot_and_evaluates(tmp_path: Path) -> None:
    data_root = tmp_path / "benchmarks" / "multitask_val_v1" / "data"
    split_manifest = tmp_path / "benchmarks" / "multitask_val_v1" / "splits" / "val.txt"
    prediction_root = tmp_path / "external_predictions"
    split_manifest.parent.mkdir(parents=True)
    split_manifest.write_text("part1/json/a.json\n", encoding="utf-8")
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
            "image_width": 100,
            "image_height": 50,
            "instances": [{"label": "icon", "bbox": [10, 10, 40, 40]}],
        },
    )
    _write_json(
        prediction_root / "part1" / "json" / "a.json",
        {
            "image": "part1/images/a.png",
            "instances": [{"label": "icon", "bbox": [11, 11, 41, 41], "score": 0.9}],
        },
    )

    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)
    response = client.post(
        "/api/runs/import-predictions",
        json={
            "run_id": "imported_run",
            "benchmark_id": "multitask_val_v1",
            "prediction_root": str(prediction_root),
            "task": "detection",
            "model_id": "model-a",
            "prompt_id": "grounding_layout.v2.4.main",
            "target_labels": ["icon"],
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["run_id"] == "imported_run"
    assert payload["imported_predictions"] == 1
    assert payload["missing_prediction_count"] == 0
    assert payload["report_path"].endswith("metrics.json")
    assert payload["summary_path"].endswith("summary.json")
    assert Path(payload["summary_path"]).exists()
    state_run = client.get("/api/state").json()["runs"][0]
    assert state_run["run_id"] == "imported_run"
    assert state_run["target_labels"] == ["icon"]
    detail = client.get("/api/runs/imported_run/samples/0").json()
    assert detail["diagnostics"]["matched_count"] == 1
    assert detail["sample"]["labels"] == ["icon"]
    assert detail["pred_instances"][0]["score"] == 0.9
    evaluated = client.post("/api/runs/imported_run/evaluate")
    assert evaluated.status_code == 200
    evaluated_payload = evaluated.json()
    assert evaluated_payload["run_id"] == "imported_run"
    assert evaluated_payload["report_path"].endswith("metrics.json")
    assert evaluated_payload["summary_path"].endswith("summary.json")
    assert Path(evaluated_payload["summary_path"]).exists()

    conflict = client.post(
        "/api/runs/import-predictions",
        json={
            "run_id": "imported_run",
            "benchmark_id": "multitask_val_v1",
            "prediction_root": str(prediction_root),
            "task": "detection",
            "model_id": "model-a",
        },
    )
    assert conflict.status_code == 409


def test_dashboard_import_predictions_uses_prompt_template_target_labels(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "benchmarks" / "multitask_val_v1" / "data"
    split_manifest = tmp_path / "benchmarks" / "multitask_val_v1" / "splits" / "val.txt"
    prediction_root = tmp_path / "external_predictions"
    split_manifest.parent.mkdir(parents=True)
    split_manifest.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "multitask_val_v1" / "benchmark.json",
        {
            "benchmark_id": "multitask_val_v1",
            "tasks": ["detection"],
            "layers": ["layout"],
            "labels": ["custom_arrow", "icon"],
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
            "instances": [
                {"label": "icon", "bbox": [10, 10, 40, 40]},
                {"label": "custom_arrow", "bbox": [50, 10, 80, 40]},
            ],
        },
    )
    _write_json(
        prediction_root / "part1" / "json" / "a.json",
        {
            "image": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [11, 11, 41, 41], "score": 0.9},
                {"label": "custom_arrow", "bbox": [51, 11, 81, 41], "score": 0.8},
            ],
        },
    )
    EvalBenchDatabase(tmp_path).upsert_prompt_template(
        {
            "prompt_id": "custom.arrow.import",
            "label": "Custom arrow import",
            "task": "detection",
            "system_prompt": "Inspect diagrams.",
            "user_prompt": "Find custom arrows.",
            "metadata": {"target_labels": ["custom_arrow"]},
        }
    )

    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)
    response = client.post(
        "/api/runs/import-predictions",
        json={
            "run_id": "imported_custom_arrow",
            "benchmark_id": "multitask_val_v1",
            "prediction_root": str(prediction_root),
            "task": "detection",
            "model_id": "model-a",
            "prompt_id": "custom.arrow.import",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["run_id"] == "imported_custom_arrow"
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    assert report["target_labels"] == ["custom_arrow"]
    assert report["target_labels_source"] == "prompt_metadata"
    state_run = client.get("/api/state").json()["runs"][0]
    assert state_run["target_labels"] == ["custom_arrow"]
    detail = client.get("/api/runs/imported_custom_arrow/samples/0").json()
    assert detail["sample"]["labels"] == ["custom_arrow"]
    assert [item["label"] for item in detail["pred_instances"]] == ["custom_arrow"]


def test_dashboard_exposes_pairwise_comparison(tmp_path: Path) -> None:
    data_root = tmp_path / "benchmarks" / "multitask_val_v1" / "data"
    split_manifest = tmp_path / "benchmarks" / "multitask_val_v1" / "splits" / "val.txt"
    split_manifest.parent.mkdir(parents=True)
    split_manifest.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "image_width": 100,
            "image_height": 50,
            "instances": [{"label": "icon", "bbox": [1, 2, 10, 20]}],
        },
    )
    for run_id, recall in (("baseline", 0.0), ("candidate", 1.0)):
        _write_json(
            tmp_path / "runs" / run_id / "run.json",
            {
                "run_id": run_id,
                "status": "succeeded",
                "created_at": "2026-05-09T00:10:00Z",
                "model": {"model_id": run_id, "path": f"outputs/{run_id}/best"},
                "benchmark": {
                    "benchmark_id": "multitask_val_v1",
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
            {
                "image": "part1/images/a.png",
                "instances": [{"label": "icon", "bbox": [2 + recall, 3, 11, 21]}],
                "metadata": {},
            },
        )
        _write_json(
            tmp_path / "runs" / run_id / "reports" / "metrics.json",
            {
                "run_id": run_id,
                "benchmark_id": "multitask_val_v1",
                "benchmark_split": "val",
                "task": "detection",
                "metric_profile": "detection_iou_v1",
                "target_labels": ["icon"],
                "target_labels_source": "explicit",
                "precision_iou50": recall,
                "recall_iou50": recall,
                "mean_iou": recall,
                "matched_count": int(recall),
                "gt_instance_count": 1,
                "pred_instance_count": 1,
                "samples": [
                    {
                        "index": 0,
                        "json_path": "part1/json/a.json",
                        "image": "part1/images/a.png",
                        "matched_count": int(recall),
                        "false_positive_count": 0,
                        "false_negative_count": 1 - int(recall),
                        "mean_iou": recall,
                        "labels": {
                            "icon": {
                                "matched_count": int(recall),
                                "false_positive_count": 0,
                                "false_negative_count": 1 - int(recall),
                                "mean_iou": recall,
                            }
                        },
                    }
                ],
            },
        )

    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)
    response = client.get(
        "/api/comparisons",
        params={"baseline_run_id": "baseline", "candidate_run_id": "candidate"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["delta"]["recall_iou50"] == 1.0
    assert payload["summary"]["improved_samples"] == 1
    assert payload["top_improvements"][0]["candidate_index"] == 0

    listed = client.get("/api/comparisons").json()
    assert listed["comparisons"][0]["baseline_run_id"] == "baseline"
    assert listed["comparisons"][0]["candidate_run_id"] == "candidate"
    assert listed["total"] == 1
    detail = client.get("/api/comparisons/baseline__vs__candidate")
    assert detail.status_code == 200
    assert detail.json()["comparison_id"] == "baseline__vs__candidate"
    assert client.get("/api/comparisons/not_found").status_code == 404

    filtered = client.get(
        "/api/comparisons",
        params={
            "list": "1",
            "task": "detection",
            "benchmark_id": "multitask_val_v1",
            "benchmark_split": "val",
            "baseline_run_id": "baseline",
            "candidate_run_id": "candidate",
            "label": "icon",
            "query": "candidate",
        },
    ).json()
    assert filtered["filters"] == {
        "task": "detection",
        "benchmark_id": "multitask_val_v1",
        "benchmark_split": "val",
        "label": "icon",
        "query": "candidate",
        "baseline_run_id": "baseline",
        "candidate_run_id": "candidate",
    }
    assert filtered["total"] == 1
    assert filtered["comparisons"][0]["metric_profile"] == "detection_iou_v1"
    assert filtered["comparisons"][0]["target_labels"] == ["icon"]

    filtered_wrong_pair = client.get(
        "/api/comparisons",
        params={"list": "1", "baseline_run_id": "other", "candidate_run_id": "candidate"},
    ).json()
    assert filtered_wrong_pair["total"] == 0
    assert filtered_wrong_pair["filters"]["baseline_run_id"] == "other"

    filtered_empty = client.get("/api/comparisons", params={"label": "arrow"}).json()
    assert filtered_empty["total"] == 0
    assert filtered_empty["comparisons"] == []

    sample_detail = client.get(
        "/api/comparisons/sample",
        params={
            "baseline_run_id": "baseline",
            "candidate_run_id": "candidate",
            "sample_index": 0,
        },
    )
    assert sample_detail.status_code == 200
    sample_payload = sample_detail.json()
    assert sample_payload["baseline"]["sample"]["image_url"] == "/api/runs/baseline/samples/0/image"
    assert (
        sample_payload["candidate"]["sample"]["image_url"] == "/api/runs/candidate/samples/0/image"
    )
    assert sample_payload["baseline"]["gt_instances"][0]["label"] == "icon"
    assert sample_payload["candidate"]["pred_instances"][0]["bbox"] == [3.0, 3.0, 11.0, 21.0]

    bad_request = client.get("/api/comparisons", params={"baseline_run_id": "baseline"})
    assert bad_request.status_code == 400

    second_report = client.get(
        "/api/comparisons",
        params={"baseline_run_id": "candidate", "candidate_run_id": "baseline"},
    )
    assert second_report.status_code == 200
    all_history = client.get("/api/comparisons", params={"list": "1", "limit": 2}).json()
    paged_history = client.get(
        "/api/comparisons",
        params={"list": "1", "offset": 1, "limit": 1},
    ).json()
    assert all_history["total"] == 2
    assert paged_history["offset"] == 1
    assert paged_history["limit"] == 1
    assert paged_history["comparisons"] == all_history["comparisons"][1:2]
