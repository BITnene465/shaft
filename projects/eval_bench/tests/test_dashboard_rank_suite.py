from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval_bench.dashboard import create_app
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_dashboard_exposes_independent_rank_board(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "benchmark_type": "official",
            "tasks": ["detection"],
            "layers": ["layout", "arrow"],
            "labels": ["icon", "arrow"],
            "split": "suite",
            "sample_count": 4,
            "sample_counts": {"grounding_layout": 2, "grounding_arrow": 2},
            "split_manifests": {
                "grounding_layout": str(
                    tmp_path / "benchmarks" / "bench1" / "splits" / "grounding_layout.txt"
                ),
                "grounding_arrow": str(
                    tmp_path / "benchmarks" / "bench1" / "splits" / "grounding_arrow.txt"
                ),
            },
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "suite.txt"),
            "created_at": "2026-05-09T00:00:00Z",
        },
    )
    for run_id, label, split, precision, recall, note in (
        ("run_a", "icon", "grounding_layout", 0.9, 0.8, "layout idea"),
        ("run_b", "arrow", "grounding_arrow", 0.4, 0.5, "arrow baseline"),
    ):
        _write_json(
            tmp_path / "runs" / run_id / "run.json",
            {
                "run_id": run_id,
                "status": "succeeded",
                "created_at": f"2026-05-09T00:1{0 if run_id == 'run_a' else 1}:00Z",
                "model": {"model_id": f"model-{run_id[-1]}", "path": "outputs/model/best"},
                "benchmark": {
                    "benchmark_id": "bench1",
                    "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
                    "split": split,
                    "tasks": ["detection"],
                },
                "spec": {
                    "task": "detection",
                    "metric_profile": "detection_iou_v1",
                    "prompt": {"prompt_id": "grounding_layout.v2.4.main"},
                    "target_labels": [label],
                },
            },
        )
        _write_json(
            tmp_path / "runs" / run_id / "reports" / "summary.json",
            {
                "precision_iou50": precision,
                "recall_iou50": recall,
                "mean_iou": 0.7,
                "prediction_file_count": 2,
            },
        )
        _write_json(
            tmp_path / "runs" / run_id / "note.json",
            {"run_id": run_id, "note": note, "updated_at": "2026-05-09T01:00:00Z"},
        )
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    board = client.get("/api/rank-board").json()

    assert board["total"] == 2
    assert board["evaluated_count"] == 2
    assert board["primary_metric"] == "f1_iou50"
    assert board["primary_metric_label"] == "F1@.50"
    assert board["sort_by"] == "f1_iou50"
    assert board["sort_order"] == "desc"
    assert board["score_label"] == "F1@.50"
    assert [entry["run_id"] for entry in board["entries"]] == ["run_a", "run_b"]
    assert board["entries"][0]["rank"] == 1
    assert board["entries"][0]["f1_iou50"] == pytest.approx(0.8470588235)
    assert board["entries"][0]["score"] == pytest.approx(0.8470588235)
    assert board["entries"][0]["score_delta"] == pytest.approx(0.0)
    assert board["entries"][0]["score"] > board["entries"][1]["score"]
    assert board["entries"][1]["score_delta"] < 0
    assert board["entries"][0]["note"] == "layout idea"
    assert board["entries"][0]["target_labels"] == ["icon"]
    assert board["entries"][0]["benchmark_split"] == "grounding_layout"
    assert board["facets"]["splits"] == [
        {"value": "grounding_arrow", "count": 1},
        {"value": "grounding_layout", "count": 1},
    ]
    assert board["facets"]["labels"][0] == {"value": "arrow", "count": 1}
    assert board["facets"]["metric_profiles"][0] == {
        "value": "detection_iou_v1",
        "count": 2,
    }

    runs = client.get("/api/runs", params={"benchmark_split": "grounding_layout"}).json()
    assert runs["total"] == 1
    assert runs["filters"]["benchmark_split"] == "grounding_layout"
    assert runs["runs"][0]["run_id"] == "run_a"

    split_board = client.get(
        "/api/rank-board",
        params={"benchmark_split": "grounding_arrow"},
    ).json()
    assert split_board["total"] == 1
    assert split_board["filters"]["benchmark_split"] == "grounding_arrow"
    assert split_board["entries"][0]["run_id"] == "run_b"

    paged = client.get("/api/rank-board", params={"offset": 1, "limit": 1}).json()
    assert paged["offset"] == 1
    assert paged["limit"] == 1
    assert paged["total"] == 2
    assert [entry["run_id"] for entry in paged["entries"]] == ["run_b"]
    assert paged["entries"][0]["rank"] == 2
    assert paged["entries"][0]["score_delta"] == pytest.approx(board["entries"][1]["score_delta"])

    arrow_board = client.get("/api/rank-board", params={"label": "arrow"}).json()
    assert arrow_board["total"] == 1
    assert arrow_board["entries"][0]["run_id"] == "run_b"

    high_score_board = client.get("/api/rank-board", params={"min_score": 0.8}).json()
    assert high_score_board["total"] == 1
    assert high_score_board["filters"]["min_score"] == "0.8"
    assert high_score_board["entries"][0]["run_id"] == "run_a"

    searched = client.get("/api/rank-board", params={"query": "layout idea"}).json()
    assert searched["total"] == 1
    assert searched["entries"][0]["run_id"] == "run_a"

    recall_ascending = client.get(
        "/api/rank-board",
        params={
            "sort_by": "recall_iou50",
            "sort_order": "asc",
            "metric_profile": "detection_iou_v1",
        },
    ).json()
    assert recall_ascending["sort_by"] == "recall_iou50"
    assert recall_ascending["sort_order"] == "asc"
    assert recall_ascending["primary_metric"] == "recall_iou50"
    assert recall_ascending["primary_metric_label"] == "R@.50"
    assert recall_ascending["score_label"] == "R@.50"
    assert [entry["run_id"] for entry in recall_ascending["entries"]] == ["run_b", "run_a"]
    assert recall_ascending["entries"][0]["score"] == pytest.approx(0.5)
    assert recall_ascending["entries"][0]["score_delta"] == pytest.approx(0.0)
    assert recall_ascending["entries"][1]["score_delta"] == pytest.approx(0.3)


def test_dashboard_exposes_suite_campaign_and_integrity(tmp_path: Path) -> None:
    official_root = tmp_path / "benchmarks" / "official_suite"
    (official_root / "splits").mkdir(parents=True)
    (official_root / "splits" / "grounding_layout.txt").write_text(
        "part1/json/layout_a.json\npart1/json/layout_b.json\n",
        encoding="utf-8",
    )
    (official_root / "splits" / "grounding_arrow.txt").write_text(
        "part1/json/arrow_a.json\npart1/json/arrow_b.json\n",
        encoding="utf-8",
    )
    (official_root / "splits" / "suite.txt").write_text(
        "part1/json/layout_a.json\npart1/json/layout_b.json\n"
        "part1/json/arrow_a.json\npart1/json/arrow_b.json\n",
        encoding="utf-8",
    )
    _write_json(
        official_root / "benchmark.json",
        {
            "benchmark_id": "official_suite",
            "benchmark_type": "official",
            "tasks": ["detection"],
            "layers": ["layout", "arrow"],
            "labels": ["icon", "arrow"],
            "split": "suite",
            "sample_count": 4,
            "sample_counts": {"grounding_layout": 2, "grounding_arrow": 2},
            "split_manifests": {
                "grounding_layout": str(official_root / "splits" / "grounding_layout.txt"),
                "grounding_arrow": str(official_root / "splits" / "grounding_arrow.txt"),
            },
            "root": str(official_root / "data"),
            "manifest_path": str(official_root / "splits" / "suite.txt"),
            "created_at": "2026-05-09T00:00:00Z",
            "metadata": {"version": "v2.4"},
        },
    )
    _write_json(
        tmp_path / "benchmarks" / "tmp_bench" / "benchmark.json",
        {
            "benchmark_id": "tmp_bench",
            "benchmark_type": "temporary",
            "tasks": ["detection"],
            "split": "grounding_layout",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "tmp_bench" / "data"),
            "manifest_path": str(
                tmp_path / "benchmarks" / "tmp_bench" / "splits" / "grounding_layout.txt"
            ),
        },
    )
    for run_id, benchmark_id, split, precision in (
        ("layout_run", "official_suite", "grounding_layout", 0.9),
        ("arrow_run", "official_suite", "grounding_arrow", 0.7),
        ("tmp_run", "tmp_bench", "grounding_layout", 0.99),
        ("orphan_run", "missing_bench", "grounding_layout", 0.88),
    ):
        _write_json(
            tmp_path / "runs" / run_id / "run.json",
            {
                "run_id": run_id,
                "status": "succeeded",
                "created_at": f"2026-05-09T00:0{len(run_id)}:00Z",
                "model": {"model_id": "model-a", "path": "outputs/model-a/ckpt-100"},
                "benchmark": {
                    "benchmark_id": benchmark_id,
                    "root": str(tmp_path / "benchmarks" / benchmark_id / "data"),
                    "split": split,
                    "tasks": ["detection"],
                },
                "spec": {
                    "task": "detection",
                    "metric_profile": "detection_iou_v1",
                    "prompt": {"prompt_id": f"{split}.main"},
                    "target_labels": ["icon"],
                    "inference": {
                        "max_pixels": 2_000_000,
                        "max_tokens": 2048,
                        "temperature": 0.0,
                        "top_p": 1.0,
                    },
                },
            },
        )
        _write_json(
            tmp_path / "runs" / run_id / "reports" / "summary.json",
            {
                "precision_iou50": precision,
                "recall_iou50": precision,
                "mean_iou": precision,
                "prediction_file_count": 2,
            },
        )

    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    benchmarks = client.get("/api/benchmarks").json()
    assert benchmarks["total"] == 1
    assert benchmarks["benchmarks"][0]["benchmark_id"] == "official_suite"
    assert benchmarks["benchmarks"][0]["benchmark_type"] == "official"

    runs = client.get("/api/runs").json()["runs"]
    integrity = {run["run_id"]: run["integrity_status"] for run in runs}
    assert integrity["layout_run"] == "ok"
    assert integrity["tmp_run"] == "non_official_benchmark"
    assert integrity["orphan_run"] == "missing_benchmark"

    suites = client.get("/api/suites").json()
    assert suites["total"] == 1
    suite = suites["suites"][0]
    assert suite["suite_id"] == "official_suite"
    assert suite["version"] == "v2.4"
    assert suite["integrity_status"] == "ok"
    assert suite["validation_errors"] == []
    assert [item["split"] for item in suite["task_splits"]] == [
        "grounding_arrow",
        "grounding_layout",
    ]
    assert suite["sample_universe"]["sample_count"] == 4

    campaigns = client.get("/api/campaigns").json()
    assert campaigns["total"] == 1
    campaign = campaigns["campaigns"][0]
    assert campaign["suite_id"] == "official_suite"
    assert campaign["model_id"] == "model-a"
    assert campaign["pixel_budget"] == 2_000_000
    assert campaign["task_splits"] == ["grounding_arrow", "grounding_layout"]
    assert campaign["aggregate_report"]["f1_iou50"] == pytest.approx(0.8)

    suite_board = client.get("/api/suite-rank-board").json()
    assert suite_board["total"] == 1
    assert suite_board["evaluated_count"] == 1
    assert suite_board["primary_metric"] == "aggregate_score"
    assert suite_board["entries"][0]["campaign_id"] == campaign["campaign_id"]
    assert suite_board["entries"][0]["aggregate_score"] == pytest.approx(0.8)
    assert suite_board["entries"][0]["task_splits"] == ["grounding_arrow", "grounding_layout"]

    filtered_suite_board = client.get(
        "/api/suite-rank-board",
        params={"suite_id": "official_suite", "model_id": "model-a"},
    ).json()
    assert filtered_suite_board["total"] == 1
    assert filtered_suite_board["filters"]["suite_id"] == "official_suite"

    board = client.get("/api/rank-board").json()
    assert board["total"] == 2
    assert {entry["run_id"] for entry in board["entries"]} == {"layout_run", "arrow_run"}
    assert all(entry["benchmark_type"] == "official" for entry in board["entries"])


def test_dashboard_marks_invalid_official_suite_manifest(tmp_path: Path) -> None:
    split_manifest = tmp_path / "benchmarks" / "bench1" / "splits" / "grounding_layout.txt"
    split_manifest.parent.mkdir(parents=True)
    split_manifest.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "suites" / "broken_suite" / "suite.json",
        {
            "suite_id": "broken_suite",
            "version": "v1",
            "benchmark_id": "bench1",
            "benchmark_type": "official",
            "official": True,
            "metric_profile": "detection_iou_v1",
            "sample_universe": {"sample_count": 2},
            "task_splits": [
                {
                    "split": "grounding_layout",
                    "benchmark_id": "bench1",
                    "manifest_path": str(split_manifest),
                    "sample_count": 2,
                    "tasks": ["detection"],
                    "layers": ["layout"],
                    "target_labels": ["icon"],
                }
            ],
        },
    )
    _write_json(
        tmp_path / "campaigns" / "bad_campaign" / "campaign.json",
        {
            "campaign_id": "bad_campaign",
            "suite_id": "broken_suite",
            "model_id": "model-a",
            "checkpoint": "outputs/model-a/ckpt-100",
            "prompt_set": ["grounding_layout.main"],
            "pixel_budget": 2_000_000,
            "decoding_config": {"temperature": 0.0},
            "run_ids": ["layout_run"],
            "task_splits": ["grounding_layout"],
            "aggregate_report": {"f1_iou50": 0.9},
        },
    )

    client = TestClient(create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist"))

    suites = client.get("/api/suites").json()
    assert suites["total"] == 1
    suite = suites["suites"][0]
    assert suite["suite_id"] == "broken_suite"
    assert suite["integrity_status"] == "sample_count_mismatch"
    assert "sample_count=2" in suite["integrity_reason"]
    assert suite["validation_errors"]

    suite_board = client.get("/api/suite-rank-board").json()
    assert suite_board["total"] == 0
