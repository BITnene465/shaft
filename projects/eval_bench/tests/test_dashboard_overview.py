from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval_bench.dashboard import create_app
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_dashboard_list_facets_are_not_limited_to_current_page(tmp_path: Path) -> None:
    for benchmark_id, task, layer, split in (
        ("bench_a", "detection", "layout", "val"),
        ("bench_b", "keypoint", "arrow", "test"),
    ):
        _write_json(
            tmp_path / "benchmarks" / benchmark_id / "benchmark.json",
            {
                "benchmark_id": benchmark_id,
                "tasks": [task],
                "labels": ["arrow"] if task == "keypoint" else ["icon"],
                "layers": [layer],
                "split": split,
                "sample_count": 1,
                "root": str(tmp_path / "benchmarks" / benchmark_id / "data"),
                "manifest_path": str(
                    tmp_path / "benchmarks" / benchmark_id / "splits" / f"{split}.txt"
                ),
            },
        )
    for run_id, benchmark_id, task, status, model_id, prompt_id, labels in (
        ("run_a", "bench_a", "detection", "succeeded", "model-a", "prompt-a", ["icon"]),
        ("run_b", "bench_b", "keypoint", "imported", "model-b", "prompt-b", ["arrow"]),
    ):
        _write_json(
            tmp_path / "runs" / run_id / "run.json",
            {
                "run_id": run_id,
                "status": status,
                "created_at": "2026-05-09T00:10:00Z",
                "model": {"model_id": model_id, "path": f"outputs/{model_id}"},
                "benchmark": {
                    "benchmark_id": benchmark_id,
                    "root": str(tmp_path / "benchmarks" / benchmark_id / "data"),
                    "split": "val",
                    "tasks": [task],
                },
                "spec": {
                    "task": task,
                    "prompt": {"prompt_id": prompt_id},
                    "target_labels": labels,
                    "metric_profile": (
                        "keypoint_endpoint_v1" if task == "keypoint" else "detection_iou_v1"
                    ),
                },
            },
        )
    client = TestClient(create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist"))

    benchmarks = client.get("/api/benchmarks", params={"limit": 1}).json()
    runs = client.get("/api/runs", params={"limit": 1}).json()

    assert len(benchmarks["benchmarks"]) == 1
    assert benchmarks["total"] == 2
    assert benchmarks["facets"]["tasks"] == [
        {"value": "detection", "count": 1},
        {"value": "keypoint", "count": 1},
    ]
    assert benchmarks["facets"]["layers"] == [
        {"value": "arrow", "count": 1},
        {"value": "layout", "count": 1},
    ]
    assert len(runs["runs"]) == 1
    assert runs["total"] == 2
    assert runs["facets"]["models"] == [
        {"value": "model-a", "count": 1},
        {"value": "model-b", "count": 1},
    ]
    assert runs["facets"]["prompts"] == [
        {"value": "prompt-a", "count": 1},
        {"value": "prompt-b", "count": 1},
    ]
    assert runs["facets"]["labels"] == [
        {"value": "arrow", "count": 1},
        {"value": "icon", "count": 1},
    ]


def test_dashboard_resolves_target_labels_for_agents(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "benchmarks" / "multitask_val_v1" / "benchmark.json",
        {
            "benchmark_id": "multitask_val_v1",
            "tasks": ["detection", "keypoint"],
            "layers": ["layout", "arrow"],
            "labels": ["arrow", "icon", "image", "shape"],
            "split": "val",
            "sample_count": 4,
            "root": str(tmp_path / "benchmarks" / "multitask_val_v1" / "data"),
            "manifest_path": str(
                tmp_path / "benchmarks" / "multitask_val_v1" / "splits" / "val.txt"
            ),
            "created_at": "2026-05-26T00:00:00Z",
        },
    )
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    assert client.get("/api/agent/commands").status_code == 404

    explicit = client.get(
        "/api/target-labels",
        params=[
            ("benchmark_id", "multitask_val_v1"),
            ("task", "detection"),
            ("prompt_id", "grounding_arrow.v2.4.main"),
            ("target_label", "arrow"),
        ],
    )
    assert explicit.status_code == 200
    explicit_payload = explicit.json()
    assert explicit_payload["task"] == "detection"
    assert explicit_payload["benchmark_id"] == "multitask_val_v1"
    assert explicit_payload["target_labels"] == ["arrow"]
    assert explicit_payload["target_labels_source"] == "explicit"
    assert explicit_payload["explicit_target_labels"] == ["arrow"]
    assert explicit_payload["label_subtasks_supported"] is True
    assert explicit_payload["valid"] is True
    assert explicit_payload["errors"] == []

    prompt_default = client.get(
        "/api/target-labels",
        params={
            "benchmark_id": "multitask_val_v1",
            "prompt_id": "grounding_layout.v2.4.main",
        },
    )
    assert prompt_default.status_code == 200
    prompt_payload = prompt_default.json()
    assert prompt_payload["task"] == "detection"
    assert prompt_payload["target_labels"] == ["icon", "image", "shape"]
    assert prompt_payload["target_labels_source"] == "prompt_metadata"
    assert prompt_payload["prompt_target_labels"] == ["icon", "image", "shape"]
    assert prompt_payload["candidate_labels"] == ["arrow", "icon", "image", "shape"]

    invalid_keypoint = client.get(
        "/api/target-labels",
        params=[
            ("benchmark_id", "multitask_val_v1"),
            ("task", "keypoint"),
            ("prompt_id", "keypoint_arrow.test.main"),
            ("target_label", "icon"),
        ],
    )
    assert invalid_keypoint.status_code == 200
    invalid_payload = invalid_keypoint.json()
    assert invalid_payload["target_labels"] == ["icon"]
    assert invalid_payload["label_subtasks_supported"] is False
    assert invalid_payload["valid"] is False
    assert any(
        "keypoint target_labels only support arrow" in item for item in invalid_payload["errors"]
    )

    missing_benchmark = client.get(
        "/api/target-labels",
        params={"benchmark_id": "missing", "task": "detection"},
    )
    assert missing_benchmark.status_code == 404


def test_dashboard_updates_editable_run_note(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "runs" / "run1" / "run.json",
        {
            "run_id": "run1",
            "status": "succeeded",
            "created_at": "2026-05-09T00:10:00Z",
            "model": {"model_id": "model-a", "path": "outputs/model-a/best"},
            "benchmark": {
                "benchmark_id": "bench1",
                "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
                "split": "val",
                "tasks": ["detection"],
            },
            "spec": {"task": "detection"},
        },
    )
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    empty_note = client.get("/api/runs/run1/note")
    assert empty_note.status_code == 200
    assert empty_note.json()["note"] == ""
    assert empty_note.json()["max_length"] == 20_000
    assert client.get("/api/state").json()["runs"][0]["note"] == ""

    updated = client.patch(
        "/api/runs/run1/note",
        json={"note": "reproduce with ckpt=epoch_3 and prompt v2"},
    )

    assert updated.status_code == 200
    payload = updated.json()
    assert payload["run_id"] == "run1"
    assert payload["note"] == "reproduce with ckpt=epoch_3 and prompt v2"
    assert payload["updated_at"].endswith("Z")
    note_path = tmp_path / "runs" / "run1" / "note.json"
    assert note_path.exists()
    assert json.loads(note_path.read_text(encoding="utf-8"))["note"] == payload["note"]
    state_run = client.get("/api/state").json()["runs"][0]
    assert state_run["note"] == payload["note"]
    assert state_run["note_updated_at"] == payload["updated_at"]
    assert state_run["note_max_length"] == 20_000

    appended = client.post(
        "/api/runs/run1/note/append",
        json={"heading": "follow-up", "note": "next: inspect false positives"},
    )
    assert appended.status_code == 200
    append_payload = appended.json()
    assert append_payload["note"].startswith("reproduce with ckpt=epoch_3 and prompt v2\n\n")
    assert "## follow-up\nnext: inspect false positives" in append_payload["note"]
    state_run = client.get("/api/state").json()["runs"][0]
    assert state_run["note"] == append_payload["note"]
    assert state_run["note_updated_at"] == append_payload["updated_at"]

    stale_append = client.post(
        "/api/runs/run1/note/append",
        json={
            "heading": "agent",
            "note": "stale append",
            "expected_updated_at": "2026-01-01T00:00:00Z",
        },
    )
    assert stale_append.status_code == 409
    guarded_append = client.post(
        "/api/runs/run1/note/append",
        json={
            "heading": "agent",
            "note": "guarded append",
            "expected_updated_at": append_payload["updated_at"],
        },
    )
    assert guarded_append.status_code == 200
    guarded_append_payload = guarded_append.json()
    assert "## agent\nguarded append" in guarded_append_payload["note"]

    stale = client.patch(
        "/api/runs/run1/note",
        json={"note": "stale overwrite", "expected_updated_at": "2026-01-01T00:00:00Z"},
    )
    assert stale.status_code == 409
    guarded = client.patch(
        "/api/runs/run1/note",
        json={
            "note": "guarded overwrite",
            "expected_updated_at": guarded_append_payload["updated_at"],
        },
    )
    assert guarded.status_code == 200
    assert guarded.json()["note"] == "guarded overwrite"

    bad = client.patch("/api/runs/run1/note", json={"note": ["not", "text"]})
    assert bad.status_code == 400
    bad_expected = client.patch(
        "/api/runs/run1/note",
        json={"note": "x", "expected_updated_at": ["bad"]},
    )
    assert bad_expected.status_code == 400
    bad_append_expected = client.post(
        "/api/runs/run1/note/append",
        json={"note": "x", "expected_updated_at": ["bad"]},
    )
    assert bad_append_expected.status_code == 400
    bad_append = client.post("/api/runs/run1/note/append", json={"note": "x", "heading": ["bad"]})
    assert bad_append.status_code == 400
    missing = client.patch("/api/runs/missing/note", json={"note": "x"})
    assert missing.status_code == 404
    missing_append = client.post("/api/runs/missing/note/append", json={"note": "x"})
    assert missing_append.status_code == 404


def test_dashboard_logs_http_errors_with_request_id(tmp_path: Path) -> None:
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    missing = client.get("/api/runs/missing/report")

    assert missing.status_code == 404
    assert missing.headers.get("x-eval-bench-request-id")
    backend_logs = client.get("/api/logs/backend").json()
    assert "request returned error" in backend_logs["text"]
    assert "/api/runs/missing/report" in backend_logs["text"]


def test_dashboard_run_report_supports_summary_query(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "runs" / "run1" / "reports" / "metrics.json",
        {"run_id": "run1", "kind": "metrics"},
    )
    _write_json(
        tmp_path / "runs" / "run1" / "reports" / "summary.json",
        {"run_id": "run1", "kind": "summary"},
    )
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    assert client.get("/api/runs/run1/report").json()["kind"] == "metrics"
    assert (
        client.get("/api/runs/run1/report", params={"summary": "true"}).json()["kind"] == "summary"
    )


def test_dashboard_serves_spa_fallback_when_frontend_is_built(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "icons" / "eval-bench").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>dashboard</body></html>", encoding="utf-8")
    (dist / "logo.png").write_bytes(b"logo")
    (dist / "icons" / "eval-bench" / "overview.png").write_bytes(b"icon")
    app = create_app(store_root=tmp_path / "store", frontend_dist=dist)
    client = TestClient(app)

    assert client.get("/").text == "<html><body>dashboard</body></html>"
    assert client.get("/runs").text == "<html><body>dashboard</body></html>"
    assert client.get("/logo.png").content == b"logo"
    assert client.get("/icons/eval-bench/overview.png").content == b"icon"
    assert client.get("/api/does-not-exist").status_code == 404
