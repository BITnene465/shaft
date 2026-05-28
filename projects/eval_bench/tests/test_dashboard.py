from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from eval_bench import services as services_module
from eval_bench.dashboard import create_app
from eval_bench.database import EvalBenchDatabase
from eval_bench.prompt_templates import DEFAULT_PROMPT_SPECS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _eval_job_payload(
    *,
    model_id: str,
    model_path: str,
    benchmark_id: str,
    task: str,
    prompt_id: str,
    backend: str = "vllm_openai",
    runtime_mode: str = "ephemeral",
    benchmark_split: str | None = None,
    target_labels: list[str] | None = None,
    max_tokens: int | None = None,
) -> dict:
    return {
        "manifest": {
            "kind": "eval_job",
            "runtime": {
                "mode": runtime_mode,
                "engine": backend,
                "args": {
                    "model": model_path,
                    "served-model-name": model_id,
                    "host": "127.0.0.1",
                },
            },
            "eval": {
                "model_id": model_id,
                "benchmark_id": benchmark_id,
                "benchmark_split": benchmark_split or "",
                "task": task,
                "prompt_id": prompt_id,
                "target_labels": list(target_labels or []),
                "generation": {"max_tokens": max_tokens} if max_tokens is not None else {},
            },
        }
    }


def _wait_for_job_status(client: TestClient, job_id: str, status: str) -> dict:
    deadline = time.monotonic() + 5
    latest: dict | None = None
    while time.monotonic() < deadline:
        jobs = client.get("/api/jobs").json()["jobs"]
        latest = next((job for job in jobs if job["job_id"] == job_id), None)
        if latest and latest["status"] == status:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach status={status!r}: {latest}")


def _wait_for_process_exit(process: subprocess.Popen[bytes], *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.05)
    return process.poll() is not None


def test_dashboard_api_exposes_store_state(tmp_path: Path) -> None:
    benchmark_manifest = tmp_path / "benchmarks" / "multitask_val_v1" / "benchmark.json"
    split_manifest = tmp_path / "benchmarks" / "multitask_val_v1" / "splits" / "val.txt"
    split_manifest.parent.mkdir(parents=True)
    split_manifest.write_text("part1/json/a.json\npart1/json/b.json\n", encoding="utf-8")
    _write_json(
        benchmark_manifest,
        {
            "benchmark_id": "multitask_val_v1",
            "tasks": ["detection", "keypoint"],
            "layers": ["layout", "arrow"],
            "labels": ["arrow", "icon"],
            "split": "val",
            "sample_count": 2,
            "root": str(tmp_path / "benchmarks" / "multitask_val_v1" / "data"),
            "manifest_path": str(split_manifest),
            "created_at": "2026-05-09T00:00:00Z",
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
                "root": str(tmp_path / "benchmarks" / "multitask_val_v1" / "data"),
                "split": "val",
                "tasks": ["detection", "keypoint"],
            },
            "spec": {"task": "detection"},
        },
    )
    _write_json(
        tmp_path / "runs" / "run1" / "predictions" / "part1" / "json" / "a.json",
        {"image": "part1/images/a.png", "instances": [], "metadata": {}},
    )
    for stem in ("a", "b"):
        _write_json(
            tmp_path / "benchmarks" / "multitask_val_v1" / "data" / "part1" / "json" / f"{stem}.json",
            {
                "image_path": f"part1/images/{stem}.png",
                "image_width": 100,
                "image_height": 50,
                "instances": [],
            },
        )

    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    health = client.get("/api/health").json()
    assert health["ok"] is True
    assert health["frontend_built"] is False
    assert health["scheduler_enabled"] is False
    assert client.get("/api/scheduler/status").json()["enabled"] is False

    initial_logs = client.get("/api/logs/backend").json()
    assert initial_logs["log_path"].endswith("backend.log")
    assert isinstance(initial_logs["lines"], list)

    state = client.get("/api/state").json()
    assert state["benchmark_count"] == 1
    assert state["run_count"] == 1
    assert state["prediction_count"] == 1
    assert state["benchmarks"][0]["layers"] == ["layout", "arrow"]
    assert state["benchmarks"][0]["labels"] == ["arrow", "icon"]
    assert state["runs"][0]["model_id"] == "model-a"

    ops_summary = client.get("/api/ops-summary").json()
    assert ops_summary["source"] == "ops_summary"
    assert ops_summary["runs"]["total"] == 1
    assert ops_summary["runs"]["waiting_evaluation"] == 1
    assert ops_summary["jobs"] == {
        "total": 0,
        "queued": 0,
        "running": 0,
        "failed": 0,
        "active": 0,
    }
    assert ops_summary["services"] == {"total": 0, "running": 0}
    assert ops_summary["scheduler"]["enabled"] is False

    benchmarks = client.get("/api/benchmarks").json()
    assert benchmarks["benchmarks"][0]["benchmark_id"] == "multitask_val_v1"
    assert benchmarks["benchmarks"][0]["labels"] == ["arrow", "icon"]
    assert benchmarks["total"] == 1
    benchmark_detail = client.get("/api/benchmarks/multitask_val_v1")
    assert benchmark_detail.status_code == 200
    assert benchmark_detail.json()["benchmark"]["benchmark_id"] == "multitask_val_v1"
    assert client.get("/api/benchmarks/not_found").status_code == 404
    filtered_benchmarks = client.get(
        "/api/benchmarks",
        params={"task": "detection", "layer": "layout", "split": "val", "query": "multitask"},
    ).json()
    assert filtered_benchmarks["filters"] == {
        "task": "detection",
        "layer": "layout",
        "split": "val",
        "query": "multitask",
    }
    assert filtered_benchmarks["total"] == 1

    runs = client.get("/api/runs").json()
    assert runs["runs"][0]["run_id"] == "run1"
    assert runs["total"] == 1
    run_detail = client.get("/api/runs/run1")
    assert run_detail.status_code == 200
    assert run_detail.json()["run"]["run_id"] == "run1"
    assert client.get("/api/runs/not_found").status_code == 404
    filtered_runs = client.get(
        "/api/runs",
        params={
            "task": "detection",
            "benchmark_id": "multitask_val_v1",
            "status": "succeeded",
            "model_id": "model-a",
            "query": "model-a",
        },
    ).json()
    assert filtered_runs["filters"]["model_id"] == "model-a"
    assert filtered_runs["total"] == 1
    assert filtered_runs["runs"][0]["run_id"] == "run1"
    assert client.get("/api/jobs").json()["jobs"] == []
    templates = client.get("/api/job-templates").json()["templates"]
    assert "eval_job" in templates
    assert "preannotate_job" not in templates
    template_detail = client.get("/api/job-templates/eval_job")
    assert template_detail.status_code == 200
    assert template_detail.json()["template"]["manifest"]["kind"] == "eval_job"
    assert client.get("/api/job-templates/not_found").status_code == 404
    layout_prompt_id = str(DEFAULT_PROMPT_SPECS[1]["prompt_id"])
    prompt_templates = client.get("/api/prompt-templates").json()
    assert layout_prompt_id in prompt_templates["by_id"]
    prompt_template_detail = client.get(f"/api/prompt-templates/{layout_prompt_id}")
    assert prompt_template_detail.status_code == 200
    assert prompt_template_detail.json()["template"]["prompt_id"] == layout_prompt_id
    saved_prompt = client.post(
        "/api/prompt-templates",
        json={
            "prompt_id": "custom.layout",
            "label": "Custom Layout",
            "task": "detection",
            "system_prompt": "JSON only.",
            "user_prompt": "Detect icons.",
            "parser": "raw_data_detection_v1",
            "metric_profile": "detection_iou_v1",
            "generation": {"max_tokens": 2048},
            "data": {"max_pixels": 123456},
        },
    )
    assert saved_prompt.status_code == 201
    assert saved_prompt.json()["prompt_id"] == "custom.layout"
    custom_prompt_detail = client.get("/api/prompt-templates/custom.layout")
    assert custom_prompt_detail.status_code == 200
    assert custom_prompt_detail.json()["template"]["prompt_id"] == "custom.layout"
    assert client.get("/api/prompt-templates/not_found").status_code == 404

    model_path = tmp_path / "models" / "model-a" / "best"
    model_path.mkdir(parents=True)
    preflight = client.post(
        "/api/jobs/preflight",
        json=_eval_job_payload(
            model_id="model-a",
            model_path=str(model_path),
            benchmark_id="multitask_val_v1",
            task="detection",
            prompt_id="custom.layout",
            backend="dry_run",
            runtime_mode="external",
            max_tokens=4096,
        ),
    )
    assert preflight.status_code == 200
    assert preflight.json()["ok"] is True
    assert preflight.json()["resolved_payload"]["prompt_text"] == "Detect icons."
    bad_label_preflight = client.post(
        "/api/jobs/preflight",
        json=_eval_job_payload(
            model_id="model-a",
            model_path=str(model_path),
            benchmark_id="multitask_val_v1",
            task="detection",
            prompt_id="custom.layout",
            target_labels=["typo_label"],
            max_tokens=4096,
        ),
    )
    assert bad_label_preflight.status_code == 200
    bad_label_payload = bad_label_preflight.json()
    assert bad_label_payload["ok"] is False
    assert any(
        "target_labels not found in benchmark label index: typo_label" in item
        for item in bad_label_payload["errors"]
    )
    created = client.post(
        "/api/jobs",
        json=_eval_job_payload(
            model_id="model-a",
            model_path=str(model_path),
            benchmark_id="multitask_val_v1",
            task="detection",
            prompt_id="custom.layout",
            backend="dry_run",
            runtime_mode="external",
            max_tokens=4096,
        ),
    )
    assert created.status_code == 201
    created_payload = created.json()
    assert created_payload["kind"] == "eval"
    assert created_payload["status"] == "queued"
    assert created_payload["run_id"] == created_payload["job_id"]
    assert created_payload["payload"]["run_id"] == created_payload["run_id"]
    assert created_payload["metadata"]["run_id"] == created_payload["run_id"]
    assert created_payload["payload"]["model_id"] == "model-a"
    assert created_payload["payload"]["prompt_text"] == "Detect icons."
    jobs = client.get("/api/jobs").json()
    assert jobs["jobs"][0]["job_id"] == created_payload["job_id"]
    assert jobs["jobs"][0]["run_id"] == created_payload["run_id"]
    assert jobs["total"] == 1
    job_detail = client.get(f"/api/jobs/{created_payload['job_id']}")
    assert job_detail.status_code == 200
    assert job_detail.json()["job"]["job_id"] == created_payload["job_id"]
    assert job_detail.json()["job"]["run_id"] == created_payload["run_id"]
    assert client.get("/api/jobs/not_found").status_code == 404
    filtered_jobs = client.get(
        "/api/jobs",
        params={"kind": "eval", "status": "queued", "query": "custom.layout"},
    ).json()
    assert filtered_jobs["filters"] == {
        "kind": "eval",
        "status": "queued",
        "query": "custom.layout",
    }
    assert filtered_jobs["total"] == 1
    assert filtered_jobs["jobs"][0]["job_id"] == created_payload["job_id"]
    assert filtered_jobs["jobs"][0]["run_id"] == created_payload["run_id"]

    processed = client.post("/api/jobs/process-next")
    assert processed.status_code == 200
    processed_payload = processed.json()
    assert processed_payload["processed"] is True
    assert processed_payload["background"] is True
    assert processed_payload["job"]["status"] == "running"
    completed_job = _wait_for_job_status(client, created_payload["job_id"], "succeeded")
    assert completed_job["run_id"] == created_payload["run_id"]
    assert completed_job["metadata"]["run_id"] == created_payload["run_id"]
    assert completed_job["metadata"]["progress_phase"] == "succeeded"
    assert (tmp_path / "runs" / created_payload["run_id"] / "run.json").exists()
    assert client.get("/api/state").json()["run_count"] == 2


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
        "keypoint target_labels only support arrow" in item
        for item in invalid_payload["errors"]
    )

    missing_benchmark = client.get(
        "/api/target-labels",
        params={"benchmark_id": "missing", "task": "detection"},
    )
    assert missing_benchmark.status_code == 404


def test_dashboard_create_job_persists_preflight_warnings(tmp_path: Path) -> None:
    model_path = tmp_path / "models" / "model-a" / "best"
    model_path.mkdir(parents=True)
    (tmp_path / "benchmarks" / "bench_no_labels" / "splits").mkdir(parents=True)
    (tmp_path / "benchmarks" / "bench_no_labels" / "splits" / "val.txt").write_text(
        "part1/json/a.json\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "benchmarks" / "bench_no_labels" / "benchmark.json",
        {
            "benchmark_id": "bench_no_labels",
            "tasks": ["detection"],
            "split": "val",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "bench_no_labels" / "data"),
            "manifest_path": str(
                tmp_path / "benchmarks" / "bench_no_labels" / "splits" / "val.txt"
            ),
        },
    )
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    warning_job = client.post(
        "/api/jobs",
        json=_eval_job_payload(
            model_id="model-a",
            model_path=str(model_path),
            benchmark_id="bench_no_labels",
            task="detection",
            prompt_id="grounding_layout.v2.4.main",
            target_labels=["icon"],
            max_tokens=4096,
        ),
    )

    assert warning_job.status_code == 201
    payload = warning_job.json()
    assert payload["payload"]["target_labels"] == ["icon"]
    assert any(
        "target_labels could not be preflight-validated" in item
        for item in payload["metadata"]["preflight_warnings"]
    )


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


def test_dashboard_exposes_independent_rank_board(tmp_path: Path) -> None:
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
    assert board["score_formula"] == "F1@.50"
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
        params={"sort_by": "recall_iou50", "sort_order": "asc", "metric_profile": "detection_iou_v1"},
    ).json()
    assert recall_ascending["sort_by"] == "recall_iou50"
    assert recall_ascending["sort_order"] == "asc"
    assert recall_ascending["primary_metric"] == "recall_iou50"
    assert recall_ascending["primary_metric_label"] == "R@.50"
    assert recall_ascending["score_formula"] == "R@.50"
    assert [entry["run_id"] for entry in recall_ascending["entries"]] == ["run_b", "run_a"]
    assert recall_ascending["entries"][0]["score"] == pytest.approx(0.5)
    assert recall_ascending["entries"][0]["score_delta"] == pytest.approx(0.0)
    assert recall_ascending["entries"][1]["score_delta"] == pytest.approx(0.3)

    rank_scheme = {
        "name": "bench1_quality",
        "terms": [
            {
                "benchmark_id": "bench1",
                "metric": "precision_iou50",
                "weight": 0.25,
                "missing": "drop",
            },
            {
                "benchmark_id": "bench1",
                "metric": "mean_iou",
                "weight": 0.75,
                "missing": "zero",
            },
        ],
    }
    weighted = client.get(
        "/api/rank-board",
        params={"rank_scheme": json.dumps(rank_scheme)},
    ).json()
    assert weighted["primary_metric"] == "weighted_score"
    assert weighted["primary_metric_label"] == "bench1_quality"
    assert weighted["sort_by"] == "weighted_score"
    assert weighted["rank_scheme"] == rank_scheme
    assert weighted["entries"][0]["score"] == pytest.approx(0.75)
    assert weighted["entries"][0]["score_delta"] == pytest.approx(0.0)
    assert weighted["entries"][0]["score_components"][0]["metric"] == "precision_iou50"

    bad_scheme = client.get("/api/rank-board", params={"rank_scheme": '{"terms": []}'})
    assert bad_scheme.status_code == 400

    weighted_sort_without_scheme = client.get(
        "/api/rank-board",
        params={"sort_by": "weighted_score"},
    )
    assert weighted_sort_without_scheme.status_code == 400
    assert "requires rank_scheme" in weighted_sort_without_scheme.json()["detail"]


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
    assert client.get("/api/runs/run1/report", params={"summary": "true"}).json()["kind"] == "summary"


def test_dashboard_does_not_claim_next_job_while_live_job_is_running(tmp_path: Path) -> None:
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)
    database = app.state.eval_bench_database
    running = database.create_job(
        kind="eval",
        payload={"run_id": "running-job"},
        status="running",
        metadata={"runtime_pid": os.getpid()},
    )
    queued = database.create_job(kind="eval", payload={"run_id": "queued-job"})

    processed = client.post("/api/jobs/process-next")

    assert processed.status_code == 200
    payload = processed.json()
    assert payload["processed"] is False
    assert payload["job"]["job_id"] == running.job_id
    assert database.get_job(queued.job_id).status == "queued"


def test_dashboard_process_next_checks_live_jobs_beyond_first_page(tmp_path: Path) -> None:
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)
    database = app.state.eval_bench_database
    running = database.create_job(
        kind="eval",
        job_id="aaa_running",
        payload={"run_id": "running-job"},
        status="running",
        metadata={"runtime_pid": os.getpid()},
    )
    queued = database.create_job(kind="eval", job_id="zzz_queued", payload={"run_id": "queued-job"})
    for index in range(220):
        database.create_job(
            kind="eval",
            job_id=f"zzz_finished_{index:04d}",
            payload={"run_id": f"finished-{index}"},
            status="succeeded",
        )

    assert all(job.job_id != running.job_id for job in database.list_jobs(limit=200))

    processed = client.post("/api/jobs/process-next")

    assert processed.status_code == 200
    payload = processed.json()
    assert payload["processed"] is False
    assert payload["job"]["job_id"] == running.job_id
    assert database.get_job(queued.job_id).status == "queued"


def test_dashboard_job_logs_can_return_full_log(tmp_path: Path) -> None:
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)
    log_path = tmp_path / "runs" / "job1" / "logs" / "runtime.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("a\nb\nc\n", encoding="utf-8")
    app.state.eval_bench_database.create_job(
        kind="eval",
        job_id="job1",
        payload={"run_id": "job1"},
        status="running",
        metadata={"runtime_log_path": str(log_path)},
    )

    tail = client.get("/api/jobs/job1/logs", params={"max_lines": 2}).json()
    full = client.get("/api/jobs/job1/logs", params={"max_lines": 0}).json()

    assert tail["lines"] == ["b\n", "c\n"]
    assert full["lines"] == ["a\n", "b\n", "c\n"]


def test_dashboard_manages_run_job_and_service_records(tmp_path: Path) -> None:
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
    (tmp_path / "runs" / "run1" / "reports").mkdir(parents=True)
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    archived = client.post("/api/runs/run1/archive")
    assert archived.status_code == 200
    assert json.loads((tmp_path / "runs" / "run1" / "run.json").read_text(encoding="utf-8"))[
        "status"
    ] == "archived"

    deleted_run = client.delete("/api/runs/run1")
    assert deleted_run.status_code == 200
    assert not (tmp_path / "runs" / "run1").exists()
    assert Path(deleted_run.json()["trash_path"]).exists()

    model_path = tmp_path / "models" / "model-a" / "best"
    model_path.mkdir(parents=True)
    (tmp_path / "benchmarks" / "bench1" / "splits").mkdir(parents=True)
    (tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt").write_text(
        "part1/json/a.json\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["layout"],
            "split": "val",
            "sample_count": 0,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"),
        },
    )
    created_job = client.post(
        "/api/jobs",
        json=_eval_job_payload(
            model_id="model-a",
            model_path=str(model_path),
            benchmark_id="bench1",
            task="detection",
            prompt_id="grounding_layout.v2.4.main",
        ),
    )
    assert created_job.status_code == 201
    job_id = created_job.json()["job_id"]
    cancelled = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    deleted_job = client.delete(f"/api/jobs/{job_id}")
    assert deleted_job.status_code == 200
    assert client.get("/api/jobs").json()["jobs"] == []
    assert Path(deleted_job.json()["trash_path"]).exists()

    created_service = client.post(
        "/api/services",
        json={
            "kind": "external_vllm",
            "service_id": "external",
            "endpoint": "http://127.0.0.1:8000/v1",
        },
    )
    assert created_service.status_code == 201
    deleted_service = client.delete("/api/services/external")
    assert deleted_service.status_code == 200
    assert deleted_service.json()["service"]["service_id"] == "external"
    assert client.get("/api/services").json()["services"] == []


def test_dashboard_cancel_running_job_requests_runtime_termination(tmp_path: Path) -> None:
    app = create_app(store_root=tmp_path, enable_orchestrator=False)
    runtime = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        job = app.state.eval_bench_database.create_job(
            kind="eval",
            payload={"run_id": "running-job"},
            status="running",
            metadata={"runtime_pid": runtime.pid},
        )
        client = TestClient(app)

        response = client.post(f"/api/jobs/{job.job_id}/cancel")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "cancelled"
        assert payload["metadata"]["cancel_requested"] is True
        assert payload["metadata"]["runtime_terminated_pid"] == runtime.pid
        runtime.wait(timeout=5)
        assert _wait_for_process_exit(runtime)
    finally:
        if runtime.poll() is None:
            runtime.terminate()
            runtime.wait(timeout=5)


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


def test_dashboard_exposes_service_registry(tmp_path: Path) -> None:
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    created = client.post(
        "/api/services",
        json={
            "kind": "local_vllm",
            "service_id": "local-vllm-0",
            "model_path": "outputs/model/best",
            "served_model_name": "qwen3vl-best",
            "port": 8000,
            "cuda_visible_devices": "0,1",
            "tensor_parallel_size": 2,
        },
    )

    assert created.status_code == 201
    payload = created.json()
    assert payload["service_id"] == "local-vllm-0"
    assert payload["config"]["model_path"] == "outputs/model/best"

    services_payload = client.get("/api/services").json()
    services = services_payload["services"]
    assert services_payload["total"] == 1
    assert services[0]["service_id"] == "local-vllm-0"
    filtered = client.get(
        "/api/services",
        params={"kind": "local_vllm", "status": "registered", "query": "qwen3vl"},
    ).json()
    assert filtered["filters"] == {
        "kind": "local_vllm",
        "status": "registered",
        "query": "qwen3vl",
    }
    assert filtered["total"] == 1
    assert filtered["services"][0]["service_id"] == "local-vllm-0"

    detail = client.get("/api/services/local-vllm-0")
    assert detail.status_code == 200
    assert detail.json()["service"]["service_id"] == "local-vllm-0"
    assert client.get("/api/services/not_found").status_code == 404

    command = client.get("/api/services/local-vllm-0/command").json()["command"]
    assert command[1:4] == ["-m", "vllm.entrypoints.openai.api_server", "--model"]
    assert "outputs/model/best" in command

    invalid = client.post("/api/services", json={"kind": "external_vllm"})
    assert invalid.status_code == 400


def test_dashboard_exposes_service_health_and_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_probe(endpoint: str, *, timeout_s: float) -> dict:
        return {
            "ok": True,
            "status": "ready",
            "status_code": 200,
            "url": f"{endpoint}/models",
            "message": "HTTP 200",
            "checked_at": "2026-05-09T00:00:00Z",
        }

    monkeypatch.setattr(services_module, "_probe_openai_endpoint", fake_probe)
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)
    client.post(
        "/api/services",
        json={
            "kind": "external_vllm",
            "service_id": "external",
            "endpoint": "http://127.0.0.1:8000/v1",
        },
    )

    health = client.post("/api/services/external/health").json()
    assert health["status"] == "running"
    assert health["runtime"]["health"]["ok"] is True

    logs = client.get("/api/services/external/logs").json()
    assert logs["service_id"] == "external"
    assert logs["lines"] == []
    assert logs["text"] == ""


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
    assert samples[0]["image_tile_url_template"] == "/api/runs/run1/samples/0/image/tiles/{level}/{x}/{y}"
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

    assert client.get("/api/benchmarks/multitask_val_v1/samples/0/image/tiles/0/99/0").status_code == 404
    assert (tmp_path / "cache" / "image_proxy").exists()


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
    assert sample_payload["candidate"]["sample"]["image_url"] == "/api/runs/candidate/samples/0/image"
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
