from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

import eval_bench.dashboard as dashboard_module
from eval_bench.dashboard import create_app
from eval_bench.worker import EvalBenchWorker
from support.dashboard import wait_for_job_status as _wait_for_job_status
from support.dashboard import wait_for_process_exit as _wait_for_process_exit
from support.files import write_json as _write_json
from support.jobs import eval_job_payload as _eval_job_payload


pytestmark = pytest.mark.contract


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


def test_dashboard_job_logs_follow_deleted_run_to_trash(tmp_path: Path) -> None:
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)
    log_path = tmp_path / "runs" / "run1" / "logs" / "runtime.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("kept\nruntime\n", encoding="utf-8")
    _write_json(
        tmp_path / "runs" / "run1" / "run.json",
        {
            "kind": "eval_run",
            "run_id": "run1",
            "status": "succeeded",
            "submitter": "test",
            "model": {"model_id": "m", "path": "models/m"},
            "benchmark": {
                "benchmark_id": "bench1",
                "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
                "split": "val",
                "tasks": ["detection"],
            },
            "spec": {"task": "detection"},
            "artifact_root": str(tmp_path / "runs" / "run1"),
        },
    )
    app.state.eval_bench_database.create_job(
        kind="eval",
        job_id="job1",
        payload={"run_id": "run1"},
        status="succeeded",
        metadata={"runtime_log_path": str(log_path)},
    )

    deleted = client.delete("/api/runs/run1")
    payload = client.get("/api/jobs/job1/logs", params={"max_lines": 0}).json()

    assert deleted.status_code == 200
    assert payload["log_path"] == str(Path(deleted.json()["trash_path"]) / "logs" / "runtime.log")
    assert payload["lines"] == ["kept\n", "runtime\n"]


def test_dashboard_background_worker_marks_uncaught_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)
    database = app.state.eval_bench_database
    job = database.create_job(kind="eval", payload={"run_id": "manual-worker-failure"})

    class FailingWorker(EvalBenchWorker):
        def process_job(self, job_id: str):  # type: ignore[override]
            raise RuntimeError("manual worker crashed")

    monkeypatch.setattr(dashboard_module, "_load_worker_class", lambda: FailingWorker)

    processed = client.post("/api/jobs/process-next")

    assert processed.status_code == 200
    assert processed.json()["processed"] is True
    failed = _wait_for_job_status(client, job.job_id, "failed")
    assert failed["error"] == "manual worker crashed"
    assert failed["metadata"]["worker_action"] == "failed"
    assert failed["metadata"]["worker_failure_source"] == "dashboard"
    assert failed["metadata"]["worker_error_type"] == "RuntimeError"
    assert failed["metadata"]["progress_phase"] == "failed"
    assert failed["metadata"]["dashboard_worker_error_type"] == "RuntimeError"


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
    assert (
        json.loads((tmp_path / "runs" / "run1" / "run.json").read_text(encoding="utf-8"))["status"]
        == "archived"
    )

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
