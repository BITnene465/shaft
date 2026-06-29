from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

from eval_bench.database import EvalBenchDatabase
from eval_bench.worker import (
    EvalBenchWorker,
    _default_vllm_internal_port,
    _gpu_memory_window_is_stable,
    _is_vllm_memory_profiling_failure,
    _process_group_exists,
    _stop_ephemeral_runtime,
    _vllm_extra_body,
)
from support.files import write_json as _write_json
from support.jobs import ephemeral_eval_job_payload as _ephemeral_eval_job_payload


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until_pid_exits(pid: int, *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.05)
    return not _pid_exists(pid)


def _wait_until_process_group_exits(pgid: int, *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return True
        time.sleep(0.05)
    return not _process_group_exists(pgid)


def _start_runtime_like_process_group(
    tmp_path: Path,
    *,
    name: str,
    parent_exits: bool,
) -> tuple[subprocess.Popen[bytes], int]:
    child_pid_path = tmp_path / f"{name}.child.pid"
    parent_tail = "sys.exit(0)" if parent_exits else "time.sleep(60)"
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, subprocess, sys, time; "
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)']); "
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
                f"{parent_tail}"
            ),
        ],
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if child_pid_path.exists():
            return launcher, int(child_pid_path.read_text(encoding="utf-8"))
        if launcher.poll() is not None and not child_pid_path.exists():
            raise RuntimeError(
                f"runtime-like launcher exited before writing child pid: {launcher.returncode}"
            )
        time.sleep(0.05)
    _stop_ephemeral_runtime(launcher)
    raise TimeoutError("runtime-like launcher did not write child pid")


def test_vllm_extra_body_rejects_request_pixel_budget_overrides() -> None:
    try:
        _vllm_extra_body({"extra": {"extra_body": {"mm_processor_kwargs": {"max_pixels": 123}}}})
    except ValueError as exc:
        assert "pixel budget" in str(exc)
    else:
        raise AssertionError("expected pixel budget override to be rejected")


def test_stop_ephemeral_runtime_cleans_process_group_when_parent_already_exited(
    tmp_path: Path,
) -> None:
    launcher, child_pid = _start_runtime_like_process_group(
        tmp_path,
        name="parent-exited",
        parent_exits=True,
    )
    launcher.wait(timeout=5)
    assert launcher.poll() is not None
    assert _pid_exists(child_pid)

    _stop_ephemeral_runtime(launcher)

    assert _wait_until_pid_exits(child_pid)
    assert _wait_until_process_group_exits(launcher.pid)


def test_stop_ephemeral_runtime_cleans_process_group_when_parent_is_alive(
    tmp_path: Path,
) -> None:
    launcher, child_pid = _start_runtime_like_process_group(
        tmp_path,
        name="parent-alive",
        parent_exits=False,
    )
    try:
        assert launcher.poll() is None
        assert _pid_exists(child_pid)

        _stop_ephemeral_runtime(launcher)

        assert launcher.poll() is not None
        assert _wait_until_pid_exits(child_pid)
        assert _wait_until_process_group_exits(launcher.pid)
    finally:
        _stop_ephemeral_runtime(launcher)


def test_default_vllm_internal_port_is_separate_from_api_port() -> None:
    port = _default_vllm_internal_port({"port": 8000})

    assert 28000 <= port < 28100


def test_vllm_memory_profiling_failure_is_detected_from_runtime_log(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.log"
    log_path.write_text(
        "AssertionError: Error in memory profiling. "
        "Initial free memory 74.0 GiB, current free memory 75.03 GiB.",
        encoding="utf-8",
    )

    assert _is_vllm_memory_profiling_failure(log_path)
    assert not _is_vllm_memory_profiling_failure(tmp_path / "missing.log")


def test_vllm_low_free_memory_startup_failure_is_retryable(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.log"
    log_path.write_text(
        "ValueError: Free memory on device cuda:1 (14.70/79.33 GiB) on startup "
        "is less than desired GPU memory utilization (0.8, 63.46 GiB).",
        encoding="utf-8",
    )

    assert _is_vllm_memory_profiling_failure(log_path)


def test_gpu_memory_window_stability_uses_per_device_delta() -> None:
    assert _gpu_memory_window_is_stable(
        [(74_000, 73_900), (74_050, 73_950), (74_100, 74_000)],
        max_delta_mib=256,
    )
    assert not _gpu_memory_window_is_stable(
        [(74_000, 73_900), (75_000, 73_950), (75_100, 74_000)],
        max_delta_mib=256,
    )


def test_worker_stops_ephemeral_runtime_after_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime: dict[str, subprocess.Popen[bytes] | int] = {}

    def fake_start_runtime(self, job):
        process, child_pid = _start_runtime_like_process_group(
            tmp_path,
            name="worker-success-runtime",
            parent_exits=False,
        )
        runtime["process"] = process
        runtime["child_pid"] = child_pid
        log_path = tmp_path / "runs" / job.job_id / "logs" / "runtime.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("runtime ready\n", encoding="utf-8")
        return process, log_path

    def fake_prepare_run(self, job):
        run_path = tmp_path / "runs" / job.job_id / "run.json"
        _write_json(run_path, {"status": "queued", "metadata": {}})
        return run_path

    def fake_run_inference(self, job):
        report_path = tmp_path / "runs" / job.job_id / "reports" / "metrics.json"
        _write_json(report_path, {"precision_iou50": 1.0})
        return report_path

    monkeypatch.setattr(EvalBenchWorker, "start_ephemeral_runtime", fake_start_runtime)
    monkeypatch.setattr(EvalBenchWorker, "prepare_run", fake_prepare_run)
    monkeypatch.setattr(EvalBenchWorker, "run_vllm_openai_inference", fake_run_inference)
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(kind="eval", payload=_ephemeral_eval_job_payload())

    processed = EvalBenchWorker(tmp_path).process_next()

    assert processed is not None
    assert processed.job_id == job.job_id
    assert processed.status == "succeeded"
    process = runtime["process"]
    child_pid = runtime["child_pid"]
    assert isinstance(process, subprocess.Popen)
    assert isinstance(child_pid, int)
    assert process.poll() is not None
    assert _wait_until_pid_exits(child_pid)
    assert _wait_until_process_group_exits(process.pid)


def test_worker_stops_ephemeral_runtime_after_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime: dict[str, subprocess.Popen[bytes] | int] = {}

    def fake_start_runtime(self, job):
        process, child_pid = _start_runtime_like_process_group(
            tmp_path,
            name="worker-failure-runtime",
            parent_exits=False,
        )
        runtime["process"] = process
        runtime["child_pid"] = child_pid
        log_path = tmp_path / "runs" / job.job_id / "logs" / "runtime.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("runtime ready\n", encoding="utf-8")
        return process, log_path

    def fake_prepare_run(self, job):
        raise RuntimeError("boom after runtime start")

    monkeypatch.setattr(EvalBenchWorker, "start_ephemeral_runtime", fake_start_runtime)
    monkeypatch.setattr(EvalBenchWorker, "prepare_run", fake_prepare_run)
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(kind="eval", payload=_ephemeral_eval_job_payload())

    processed = EvalBenchWorker(tmp_path).process_next()

    assert processed is not None
    assert processed.job_id == job.job_id
    assert processed.status == "failed"
    assert "boom after runtime start" in str(processed.error)
    process = runtime["process"]
    child_pid = runtime["child_pid"]
    assert isinstance(process, subprocess.Popen)
    assert isinstance(child_pid, int)
    assert process.poll() is not None
    assert _wait_until_pid_exits(child_pid)
    assert _wait_until_process_group_exits(process.pid)
