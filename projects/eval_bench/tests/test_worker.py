from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from eval_bench.adapters.vllm_openai import GeneratedText
from eval_bench.database import EvalBenchDatabase
from eval_bench.sample_paths import sample_image_path
from eval_bench.worker import (
    EvalBenchWorker,
    _default_vllm_internal_port,
    _gpu_memory_window_is_stable,
    _is_vllm_memory_profiling_failure,
    _process_group_exists,
    _resolve_prompt,
    _stop_ephemeral_runtime,
)


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
    runtime_mode: str = "existing_service",
    served_model_name: str | None = None,
    endpoint: str | None = None,
    service_id: str | None = None,
    system_prompt: str | None = None,
    prompt_text: str | None = None,
    cuda_visible_devices: str | None = None,
    tensor_parallel_size: int | None = None,
    port: int | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_num_seqs: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_pixels: int | None = None,
    batch_size: int | None = None,
    metadata: dict | None = None,
) -> dict:
    runtime_args = {
        "model": model_path,
        "served-model-name": served_model_name or model_id,
        "host": "127.0.0.1",
        "port": port,
        "tensor-parallel-size": tensor_parallel_size,
        "max-model-len": max_model_len,
        "gpu-memory-utilization": gpu_memory_utilization,
        "max-num-seqs": max_num_seqs,
    }
    payload = {
        "manifest": {
            "kind": "eval_job",
            "runtime": {
                "mode": runtime_mode,
                "engine": backend,
                "endpoint": endpoint,
                "service_id": service_id,
                "env": {"CUDA_VISIBLE_DEVICES": cuda_visible_devices},
                "args": {
                    key: value for key, value in runtime_args.items() if value not in (None, "")
                },
            },
            "eval": {
                "model_id": model_id,
                "benchmark_id": benchmark_id,
                "task": task,
                "prompt_id": prompt_id,
                "system_prompt": system_prompt,
                "prompt_text": prompt_text,
                "generation": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                "data": {
                    "max_pixels": max_pixels,
                    "batch_size": batch_size,
                },
            },
        }
    }
    if metadata is not None:
        payload["manifest"]["metadata"] = metadata
    return payload


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


def _ephemeral_eval_job_payload() -> dict:
    return {
        "manifest": {
            "kind": "eval_job",
            "runtime": {
                "mode": "ephemeral",
                "engine": "vllm_openai",
                "env": {"CUDA_VISIBLE_DEVICES": "0"},
                "args": {
                    "model": "outputs/model-a/best",
                    "served-model-name": "served-model",
                    "host": "127.0.0.1",
                    "port": 8000,
                    "tensor-parallel-size": 1,
                    "max-model-len": 32768,
                    "gpu-memory-utilization": 0.9,
                    "max-num-seqs": 8,
                },
            },
            "eval": {
                "model_id": "served-model",
                "benchmark_id": "bench1",
                "task": "detection",
                "prompt_id": "grounding_layout.test.main",
                "prompt_text": "detect icons",
                "generation": {"max_tokens": 16, "temperature": 0, "top_p": 1},
                "data": {"batch_size": 1, "max_pixels": 1048576},
            },
        }
    }


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


def test_gpu_memory_window_stability_uses_per_device_delta() -> None:
    assert _gpu_memory_window_is_stable(
        [(74_000, 73_900), (74_050, 73_950), (74_100, 74_000)],
        max_delta_mib=256,
    )
    assert not _gpu_memory_window_is_stable(
        [(74_000, 73_900), (75_000, 73_950), (75_100, 74_000)],
        max_delta_mib=256,
    )


def test_image_path_from_gt_uses_existing_non_png_suffix(tmp_path: Path) -> None:
    image_path = tmp_path / "part2" / "images" / "prod_000876.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")

    resolved = sample_image_path(
        Path("part2/json/prod_000876.json"),
        {},
        root=tmp_path,
    )

    assert resolved == Path("part2/images/prod_000876.jpg")


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


def test_worker_preserves_cancelled_job_status(tmp_path: Path) -> None:
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(
        kind="eval",
        payload=_ephemeral_eval_job_payload(),
        status="running",
    )
    database.cancel_job(job.job_id)

    processed = EvalBenchWorker(tmp_path).process_job(job.job_id)

    assert processed.status == "cancelled"
    assert processed.metadata["progress_phase"] == "cancelled"
    assert processed.metadata["worker_action"] == "cancelled"


def test_worker_prepares_run_manifest_from_queued_job(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection", "keypoint"],
            "layers": ["layout", "arrow"],
            "split": "val",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"),
        },
    )
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(
        kind="eval",
        payload=_eval_job_payload(
            model_id="model-a",
            model_path="outputs/model-a/best",
            benchmark_id="bench1",
            task="keypoint",
            prompt_id="point_arrow.test.main",
            system_prompt="system snapshot",
            prompt_text="predict arrow endpoints",
            service_id="local-vllm-0",
            cuda_visible_devices="0,1,2",
            tensor_parallel_size=3,
            port=8001,
            max_model_len=65536,
            gpu_memory_utilization=0.82,
            max_num_seqs=16,
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            max_pixels=1048576,
            batch_size=2,
        ),
    )

    processed = EvalBenchWorker(tmp_path).process_next()

    assert processed is not None
    assert processed.job_id == job.job_id
    assert processed.status == "succeeded"
    assert "resolved_manifest" not in processed.metadata
    run_path = tmp_path / "runs" / job.job_id / "run.json"
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert run_payload["status"] == "queued"
    assert run_payload["spec"]["task"] == "keypoint"
    assert run_payload["spec"]["target_labels"] == ["arrow"]
    assert run_payload["spec"]["metadata"]["target_labels_source"] == "suite_default"
    assert run_payload["spec"]["inference"]["batch_size"] == 2
    assert run_payload["spec"]["inference"]["service_id"] == "local-vllm-0"
    assert run_payload["spec"]["inference"]["cuda_visible_devices"] == "0,1,2"
    assert run_payload["spec"]["inference"]["tensor_parallel_size"] == 3
    assert run_payload["spec"]["inference"]["port"] == 8001
    assert run_payload["spec"]["inference"]["max_model_len"] == 65536
    assert run_payload["spec"]["inference"]["gpu_memory_utilization"] == 0.82
    assert run_payload["spec"]["inference"]["max_num_seqs"] == 16
    assert run_payload["spec"]["inference"]["temperature"] == 0.1
    assert run_payload["spec"]["inference"]["top_p"] == 0.9
    assert run_payload["spec"]["inference"]["max_pixels"] == 1048576
    assert run_payload["spec"]["prompt"]["prompt_id"] == "point_arrow.test.main"
    assert run_payload["spec"]["prompt"]["text_hash"]
    assert run_payload["spec"]["prompt"]["metadata"]["source"] == "inline"
    assert run_payload["spec"]["prompt"]["metadata"]["system_prompt"] == "system snapshot"
    assert run_payload["spec"]["prompt"]["metadata"]["user_prompt"] == "predict arrow endpoints"
    assert run_payload["metadata"]["source_job_id"] == job.job_id
    assert "job_manifest" not in run_payload["metadata"]


def test_worker_marks_invalid_job_failed(tmp_path: Path) -> None:
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(
        kind="eval",
        payload=_eval_job_payload(
            model_id="model-a",
            model_path="outputs/model-a/best",
            benchmark_id="missing",
            task="detection",
            prompt_id="grounding_layout.test.main",
        ),
    )

    processed = EvalBenchWorker(tmp_path).process_next()

    assert processed is not None
    assert processed.job_id == job.job_id
    assert processed.status == "failed"
    assert "benchmark manifest does not exist" in str(processed.error)


def test_worker_default_prompt_resolution_is_repo_rooted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    system_prompt, user_prompt, prompt_id = _resolve_prompt(
        {"prompt_id": "grounding_layout.test.main"},
        task="detection",
    )

    assert "Return only valid compact JSON" in system_prompt
    assert "Detect all visible top-level layout elements" in user_prompt
    assert prompt_id.startswith("shaft.grounding_layout.prompt_pool.")
    assert prompt_id.endswith(".main")


def test_worker_dry_run_writes_predictions_and_report(tmp_path: Path) -> None:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection", "keypoint"],
            "layers": ["layout", "arrow"],
            "split": "val",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(split_path),
        },
    )
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "data" / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [{"label": "icon", "bbox": [0, 0, 10, 10]}],
        },
    )
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(
        kind="eval",
        payload=_eval_job_payload(
            model_id="dry-model",
            model_path="outputs/dry/best",
            benchmark_id="bench1",
            task="detection",
            prompt_id="grounding_layout.test.main",
            backend="dry_run",
            runtime_mode="external",
            metadata={"notes": "checkpoint=5000; full benchmark sweep"},
        ),
    )

    processed = EvalBenchWorker(tmp_path).process_next()

    assert processed is not None
    assert processed.status == "succeeded"
    assert processed.metadata["worker_action"] == "dry_run"
    assert processed.metadata["progress_phase"] == "succeeded"
    assert processed.metadata["progress_done"] == 1
    assert processed.metadata["progress_total"] == 1
    assert processed.metadata["progress_current_sample"] == "part1/json/a.json"
    assert processed.metadata["run_note_path"].endswith("runs/" + job.job_id + "/note.json")
    note = json.loads((tmp_path / "runs" / job.job_id / "note.json").read_text(encoding="utf-8"))
    assert note["note"] == "checkpoint=5000; full benchmark sweep"
    assert (tmp_path / "runs" / job.job_id / "predictions" / "part1" / "json" / "a.json").exists()
    report_path = tmp_path / "runs" / job.job_id / "reports" / "metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["prediction_file_count"] == 1
    assert report["target_labels"] == ["icon", "image", "shape"]
    assert report["target_labels_source"] == "suite_default"
    assert report["recall_iou50"] == 0.0
    run_payload = json.loads((tmp_path / "runs" / job.job_id / "run.json").read_text(encoding="utf-8"))
    assert run_payload["status"] == "succeeded"


def test_worker_vllm_openai_writes_predictions_raw_outputs_and_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeAdapter:
        def __init__(self, **kwargs):
            assert kwargs["endpoint"] == "http://127.0.0.1:8000"
            assert kwargs["served_model_name"] == "served-model"

        def generate(self, **kwargs):
            assert kwargs["user_prompt"] == "detect icons"
            assert kwargs["max_tokens"] == 4096
            assert kwargs["max_pixels"] == 1048576
            return GeneratedText(
                text='[{"label":"icon","bbox_2d":[0,0,1000,1000]}]',
                latency_ms=12.5,
                raw_response={"choices": []},
                image_request={"source_pixels": 5000, "target_pixels": 5000, "resized": False},
            )

    monkeypatch.setattr("eval_bench.worker.OpenAICompatibleVLLMAdapter", FakeAdapter)
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    (data_root / "part1" / "images").mkdir(parents=True)
    (data_root / "part1" / "images" / "a.png").write_bytes(b"image")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["layout"],
            "split": "val",
            "sample_count": 1,
            "root": str(data_root),
            "manifest_path": str(split_path),
        },
    )
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "image_width": 100,
            "image_height": 50,
            "instances": [{"label": "icon", "bbox": [0, 0, 100, 50]}],
        },
    )
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(
        kind="eval",
        payload=_eval_job_payload(
            model_id="model-a",
            model_path="outputs/model-a/best",
            served_model_name="served-model",
            benchmark_id="bench1",
            task="detection",
            prompt_id="grounding_layout.test.main",
            prompt_text="detect icons",
            backend="vllm_openai",
            endpoint="http://127.0.0.1:8000",
            max_tokens=4096,
            max_pixels=1048576,
        ),
    )

    processed = EvalBenchWorker(tmp_path).process_next()

    assert processed is not None
    assert processed.status == "succeeded"
    assert processed.metadata["worker_action"] == "vllm_openai"
    assert processed.metadata["progress_phase"] == "succeeded"
    assert processed.metadata["progress_done"] == 1
    assert processed.metadata["progress_total"] == 1
    assert processed.metadata["progress_current_sample"] == "part1/json/a.json"
    prediction_path = tmp_path / "runs" / job.job_id / "predictions" / "part1" / "json" / "a.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction["instances"][0]["bbox"] == [0.0, 0.0, 100.0, 50.0]
    assert prediction["metadata"]["latency_ms"] == 12.5
    assert prediction["metadata"]["inference_params"]["max_pixels"] == 1048576
    assert prediction["metadata"]["inference_params"]["image_request"]["resized"] is False
    assert (tmp_path / "runs" / job.job_id / "raw_outputs" / "part1" / "txt" / "a.txt").exists()
    report = json.loads(
        (tmp_path / "runs" / job.job_id / "reports" / "metrics.json").read_text(encoding="utf-8")
    )
    assert report["precision_iou50"] == 1.0
    assert report["recall_iou50"] == 1.0


def test_worker_vllm_openai_runs_requests_concurrently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0
    calls: list[str] = []

    class FakeAdapter:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            nonlocal active, max_active
            image_name = Path(kwargs["image_path"]).name
            with lock:
                active += 1
                max_active = max(max_active, active)
                calls.append(image_name)
            time.sleep(0.05)
            with lock:
                active -= 1
            return GeneratedText(
                text='[{"label":"icon","bbox_2d":[0,0,1000,1000]}]',
                latency_ms=12.5,
                raw_response={"choices": []},
                image_request={"source_pixels": 5000, "target_pixels": 5000, "resized": False},
            )

    monkeypatch.setattr("eval_bench.worker.OpenAICompatibleVLLMAdapter", FakeAdapter)
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(
        "\n".join(f"part1/json/{name}.json" for name in ["a", "b", "c", "d"]) + "\n",
        encoding="utf-8",
    )
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    (data_root / "part1" / "images").mkdir(parents=True)
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["layout"],
            "split": "val",
            "sample_count": 4,
            "root": str(data_root),
            "manifest_path": str(split_path),
        },
    )
    for name in ["a", "b", "c", "d"]:
        (data_root / "part1" / "images" / f"{name}.png").write_bytes(b"image")
        _write_json(
            data_root / "part1" / "json" / f"{name}.json",
            {
                "image_path": f"part1/images/{name}.png",
                "image_width": 100,
                "image_height": 50,
                "instances": [{"label": "icon", "bbox": [0, 0, 100, 50]}],
            },
        )
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(
        kind="eval",
        payload=_eval_job_payload(
            model_id="model-a",
            model_path="outputs/model-a/best",
            served_model_name="served-model",
            benchmark_id="bench1",
            task="detection",
            prompt_id="grounding_layout.test.main",
            prompt_text="detect icons",
            backend="vllm_openai",
            endpoint="http://127.0.0.1:8000",
            max_tokens=4096,
            max_num_seqs=3,
            batch_size=1,
        ),
    )

    processed = EvalBenchWorker(tmp_path).process_next()

    assert processed is not None
    assert processed.status == "succeeded"
    assert max_active > 1
    assert sorted(calls) == ["a.png", "b.png", "c.png", "d.png"]
    prediction_path = tmp_path / "runs" / job.job_id / "predictions" / "part1" / "json" / "a.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction["metadata"]["inference_params"]["request_concurrency"] == 3
    assert len(list((tmp_path / "runs" / job.job_id / "predictions").rglob("*.json"))) == 4
