from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Callable

from .artifacts import DEFAULT_STORE_ROOT, BenchmarkArtifacts, RunArtifacts, atomic_write_json
from .adapters.vllm_openai import OpenAICompatibleVLLMAdapter
from .database import EvalBenchDatabase, JobRecord
from .evaluator import evaluate_run
from .job_spec import resolve_job_payload
from .label_policy import normalize_target_labels, resolve_target_labels
from .prediction_parser import parse_prediction_text
from .schema import (
    BenchmarkRef,
    EvalRunManifest,
    EvalSpec,
    InferenceParams,
    ModelRef,
    PredictionDocument,
    PromptRef,
    TaskKind,
    utc_now_iso,
)
from .services import build_vllm_command_from_config, _probe_openai_endpoint


REPO_ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger("eval_bench.worker")


class JobCancelled(RuntimeError):
    pass


class EvalBenchWorker:
    def __init__(self, root: str | Path = DEFAULT_STORE_ROOT) -> None:
        self.root = Path(root)
        self.database = EvalBenchDatabase(self.root)

    def claim_next(self, *, kind: str | None = "eval") -> JobRecord | None:
        return self.database.claim_next_job(kind=kind)

    def process_next(self, *, kind: str | None = "eval") -> JobRecord | None:
        job = self.claim_next(kind=kind)
        if job is None:
            return None
        return self.process_job(job.job_id)

    def process_job(self, job_id: str) -> JobRecord:
        job = self.database.get_job(job_id)
        if job is None:
            raise KeyError(f"unknown job_id: {job_id}")
        if job.status == "queued":
            claimed = self.database.claim_next_job(kind=job.kind)
            if claimed is None or claimed.job_id != job.job_id:
                raise RuntimeError(f"job is not claimable: {job.job_id}")
            job = claimed
        return self._process_claimed_job(job)

    def _process_claimed_job(self, job: JobRecord) -> JobRecord:
        resolved_kind = job.kind
        runtime_process: subprocess.Popen[bytes] | None = None
        prompt_templates = {
            record.prompt_id: record.to_dict()
            for record in self.database.list_prompt_templates(limit=1000)
        }
        try:
            self._raise_if_cancelled(job)
            self._update_progress(
                job,
                phase="resolving",
                message="Resolving job manifest and prompt template.",
            )
            resolved = resolve_job_payload(job.payload, prompt_templates=prompt_templates)
            resolved_kind = resolved.kind
            job = replace(job, payload=resolved.payload)
            self._raise_if_cancelled(job)
            if _should_start_ephemeral_runtime(job.payload):
                self._update_progress(
                    job,
                    phase="starting_runtime",
                    message="Starting ephemeral vLLM runtime.",
                )
                runtime_process, runtime_log_path = self.start_ephemeral_runtime(job)
                self._raise_if_cancelled(job)
                endpoint = _endpoint_from_job_payload(job.payload)
                job.payload["endpoint"] = endpoint
                self.database.update_job(
                    job.job_id,
                    status="running",
                    metadata_update={
                        "worker_action": "runtime_ready",
                        "progress_phase": "runtime_ready",
                        "progress_message": "vLLM runtime is ready.",
                        "progress_updated_at": utc_now_iso(),
                        "runtime_pid": runtime_process.pid,
                        "runtime_log_path": str(runtime_log_path),
                        "runtime_endpoint": endpoint,
                    },
                )
            self._raise_if_cancelled(job)
            self._update_progress(
                job,
                phase="prepare_run",
                message="Writing run manifest.",
            )
            manifest_path = self.prepare_run(job)
            metadata_update = {
                "run_id": str(job.payload.get("run_id") or job.job_id),
                "run_manifest_path": str(manifest_path),
                "job_kind": resolved_kind,
                "resolved_manifest": resolved.manifest,
                "worker_action": "prepare_run",
                "progress_phase": "prepare_run",
                "progress_message": "Run manifest prepared.",
                "progress_updated_at": utc_now_iso(),
            }
            if str(job.payload.get("backend") or "") == "dry_run":
                report_path = self.run_dry_inference(job)
                metadata_update.update(
                    {
                        "worker_action": "dry_run",
                        "progress_phase": "succeeded",
                        "progress_message": "Dry-run inference and evaluation completed.",
                        "progress_updated_at": utc_now_iso(),
                        "report_path": str(report_path),
                    }
                )
            elif str(job.payload.get("backend") or "vllm_openai") == "vllm_openai" and _optional_string(
                job.payload, "endpoint"
            ):
                report_path = self.run_vllm_openai_inference(job)
                metadata_update.update(
                    {
                        "worker_action": "vllm_openai",
                        "progress_phase": "succeeded",
                        "progress_message": "Inference and evaluation completed.",
                        "progress_updated_at": utc_now_iso(),
                        "report_path": str(report_path),
                    }
                )
            if metadata_update.get("progress_phase") == "prepare_run":
                metadata_update.update(
                    {
                        "progress_phase": "succeeded",
                        "progress_message": "Run manifest prepared; no inference backend was executed.",
                        "progress_updated_at": utc_now_iso(),
                    }
                )
        except JobCancelled as exc:
            LOGGER.info("job cancelled job_id=%s kind=%s", job.job_id, job.kind)
            _mark_run_cancelled_if_exists(self.root, job, message=str(exc))
            return self.database.update_job(
                job.job_id,
                status="cancelled",
                error=None,
                metadata_update={
                    "worker_action": "cancelled",
                    "job_kind": resolved_kind,
                    "progress_phase": "cancelled",
                    "progress_message": str(exc) or "Job cancelled.",
                    "progress_updated_at": utc_now_iso(),
                },
            )
        except Exception as exc:
            if self._cancel_requested(job):
                LOGGER.info("job stopped after cancellation job_id=%s kind=%s", job.job_id, job.kind)
                _mark_run_cancelled_if_exists(self.root, job, message="Job cancelled.")
                return self.database.update_job(
                    job.job_id,
                    status="cancelled",
                    error=None,
                    metadata_update={
                        "worker_action": "cancelled",
                        "job_kind": resolved_kind,
                        "progress_phase": "cancelled",
                        "progress_message": "Job cancelled.",
                        "progress_updated_at": utc_now_iso(),
                    },
                )
            LOGGER.exception("job failed job_id=%s kind=%s error=%s", job.job_id, job.kind, exc)
            _mark_run_failed_if_exists(self.root, job, error=str(exc))
            failure_metadata: dict[str, Any] = {"worker_action": "failed"}
            inferred_log_path = _runtime_log_path_if_exists(self.root, job)
            if inferred_log_path is not None:
                failure_metadata["runtime_log_path"] = str(inferred_log_path)
            failure_metadata.update(
                {
                    "job_kind": resolved_kind,
                    "progress_phase": "failed",
                    "progress_message": str(exc),
                    "progress_updated_at": utc_now_iso(),
                }
            )
            return self.database.update_job(
                job.job_id,
                status="failed",
                error=str(exc),
                metadata_update=failure_metadata,
            )
        finally:
            if runtime_process is not None:
                _stop_ephemeral_runtime(runtime_process)
        if self._cancel_requested(job):
            _mark_run_cancelled_if_exists(self.root, job, message="Job cancelled.")
            return self.database.update_job(
                job.job_id,
                status="cancelled",
                error=None,
                metadata_update={
                    "worker_action": "cancelled",
                    "job_kind": resolved_kind,
                    "progress_phase": "cancelled",
                    "progress_message": "Job cancelled.",
                    "progress_updated_at": utc_now_iso(),
                },
            )
        return self.database.update_job(
            job.job_id,
            status="succeeded",
            metadata_update=metadata_update,
        )

    def _update_progress(
        self,
        job: JobRecord,
        *,
        phase: str,
        done: int | None = None,
        total: int | None = None,
        message: str | None = None,
        current_sample: str | None = None,
    ) -> None:
        self._raise_if_cancelled(job)
        metadata: dict[str, Any] = {
            "worker_action": phase,
            "progress_phase": phase,
            "progress_updated_at": utc_now_iso(),
        }
        if done is not None:
            metadata["progress_done"] = int(done)
        if total is not None:
            metadata["progress_total"] = int(total)
        if message is not None:
            metadata["progress_message"] = message
        if current_sample is not None:
            metadata["progress_current_sample"] = current_sample
        self.database.update_job(job.job_id, status="running", metadata_update=metadata)

    def _cancel_requested(self, job: JobRecord) -> bool:
        current = self.database.get_job(job.job_id)
        if current is None:
            return False
        metadata = current.metadata if isinstance(current.metadata, dict) else {}
        return current.status == "cancelled" or bool(metadata.get("cancel_requested"))

    def _raise_if_cancelled(self, job: JobRecord) -> None:
        if self._cancel_requested(job):
            raise JobCancelled("Job cancelled by user.")

    def start_ephemeral_runtime(self, job: JobRecord) -> tuple[subprocess.Popen[bytes], Path]:
        run_id = str(job.payload.get("run_id") or job.job_id)
        artifacts = RunArtifacts(self.root, run_id)
        artifacts.ensure()
        log_path = artifacts.logs_dir / "runtime.log"
        config = _runtime_config_from_job_payload(job.payload)
        command = build_vllm_command_from_config(
            config,
            service_id=str(job.payload.get("service_id") or job.payload.get("model_id") or job.job_id),
        )
        env = os.environ.copy()
        env.update(_runtime_env_from_payload(job.payload))
        cuda_devices = job.payload.get("cuda_visible_devices")
        if cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
        log_file = log_path.open("ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()
        endpoint = _endpoint_from_job_payload(job.payload)
        self.database.update_job(
            job.job_id,
            status="running",
            metadata_update={
                "runtime_pid": process.pid,
                "runtime_log_path": str(log_path),
                "runtime_endpoint": endpoint,
                "progress_phase": "starting_runtime",
                "progress_message": "vLLM runtime process started; waiting for health check.",
                "progress_updated_at": utc_now_iso(),
            },
        )
        _wait_for_runtime_ready(
            endpoint,
            process=process,
            timeout_s=float(job.payload.get("runtime_timeout_s") or 900.0),
            cancel_check=lambda: self._raise_if_cancelled(job),
        )
        return process, log_path

    def prepare_run(self, job: JobRecord) -> Path:
        payload = job.payload
        benchmark_id = _require_string(payload, "benchmark_id")
        task = _require_task(payload)
        benchmark_payload = _load_benchmark_payload(self.root, benchmark_id)
        benchmark_tasks = [str(item) for item in benchmark_payload.get("tasks") or []]
        if task not in benchmark_tasks:
            raise ValueError(
                f"job task={task!r} is not available in benchmark {benchmark_id}: {benchmark_tasks}"
            )

        run_id = str(payload.get("run_id") or job.job_id)
        benchmark_artifacts = BenchmarkArtifacts(self.root, benchmark_id)
        inference_params = InferenceParams(
            backend=str(payload.get("backend") or "vllm_openai"),
            endpoint=_optional_string(payload, "endpoint"),
            served_model_name=_optional_string(payload, "served_model_name"),
            service_id=_optional_string(payload, "service_id"),
            cuda_visible_devices=_optional_string(payload, "cuda_visible_devices"),
            tensor_parallel_size=_optional_int(payload, "tensor_parallel_size"),
            port=_optional_int(payload, "port"),
            max_model_len=_optional_int(payload, "max_model_len"),
            gpu_memory_utilization=_optional_float(payload, "gpu_memory_utilization"),
            max_num_seqs=_optional_int(payload, "max_num_seqs"),
            max_tokens=int(payload.get("max_tokens") or 4096),
            temperature=float(payload.get("temperature") or 0.0),
            top_p=float(payload.get("top_p") or 1.0),
            min_pixels=_optional_int(payload, "min_pixels"),
            max_pixels=_optional_int(payload, "max_pixels"),
            batch_size=int(payload.get("batch_size") or 1),
            extra=dict(payload.get("inference_extra") or {}),
        )
        manifest = EvalRunManifest(
            run_id=run_id,
            status="queued",
            submitter=str(payload.get("submitter") or "dashboard"),
            model=ModelRef(
                model_id=_require_string(payload, "model_id"),
                path=_require_string(payload, "model_path"),
                alias=_optional_string(payload, "model_alias"),
                checkpoint_kind=_optional_string(payload, "checkpoint_kind"),
            ),
            benchmark=BenchmarkRef(
                benchmark_id=benchmark_id,
                root=str(benchmark_payload.get("root") or benchmark_artifacts.data_dir),
                split=str(benchmark_payload.get("split") or "val"),
                tasks=[item for item in benchmark_tasks if item in {"detection", "keypoint"}],
                manifest_path=str(
                    benchmark_payload.get("manifest_path")
                    or benchmark_artifacts.split_path(str(benchmark_payload.get("split") or "val"))
                ),
            ),
            spec=EvalSpec(
                spec_id=str(payload.get("spec_id") or f"{task}.default"),
                task=task,
                prompt=_prompt_ref_from_payload(payload, task=task),
                parser=str(payload.get("parser") or "shaft.codec.json_any"),
                metric_profile=str(payload.get("metric_profile") or "default"),
                visualization_profile=str(payload.get("visualization_profile") or "default"),
                target_labels=resolve_target_labels(
                    explicit=payload.get("target_labels"),
                    prompt_id=str(payload.get("prompt_id") or ""),
                    task=task,
                ),
                inference=inference_params,
            ),
            artifact_root=str(RunArtifacts(self.root, run_id).run_dir),
            metadata={
                "source_job_id": job.job_id,
                "worker_action": "prepare_run",
                "job_kind": job.payload.get("job_kind"),
                "job_manifest": job.payload.get("job_manifest"),
                "runtime_mode": job.payload.get("runtime_mode"),
            },
        )
        return RunArtifacts(self.root, run_id).write_manifest(manifest)

    def run_dry_inference(self, job: JobRecord) -> Path:
        run_id = str(job.payload.get("run_id") or job.job_id)
        artifacts = RunArtifacts(self.root, run_id)
        manifest_payload = _load_json(artifacts.manifest_path)
        benchmark = dict(manifest_payload.get("benchmark") or {})
        split_path = Path(str(benchmark.get("manifest_path") or ""))
        benchmark_root = Path(str(benchmark.get("root") or ""))
        if not split_path.exists():
            raise FileNotFoundError(f"benchmark split manifest does not exist: {split_path}")
        json_relatives = _read_split(split_path)
        total = len(json_relatives)
        self._update_progress(job, phase="inference", done=0, total=total, message="Dry-run inference.")
        for index, json_relative in enumerate(json_relatives, start=1):
            self._raise_if_cancelled(job)
            gt_doc = _load_json(benchmark_root / json_relative)
            image = _image_path_from_gt(json_relative, gt_doc)
            prediction = PredictionDocument(
                image=str(image),
                instances=[],
                metadata={
                    "producer": "eval_bench",
                    "run_id": run_id,
                    "model_id": str(job.payload.get("model_id") or ""),
                    "task": str(job.payload.get("task") or ""),
                    "latency_ms": 0.0,
                    "inference_params": {
                        "backend": "dry_run",
                        "max_tokens": int(job.payload.get("max_tokens") or 4096),
                    },
                    "parser": {"valid": True, "mode": "dry_run"},
                },
            )
            artifacts.write_prediction(prediction, task=_require_task(job.payload))
            self._update_progress(
                job,
                phase="inference",
                done=index,
                total=total,
                message=f"Dry-run prediction {index}/{total}.",
                current_sample=str(json_relative),
            )
        self._update_progress(job, phase="evaluating", done=total, total=total, message="Computing metrics.")
        self._raise_if_cancelled(job)
        report_path = evaluate_run(store_root=self.root, run_id=run_id)
        _update_run_status(artifacts.manifest_path, status="succeeded")
        return report_path

    def run_vllm_openai_inference(self, job: JobRecord) -> Path:
        run_id = str(job.payload.get("run_id") or job.job_id)
        artifacts = RunArtifacts(self.root, run_id)
        manifest_payload = _load_json(artifacts.manifest_path)
        _update_run_status(artifacts.manifest_path, status="running")
        task = _require_task(job.payload)
        spec = dict(manifest_payload.get("spec") or {})
        inference = dict(spec.get("inference") or {})
        benchmark = dict(manifest_payload.get("benchmark") or {})
        split_path = Path(str(benchmark.get("manifest_path") or ""))
        benchmark_root = Path(str(benchmark.get("root") or ""))
        if not split_path.exists():
            raise FileNotFoundError(f"benchmark split manifest does not exist: {split_path}")
        system_prompt, user_prompt, prompt_id = _resolve_prompt(job.payload, task=task)
        adapter = OpenAICompatibleVLLMAdapter(
            endpoint=_require_string(job.payload, "endpoint"),
            served_model_name=str(
                inference.get("served_model_name")
                or job.payload.get("served_model_name")
                or job.payload.get("model_id")
            ),
            api_key=_api_key_from_payload(job.payload),
            timeout_s=float(job.payload.get("timeout_s") or 600.0),
        )
        json_relatives = _read_split(split_path)
        total = len(json_relatives)
        self._update_progress(job, phase="inference", done=0, total=total, message="Running model inference.")
        for index, json_relative in enumerate(json_relatives, start=1):
            self._raise_if_cancelled(job)
            gt_doc = _load_json(benchmark_root / json_relative)
            image = _image_path_from_gt(json_relative, gt_doc)
            image_path = benchmark_root / image
            image_width, image_height = _image_size(gt_doc, image_path)
            result = adapter.generate(
                image_path=image_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=int(inference.get("max_tokens") or job.payload.get("max_tokens") or 4096),
                temperature=float(inference.get("temperature") or job.payload.get("temperature") or 0.0),
                top_p=float(inference.get("top_p") or job.payload.get("top_p") or 1.0),
            )
            _write_raw_output(artifacts, image=image, text=result.text)
            prediction = parse_prediction_text(
                text=result.text,
                task=task,
                image=str(image),
                image_width=image_width,
                image_height=image_height,
                metadata={
                    "producer": "eval_bench",
                    "run_id": run_id,
                    "model_id": str(job.payload.get("model_id") or ""),
                    "task": task,
                    "created_at": utc_now_iso(),
                    "latency_ms": result.latency_ms,
                    "inference_params": inference,
                    "parser": {
                        "name": "eval_bench.prediction_parser",
                        "prompt_id": prompt_id,
                    },
                },
            )
            artifacts.write_prediction(prediction, task=task)
            self._update_progress(
                job,
                phase="inference",
                done=index,
                total=total,
                message=f"Model inference {index}/{total}.",
                current_sample=str(json_relative),
            )
        self._update_progress(job, phase="evaluating", done=total, total=total, message="Computing metrics.")
        self._raise_if_cancelled(job)
        report_path = evaluate_run(store_root=self.root, run_id=run_id)
        _update_run_status(artifacts.manifest_path, status="succeeded")
        return report_path


def _load_benchmark_payload(root: Path, benchmark_id: str) -> dict[str, Any]:
    path = BenchmarkArtifacts(root, benchmark_id).manifest_path
    if not path.exists():
        raise FileNotFoundError(f"benchmark manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark manifest must be a JSON object: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _update_run_status(path: Path, *, status: str) -> None:
    payload = _load_json(path)
    payload["status"] = status
    atomic_write_json(path, payload)


def _mark_run_failed_if_exists(root: Path, job: JobRecord, *, error: str) -> None:
    run_id = str(job.payload.get("run_id") or job.job_id)
    path = RunArtifacts(root, run_id).manifest_path
    if not path.exists():
        return
    payload = _load_json(path)
    payload["status"] = "failed"
    metadata = dict(payload.get("metadata") or {})
    metadata["error"] = error
    payload["metadata"] = metadata
    atomic_write_json(path, payload)


def _mark_run_cancelled_if_exists(root: Path, job: JobRecord, *, message: str) -> None:
    run_id = str(job.payload.get("run_id") or job.job_id)
    path = RunArtifacts(root, run_id).manifest_path
    if not path.exists():
        return
    payload = _load_json(path)
    payload["status"] = "cancelled"
    metadata = dict(payload.get("metadata") or {})
    metadata["cancelled_at"] = utc_now_iso()
    metadata["cancelled_message"] = message
    payload["metadata"] = metadata
    atomic_write_json(path, payload)


def _runtime_log_path_if_exists(root: Path, job: JobRecord) -> Path | None:
    run_id = str(job.payload.get("run_id") or job.job_id)
    log_path = RunArtifacts(root, run_id).logs_dir / "runtime.log"
    return log_path if log_path.exists() else None


def _should_start_ephemeral_runtime(payload: dict[str, Any]) -> bool:
    return (
        str(payload.get("runtime_mode") or "existing_service") == "ephemeral"
        and str(payload.get("backend") or "vllm_openai") == "vllm_openai"
    )


def _runtime_config_from_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_path": payload.get("model_path"),
        "served_model_name": payload.get("served_model_name") or payload.get("model_id"),
        "host": payload.get("host") or "127.0.0.1",
        "port": payload.get("port"),
        "tensor_parallel_size": payload.get("tensor_parallel_size"),
        "max_model_len": payload.get("max_model_len"),
        "gpu_memory_utilization": payload.get("gpu_memory_utilization"),
        "max_num_seqs": payload.get("max_num_seqs"),
        "extra_args": payload.get("extra_args"),
    }


def _runtime_env_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    manifest = payload.get("job_manifest") or payload.get("manifest")
    if not isinstance(manifest, dict):
        return {}
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        return {}
    env = runtime.get("env")
    if not isinstance(env, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in env.items()
        if key and value not in (None, "")
    }


def _endpoint_from_job_payload(payload: dict[str, Any]) -> str:
    endpoint = payload.get("endpoint")
    if isinstance(endpoint, str) and endpoint.strip():
        return endpoint.strip()
    host = str(payload.get("host") or "127.0.0.1")
    port = str(payload.get("port") or "8000")
    return f"http://{host}:{port}"


def _wait_for_runtime_ready(
    endpoint: str,
    *,
    process: subprocess.Popen[bytes],
    timeout_s: float,
    cancel_check: Callable[[], None] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_message = "runtime did not become ready"
    while time.monotonic() < deadline:
        if cancel_check is not None:
            cancel_check()
        if process.poll() is not None:
            raise RuntimeError(f"runtime process exited before ready: exitcode={process.returncode}")
        health = _probe_openai_endpoint(endpoint, timeout_s=2.0)
        if bool(health.get("ok")):
            return
        last_message = str(health.get("message") or last_message)
        time.sleep(2.0)
    raise TimeoutError(f"runtime health check timed out for {endpoint}: {last_message}")


def _stop_ephemeral_runtime(process: subprocess.Popen[bytes]) -> None:
    # start_ephemeral_runtime starts a fresh session, so the launcher pid is the process group id.
    # vLLM may leave engine children alive after the launcher exits; cleanup must target the group.
    _signal_process_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    if _wait_process_group_exit(process.pid, timeout_s=30.0):
        return
    _signal_process_group(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    if not _wait_process_group_exit(process.pid, timeout_s=10.0):
        LOGGER.warning(
            "ephemeral runtime process group still exists after SIGKILL pgid=%s",
            process.pid,
        )


def terminate_runtime_process_group(pgid: int) -> bool:
    _signal_process_group(pgid, signal.SIGTERM)
    if _wait_process_group_exit(pgid, timeout_s=2.0):
        return True
    _signal_process_group(pgid, signal.SIGKILL)
    return _wait_process_group_exit(pgid, timeout_s=2.0)


def _signal_process_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return


def _wait_process_group_exit(pgid: int, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return True
        time.sleep(0.2)
    return not _process_group_exists(pgid)


def _process_group_exists(pgid: int) -> bool:
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"job payload must include non-empty {key!r}.")
    return value.strip()


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"job payload {key!r} must be a string when set.")
    return value.strip() or None


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _require_task(payload: dict[str, Any]) -> TaskKind:
    task = _require_string(payload, "task")
    if task not in {"detection", "keypoint"}:
        raise ValueError(f"unsupported job task={task!r}.")
    return task


def _label_list(value: Any) -> list[str]:
    return normalize_target_labels(value)


def _read_split(path: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _image_path_from_gt(json_relative: Path, payload: dict[str, Any]) -> Path:
    image_path = payload.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        return Path(image_path)
    if len(json_relative.parts) >= 2 and json_relative.parts[1] == "json":
        return Path(json_relative.parts[0]) / "images" / json_relative.with_suffix(".png").name
    return json_relative.with_suffix(".png")


def _resolve_prompt(payload: dict[str, Any], *, task: TaskKind) -> tuple[str, str, str]:
    system_prompt = str(
        payload.get("system_prompt")
        or "You are a visual annotation assistant. Return only valid JSON with no markdown or extra text."
    ).strip()
    prompt_text = payload.get("prompt_text") or payload.get("user_prompt")
    prompt_id = _require_string(payload, "prompt_id")
    if isinstance(prompt_text, str) and prompt_text.strip():
        return system_prompt, prompt_text.strip(), prompt_id
    prompt_path = _optional_string(payload, "prompt_path")
    if prompt_path is None:
        prompt_path = str(_default_prompt_path(prompt_id=prompt_id, task=task))
    loaded_system, loaded_user, loaded_id = _load_prompt_file(_resolve_repo_path(prompt_path))
    return loaded_system or system_prompt, loaded_user, loaded_id or prompt_id


def _prompt_ref_from_payload(payload: dict[str, Any], *, task: TaskKind) -> PromptRef:
    prompt_id = _require_string(payload, "prompt_id")
    prompt_path = _optional_string(payload, "prompt_path")
    system_prompt, user_prompt, resolved_id = _resolve_prompt(payload, task=task)
    source = "inline" if _optional_string(payload, "prompt_text") or _optional_string(payload, "user_prompt") else "file"
    resolved_path = prompt_path
    if resolved_path is None and source == "file":
        resolved_path = str(_default_prompt_path(prompt_id=prompt_id, task=task))
    text_hash = _prompt_hash(system_prompt=system_prompt, user_prompt=user_prompt)
    return PromptRef(
        prompt_id=resolved_id or prompt_id,
        path=resolved_path,
        text_hash=text_hash,
        metadata={
            "source": source,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
    )


def _prompt_hash(*, system_prompt: str, user_prompt: str) -> str:
    payload = json.dumps(
        {"system_prompt": system_prompt, "user_prompt": user_prompt},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default_prompt_path(*, prompt_id: str, task: TaskKind) -> Path:
    lower_id = prompt_id.lower()
    if task == "keypoint" or "keypoint" in lower_id:
        return REPO_ROOT / "configs/prompts/keypoint_arrow.yaml"
    if "arrow" in lower_id:
        return REPO_ROOT / "configs/prompts/grounding_arrow.yaml"
    return REPO_ROOT / "configs/prompts/grounding_layout.yaml"


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


def _load_prompt_file(path: Path) -> tuple[str, str, str | None]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"prompt file must contain a mapping: {path}")
    prompt = payload.get("prompt") or {}
    metadata = payload.get("metadata") or {}
    if not isinstance(prompt, dict):
        raise ValueError(f"prompt file must contain prompt mapping: {path}")
    return (
        str(prompt.get("system_prompt") or "").strip(),
        str(prompt.get("user_prompt") or "").strip(),
        str(metadata.get("id") or "").strip() or None,
    )


def _api_key_from_payload(payload: dict[str, Any]) -> str | None:
    api_key = _optional_string(payload, "api_key")
    if api_key:
        return api_key
    api_key_env = _optional_string(payload, "api_key_env")
    return os.getenv(api_key_env) if api_key_env else None


def _image_size(payload: dict[str, Any], image_path: Path) -> tuple[int, int]:
    width = payload.get("image_width")
    height = payload.get("image_height")
    if width and height:
        return int(width), int(height)
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def _write_raw_output(artifacts: RunArtifacts, *, image: Path, text: str) -> Path:
    image_path = Path(image)
    parts = image_path.parts
    if len(parts) >= 3 and parts[1] == "images":
        relative = Path(parts[0]) / "txt" / image_path.with_suffix(".txt").name
    else:
        relative = image_path.with_suffix(".txt")
    path = artifacts.raw_outputs_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
