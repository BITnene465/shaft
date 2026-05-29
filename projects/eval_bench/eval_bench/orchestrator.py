from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import threading
from typing import Any

from .artifacts import DEFAULT_STORE_ROOT
from .database import EvalBenchDatabase, JobRecord
from .job_lifecycle import job_holds_scheduler_resources, mark_worker_failure
from . import runtime_resources
from .schema import utc_now_iso


LOGGER = logging.getLogger("eval_bench.orchestrator")


@dataclass(frozen=True)
class SchedulerConfig:
    max_concurrent_jobs: int = 2
    interval_s: float = 2.0


class EvalBenchOrchestrator:
    """Background scheduler for Eval Bench jobs.

    The orchestrator only schedules jobs. The worker still owns the job execution
    and artifact-writing semantics.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_STORE_ROOT,
        *,
        config: SchedulerConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config or SchedulerConfig()
        if self.config.max_concurrent_jobs <= 0:
            raise ValueError("max_concurrent_jobs must be > 0")
        if self.config.interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self.database = EvalBenchDatabase(self.root)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._loop_thread: threading.Thread | None = None
        self._worker_threads: dict[str, threading.Thread] = {}

    @classmethod
    def from_env(cls, root: str | Path = DEFAULT_STORE_ROOT) -> "EvalBenchOrchestrator":
        return cls(
            root,
            config=SchedulerConfig(
                max_concurrent_jobs=_env_int("EVAL_BENCH_SCHEDULER_MAX_CONCURRENT_JOBS", 2),
                interval_s=_env_float("EVAL_BENCH_SCHEDULER_INTERVAL_S", 2.0),
            ),
        )

    def start(self) -> None:
        with self._lock:
            if self._loop_thread is not None and self._loop_thread.is_alive():
                return
            self._stop_event.clear()
            self._loop_thread = threading.Thread(
                target=self._run_loop,
                name="eval-bench-orchestrator",
                daemon=True,
            )
            self._loop_thread.start()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._loop_thread
        if thread is not None:
            thread.join(timeout=timeout_s)

    def status(self) -> dict[str, Any]:
        running_jobs = self.live_running_jobs()
        return {
            "enabled": self._loop_thread is not None,
            "loop_alive": bool(self._loop_thread and self._loop_thread.is_alive()),
            "max_concurrent_jobs": self.config.max_concurrent_jobs,
            "interval_s": self.config.interval_s,
            "live_running_jobs": [job.job_id for job in running_jobs],
            "live_running_count": len(running_jobs),
            "active_worker_threads": [
                job_id for job_id, thread in self._worker_threads.items() if thread.is_alive()
            ],
            "reserved_cuda_devices": sorted(_reserved_cuda_devices(running_jobs)),
            "reserved_runtime_ports": sorted(_reserved_runtime_ports(running_jobs)),
        }

    def live_running_jobs(self) -> list[JobRecord]:
        return [
            job
            for job in self.database.matching_jobs()
            if job_holds_scheduler_resources(job) and _job_has_live_process(job)
        ]

    def schedule_once(self) -> list[JobRecord]:
        """Claim and start all currently schedulable queued eval jobs."""
        with self._lock:
            self._prune_threads()
            running_jobs = self.live_running_jobs()
            available_slots = self.config.max_concurrent_jobs - len(running_jobs)
            if available_slots <= 0:
                return []
            reserved_devices = _reserved_cuda_devices(running_jobs)
            reserved_ports = _reserved_runtime_ports(running_jobs)
            launched: list[JobRecord] = []
            for job in _fifo_jobs(self.database.matching_jobs(kind="eval", status="queued")):
                if job.status != "queued" or job.kind != "eval":
                    continue
                if len(launched) >= available_slots:
                    break
                fit = _resource_fit(job, reserved_devices, reserved_ports)
                if not fit.ok:
                    self.database.update_job(
                        job.job_id,
                        status="queued",
                        metadata_update={
                            "scheduler_last_checked_at": utc_now_iso(),
                            "scheduler_blocked_reason": fit.reason,
                        },
                    )
                    continue
                claimed = self.database.claim_job(job.job_id)
                if claimed is None:
                    continue
                claimed = self.database.update_job(
                    claimed.job_id,
                    status="running",
                    metadata_update={
                        "dashboard_worker_pid": os.getpid(),
                        "scheduler_pid": os.getpid(),
                        "scheduler_claimed_at": utc_now_iso(),
                        "scheduler_blocked_reason": "",
                        "progress_phase": "worker_starting",
                        "progress_message": "Eval Bench scheduler claimed the job.",
                        "progress_updated_at": utc_now_iso(),
                    },
                )
                thread = threading.Thread(
                    target=self._run_job,
                    args=(claimed.job_id,),
                    name=f"eval-bench-job-{claimed.job_id}",
                    daemon=True,
                )
                self._worker_threads[claimed.job_id] = thread
                thread.start()
                launched.append(claimed)
                reserved_devices.update(fit.devices)
                reserved_ports.update(fit.ports)
            return launched

    def _run_loop(self) -> None:
        LOGGER.info(
            "orchestrator started max_concurrent_jobs=%s interval_s=%s",
            self.config.max_concurrent_jobs,
            self.config.interval_s,
        )
        while not self._stop_event.is_set():
            try:
                self.schedule_once()
            except Exception:
                LOGGER.exception("orchestrator tick failed")
            self._stop_event.wait(self.config.interval_s)
        LOGGER.info("orchestrator stopped")

    def _run_job(self, job_id: str) -> None:
        try:
            self._load_worker_class()(self.root).process_job(job_id)
        except Exception as exc:
            LOGGER.exception("orchestrator worker failed job_id=%s error=%s", job_id, exc)
            mark_worker_failure(self.database, job_id, exc, source="scheduler", logger=LOGGER)
        finally:
            with self._lock:
                self._worker_threads.pop(job_id, None)

    def _load_worker_class(self) -> type[Any]:
        from .worker import EvalBenchWorker

        return EvalBenchWorker

    def _prune_threads(self) -> None:
        self._worker_threads = {
            job_id: thread for job_id, thread in self._worker_threads.items() if thread.is_alive()
        }


@dataclass(frozen=True)
class _ResourceFit:
    ok: bool
    devices: frozenset[str]
    ports: frozenset[int]
    reason: str = ""


def _resource_fit(
    job: JobRecord,
    reserved_devices: set[str],
    reserved_ports: set[int],
) -> _ResourceFit:
    devices = _job_cuda_devices(job)
    ports = _job_runtime_ports(job)
    tensor_parallel_size = _optional_int(job.payload.get("tensor_parallel_size"))
    if tensor_parallel_size is not None and devices and tensor_parallel_size > len(devices):
        return _ResourceFit(
            ok=False,
            devices=frozenset(devices),
            ports=frozenset(ports),
            reason=(
                f"tensor_parallel_size={tensor_parallel_size} exceeds declared CUDA "
                f"devices={','.join(sorted(devices))}"
            ),
        )
    overlap = devices & reserved_devices
    if overlap:
        return _ResourceFit(
            ok=False,
            devices=frozenset(devices),
            ports=frozenset(ports),
            reason=f"CUDA devices already reserved: {','.join(sorted(overlap))}",
        )
    port_overlap = ports & reserved_ports
    if port_overlap:
        return _ResourceFit(
            ok=False,
            devices=frozenset(devices),
            ports=frozenset(ports),
            reason=f"runtime ports already reserved: {','.join(str(port) for port in sorted(port_overlap))}",
        )
    return _ResourceFit(ok=True, devices=frozenset(devices), ports=frozenset(ports))


def _reserved_cuda_devices(jobs: list[JobRecord]) -> set[str]:
    devices: set[str] = set()
    for job in jobs:
        devices.update(_job_cuda_devices(job))
    return devices


def _reserved_runtime_ports(jobs: list[JobRecord]) -> set[int]:
    ports: set[int] = set()
    for job in jobs:
        ports.update(_job_runtime_ports(job))
    return ports


def _job_cuda_devices(job: JobRecord) -> set[str]:
    candidates = [
        job.payload.get("cuda_visible_devices"),
        _nested_get(job.payload, ("manifest", "runtime", "env", "CUDA_VISIBLE_DEVICES")),
        _nested_get(job.payload, ("job_manifest", "runtime", "env", "CUDA_VISIBLE_DEVICES")),
    ]
    for value in candidates:
        if value is None:
            continue
        return {
            item.strip()
            for item in str(value).split(",")
            if item.strip() and item.strip() != "-1"
        }
    if _job_uses_ephemeral_runtime(job):
        detected = [gpu.index for gpu in runtime_resources.detect_cuda_devices()]
        tensor_parallel_size = _optional_int(job.payload.get("tensor_parallel_size"))
        if detected and tensor_parallel_size is not None:
            return set(detected[:tensor_parallel_size])
        return set(detected)
    return set()


def _job_runtime_ports(job: JobRecord) -> set[int]:
    if not _job_uses_ephemeral_runtime(job):
        return set()
    candidates = [
        job.payload.get("port"),
        _nested_get(job.payload, ("manifest", "runtime", "args", "port")),
        _nested_get(job.payload, ("job_manifest", "runtime", "args", "port")),
    ]
    for value in candidates:
        parsed = _optional_int(value)
        if parsed is not None:
            return {parsed}
    return set()


def _job_uses_ephemeral_runtime(job: JobRecord) -> bool:
    candidates = [
        job.payload.get("runtime_mode"),
        _nested_get(job.payload, ("manifest", "runtime", "mode")),
        _nested_get(job.payload, ("job_manifest", "runtime", "mode")),
    ]
    for value in candidates:
        if value is not None:
            return str(value) == "ephemeral"
    return False


def _job_has_live_process(job: JobRecord) -> bool:
    metadata = job.metadata if isinstance(job.metadata, dict) else {}
    return any(
        _pid_exists(metadata.get(key))
        for key in ("dashboard_worker_pid", "scheduler_pid", "runtime_pid")
    )


def _pid_exists(pid: Any) -> bool:
    try:
        parsed = int(pid)
    except (TypeError, ValueError):
        return False
    if parsed <= 0:
        return False
    try:
        os.kill(parsed, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _fifo_jobs(jobs: list[JobRecord]) -> list[JobRecord]:
    return sorted(jobs, key=lambda job: (job.created_at, job.job_id))


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        LOGGER.warning("invalid integer env %s=%r, using default=%s", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        LOGGER.warning("invalid float env %s=%r, using default=%s", name, value, default)
        return default
