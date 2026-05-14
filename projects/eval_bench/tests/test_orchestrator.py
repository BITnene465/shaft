from __future__ import annotations

import os
from pathlib import Path
import time

from eval_bench.database import EvalBenchDatabase
from eval_bench.orchestrator import EvalBenchOrchestrator, SchedulerConfig
from eval_bench.worker import EvalBenchWorker


def _wait_for_status(database: EvalBenchDatabase, job_id: str, status: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = database.get_job(job_id)
        if job and job.status == status:
            return
        time.sleep(0.02)
    latest = database.get_job(job_id)
    raise AssertionError(f"job {job_id} did not reach status={status}: {latest}")


def test_orchestrator_launches_multiple_jobs_when_cuda_devices_do_not_overlap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = EvalBenchDatabase(tmp_path)
    job0 = database.create_job(kind="eval", payload={"cuda_visible_devices": "0"})
    job1 = database.create_job(kind="eval", payload={"cuda_visible_devices": "1"})

    def fake_process_job(self: EvalBenchWorker, job_id: str):
        time.sleep(0.05)
        return self.database.update_job(
            job_id,
            status="succeeded",
            metadata_update={"progress_phase": "succeeded"},
        )

    monkeypatch.setattr(EvalBenchWorker, "process_job", fake_process_job)
    orchestrator = EvalBenchOrchestrator(
        tmp_path,
        config=SchedulerConfig(max_concurrent_jobs=3, interval_s=0.01),
    )

    launched = orchestrator.schedule_once()

    assert {job.job_id for job in launched} == {job0.job_id, job1.job_id}
    _wait_for_status(database, job0.job_id, "succeeded")
    _wait_for_status(database, job1.job_id, "succeeded")


def test_orchestrator_skips_queued_job_when_cuda_device_is_reserved(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = EvalBenchDatabase(tmp_path)
    running = database.create_job(
        kind="eval",
        payload={"cuda_visible_devices": "0"},
        status="running",
        metadata={"dashboard_worker_pid": os.getpid()},
    )
    blocked = database.create_job(kind="eval", payload={"cuda_visible_devices": "0"})
    schedulable = database.create_job(kind="eval", payload={"cuda_visible_devices": "1"})
    orchestrator = EvalBenchOrchestrator(
        tmp_path,
        config=SchedulerConfig(max_concurrent_jobs=3, interval_s=0.01),
    )
    monkeypatch.setattr(
        EvalBenchWorker,
        "process_job",
        lambda self, job_id: self.database.get_job(job_id),
    )

    launched = orchestrator.schedule_once()

    assert [job.job_id for job in launched] == [schedulable.job_id]
    assert database.get_job(running.job_id).status == "running"
    blocked_after = database.get_job(blocked.job_id)
    assert blocked_after.status == "queued"
    assert "already reserved" in blocked_after.metadata["scheduler_blocked_reason"]
    assert database.get_job(schedulable.job_id).status == "running"


def test_orchestrator_reserves_cancel_requested_live_job_resources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = EvalBenchDatabase(tmp_path)
    cancelled_live = database.create_job(
        kind="eval",
        payload={"cuda_visible_devices": "0"},
        status="running",
        metadata={"dashboard_worker_pid": os.getpid()},
    )
    database.cancel_job(cancelled_live.job_id)
    blocked = database.create_job(kind="eval", payload={"cuda_visible_devices": "0"})
    schedulable = database.create_job(kind="eval", payload={"cuda_visible_devices": "1"})
    orchestrator = EvalBenchOrchestrator(
        tmp_path,
        config=SchedulerConfig(max_concurrent_jobs=3, interval_s=0.01),
    )
    monkeypatch.setattr(
        EvalBenchWorker,
        "process_job",
        lambda self, job_id: self.database.get_job(job_id),
    )

    launched = orchestrator.schedule_once()

    assert [job.job_id for job in launched] == [schedulable.job_id]
    assert database.get_job(blocked.job_id).status == "queued"
    assert "already reserved" in database.get_job(blocked.job_id).metadata[
        "scheduler_blocked_reason"
    ]


def test_orchestrator_blocks_invalid_tensor_parallel_request(tmp_path: Path) -> None:
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(
        kind="eval",
        payload={"cuda_visible_devices": "0", "tensor_parallel_size": 2},
    )
    orchestrator = EvalBenchOrchestrator(
        tmp_path,
        config=SchedulerConfig(max_concurrent_jobs=3, interval_s=0.01),
    )

    assert orchestrator.schedule_once() == []
    blocked = database.get_job(job.job_id)

    assert blocked.status == "queued"
    assert "tensor_parallel_size=2" in blocked.metadata["scheduler_blocked_reason"]


def test_orchestrator_reserves_ephemeral_runtime_ports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = EvalBenchDatabase(tmp_path)
    database.create_job(
        kind="eval",
        payload={"runtime_mode": "ephemeral", "port": 8000},
        status="running",
        metadata={"runtime_pid": os.getpid()},
    )
    blocked = database.create_job(
        kind="eval",
        payload={"runtime_mode": "ephemeral", "port": 8000},
    )
    schedulable = database.create_job(
        kind="eval",
        payload={"runtime_mode": "ephemeral", "port": 8001},
    )
    monkeypatch.setattr(
        EvalBenchWorker,
        "process_job",
        lambda self, job_id: self.database.get_job(job_id),
    )
    orchestrator = EvalBenchOrchestrator(
        tmp_path,
        config=SchedulerConfig(max_concurrent_jobs=3, interval_s=0.01),
    )

    launched = orchestrator.schedule_once()

    assert [job.job_id for job in launched] == [schedulable.job_id]
    assert database.get_job(blocked.job_id).status == "queued"
    assert "runtime ports already reserved: 8000" in database.get_job(blocked.job_id).metadata[
        "scheduler_blocked_reason"
    ]
