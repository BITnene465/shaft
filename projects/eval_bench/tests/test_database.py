from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from eval_bench.database import EvalBenchDatabase


def test_database_persists_jobs_across_instances(tmp_path: Path) -> None:
    first = EvalBenchDatabase(tmp_path)
    job = first.create_job(
        kind="eval",
        payload={
            "model_id": "model-a",
            "benchmark_id": "multitask_val_v1",
            "task": "detection",
        },
    )

    second = EvalBenchDatabase(tmp_path)
    jobs = second.list_jobs()

    assert len(jobs) == 1
    assert jobs[0].job_id == job.job_id
    assert jobs[0].payload["model_id"] == "model-a"


def test_database_rejects_invalid_job_status(tmp_path: Path) -> None:
    database = EvalBenchDatabase(tmp_path)

    with pytest.raises(ValueError, match="unsupported job status"):
        database.create_job(kind="eval", payload={}, status="paused")


def test_database_persists_model_services_across_instances(tmp_path: Path) -> None:
    first = EvalBenchDatabase(tmp_path)
    service = first.upsert_service(
        kind="local_vllm",
        service_id="local-vllm-0",
        config={
            "model_path": "outputs/model/best",
            "port": 8000,
            "cuda_visible_devices": "0,1",
            "tensor_parallel_size": 2,
        },
    )

    second = EvalBenchDatabase(tmp_path)
    services = second.list_services()

    assert len(services) == 1
    assert services[0].service_id == service.service_id
    assert services[0].kind == "local_vllm"
    assert services[0].status == "registered"
    assert services[0].config["model_path"] == "outputs/model/best"


def test_database_rejects_invalid_service_status(tmp_path: Path) -> None:
    database = EvalBenchDatabase(tmp_path)

    with pytest.raises(ValueError, match="unsupported service status"):
        database.upsert_service(kind="local_vllm", config={}, status="paused")


def test_database_supports_wal_and_concurrent_read_write(tmp_path: Path) -> None:
    database = EvalBenchDatabase(tmp_path)
    with database.engine.begin() as connection:
        journal_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()
    assert str(journal_mode).lower() == "wal"

    def create_job(index: int) -> str:
        local = EvalBenchDatabase(tmp_path)
        return local.create_job(kind="eval", payload={"index": index}).job_id

    def list_jobs() -> int:
        local = EvalBenchDatabase(tmp_path)
        return len(local.list_jobs(limit=1000))

    with ThreadPoolExecutor(max_workers=6) as executor:
        created = list(executor.map(create_job, range(12)))
        listed = list(executor.map(lambda _: list_jobs(), range(6)))

    assert len(set(created)) == 12
    assert max(listed) == 12
