from __future__ import annotations

import subprocess
import time

from fastapi.testclient import TestClient


def wait_for_job_status(client: TestClient, job_id: str, status: str) -> dict:
    deadline = time.monotonic() + 5
    latest: dict | None = None
    while time.monotonic() < deadline:
        jobs = client.get("/api/jobs").json()["jobs"]
        latest = next((job for job in jobs if job["job_id"] == job_id), None)
        if latest and latest["status"] == status:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach status={status!r}: {latest}")


def wait_for_process_exit(process: subprocess.Popen[bytes], *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.05)
    return process.poll() is not None
