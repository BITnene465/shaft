from __future__ import annotations

from typing import Any


TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
ACTIVE_JOB_STATUSES = {"queued", "running"}


def job_cancel_requested(job: Any) -> bool:
    metadata = job.metadata if isinstance(getattr(job, "metadata", None), dict) else {}
    return getattr(job, "status", "") == "cancelled" or bool(metadata.get("cancel_requested"))


def job_holds_scheduler_resources(job: Any) -> bool:
    status = str(getattr(job, "status", ""))
    if status == "running":
        return True
    return status == "cancelled" and job_cancel_requested(job)


def job_is_terminal(job: Any) -> bool:
    return str(getattr(job, "status", "")) in TERMINAL_JOB_STATUSES
