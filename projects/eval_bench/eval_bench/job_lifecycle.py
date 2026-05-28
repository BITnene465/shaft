from __future__ import annotations

import logging
from typing import Any

from .schema import utc_now_iso


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


def mark_worker_failure(
    database: Any,
    job_id: str,
    exc: Exception,
    *,
    source: str,
    logger: logging.Logger,
) -> bool:
    job = database.get_job(job_id)
    if job is None:
        logger.warning("worker failure cannot update unknown job_id=%s", job_id)
        return False
    if str(getattr(job, "status", "")) in {"succeeded", "cancelled"}:
        return False
    error_type = type(exc).__name__
    database.update_job(
        job_id,
        status="failed",
        error=str(exc),
        metadata_update={
            "worker_action": "failed",
            "worker_failure_source": source,
            "worker_error_type": error_type,
            f"{source}_worker_error_type": error_type,
            "progress_phase": "failed",
            "progress_message": str(exc) or error_type,
            "progress_updated_at": utc_now_iso(),
        },
    )
    return True
