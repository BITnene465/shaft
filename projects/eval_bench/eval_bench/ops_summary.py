from __future__ import annotations

from pathlib import Path
from typing import Any

from .database import EvalBenchDatabase
from .services import EvalBenchServiceManager
from .store import EvalBenchStore, RunSummary


def build_ops_summary(
    store_root: str | Path,
    *,
    scheduler_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store = EvalBenchStore(store_root)
    database = EvalBenchDatabase(store_root)
    service_manager = EvalBenchServiceManager(store_root)
    state = store.state()
    runs = state.runs
    evaluated_runs = sum(1 for run in runs if run.report_path)
    runs_with_predictions = sum(1 for run in runs if run.prediction_count > 0)
    waiting_evaluation = sum(
        1 for run in runs if not run.report_path and run.prediction_count > 0
    )
    queued_jobs = database.job_page(status="queued", limit=1).total
    running_jobs = database.job_page(status="running", limit=1).total
    failed_jobs = database.job_page(status="failed", limit=1).total
    total_jobs = database.job_page(limit=1).total
    total_services = service_manager.service_page(limit=1).total
    running_services = service_manager.service_page(status="running", limit=1).total
    rank_board = store.rank_board(limit=1)
    best_entry = rank_board.entries[0] if rank_board.entries else None
    return {
        "source": "ops_summary",
        "store_root": state.store_root,
        "runs": {
            "total": state.run_count,
            "evaluated": evaluated_runs,
            "with_predictions": runs_with_predictions,
            "waiting_evaluation": waiting_evaluation,
            "best_f1_run": _best_run_payload(best_entry.run_id, runs) if best_entry else None,
            "best_f1": best_entry.f1_iou50 if best_entry else None,
        },
        "benchmarks": {
            "total": state.benchmark_count,
            "sample_count": state.total_benchmark_samples,
            "prediction_count": state.prediction_count,
        },
        "jobs": {
            "total": total_jobs,
            "queued": queued_jobs,
            "running": running_jobs,
            "failed": failed_jobs,
            "active": queued_jobs + running_jobs,
        },
        "services": {
            "total": total_services,
            "running": running_services,
        },
        "scheduler": scheduler_status or {"enabled": False},
    }


def _best_run_payload(run_id: str, runs: list[RunSummary]) -> dict[str, Any] | None:
    for run in runs:
        if run.run_id == run_id:
            return {
                "run_id": run.run_id,
                "status": run.status,
                "benchmark_id": run.benchmark_id,
                "task": run.spec_task,
                "target_labels": run.target_labels,
                "model_id": run.model_id,
                "prompt_id": run.prompt_id,
                "metric_profile": run.metric_profile,
                "prediction_count": run.prediction_count,
                "report_count": run.report_count,
                "created_at": run.created_at,
                "note": run.note,
            }
    return None
