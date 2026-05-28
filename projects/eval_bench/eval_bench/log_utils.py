from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import RunArtifacts


def tail_text_lines(path: Path, *, max_lines: int) -> list[str]:
    if max_lines < 0:
        raise ValueError("max_lines must be >= 0.")
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    return lines if max_lines <= 0 else lines[-max_lines:]


def job_runtime_log_path(store_root: str | Path, job: Any) -> Path:
    root = Path(store_root)
    metadata = job.metadata if isinstance(job.metadata, dict) else {}
    runtime_log_path = metadata.get("runtime_log_path")
    if isinstance(runtime_log_path, str) and runtime_log_path.strip():
        path = Path(runtime_log_path)
        if path.exists():
            return path
    run_id = str(job.payload.get("run_id") or job.job_id)
    active_path = RunArtifacts(root, run_id).logs_dir / "runtime.log"
    if active_path.exists():
        return active_path
    trash_path = root / "trash" / "runs" / run_id / "logs" / "runtime.log"
    if trash_path.exists():
        return trash_path
    suffixed_trash_paths = sorted((root / "trash" / "runs").glob(f"{run_id}.*/logs/runtime.log"))
    if suffixed_trash_paths:
        return suffixed_trash_paths[-1]
    return path if isinstance(runtime_log_path, str) and runtime_log_path.strip() else active_path
