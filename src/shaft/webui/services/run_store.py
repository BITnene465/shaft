from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from shaft.model.finetune_plan import resolved_finetune_summary_path
from shaft.observability import (
    PROGRESS_SNAPSHOT_FILENAME,
    format_progress_percentage,
    select_progress_display_task_id,
)
from shaft.training.optimizer_plan import resolved_optimizer_summary_path
from shaft.webui.types import ShaftRunRecord


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


class ShaftRunStore:
    def __init__(self, *, root_dir: str | Path | None = None) -> None:
        self.root_dir = (
            Path(root_dir) if root_dir is not None else (_repo_root() / ".tmp" / "webui" / "runs")
        )
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def get_run_dir(self, run_id: str) -> Path:
        return self.root_dir / str(run_id)

    def get_record_path(self, run_id: str) -> Path:
        return self.get_run_dir(run_id) / "run.json"

    def get_log_path(self, run_id: str) -> Path:
        return self.get_run_dir(run_id) / "train.log"

    def get_resolved_config_path(self, run_id: str) -> Path:
        return self.get_run_dir(run_id) / "resolved_config.yaml"

    def prepare_run_dir(self, run_id: str) -> Path:
        run_dir = self.get_run_dir(run_id)
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists for run_id={run_id!r}: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def save_record(self, record: ShaftRunRecord) -> ShaftRunRecord:
        path = self.get_record_path(record.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp_path.replace(path)
        return record

    def delete_run(self, run_id: str) -> bool:
        run_dir = self.get_run_dir(run_id)
        if not run_dir.exists():
            return False
        shutil.rmtree(run_dir)
        return True

    def load_record(self, run_id: str) -> ShaftRunRecord | None:
        path = self.get_record_path(run_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ShaftRunRecord.from_dict(payload)

    def list_records(self, *, limit: int = 50) -> list[ShaftRunRecord]:
        records: list[ShaftRunRecord] = []
        for path in sorted(self.root_dir.glob("*/run.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                records.append(ShaftRunRecord.from_dict(payload))
            except Exception:  # noqa: BLE001
                continue
        records.sort(
            key=lambda item: (item.started_at or item.created_at or "", item.run_id), reverse=True
        )
        return records[:limit]

    def tail_log(self, run_id: str, *, max_chars: int = 12000) -> str:
        path = self.get_log_path(run_id)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]

    def read_resolved_config(self, run_id: str) -> str:
        path = self.get_resolved_config_path(run_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def load_trainer_state_summary(
        self,
        output_dir: str | Path,
        *,
        repo_root: str | Path | None = None,
    ) -> dict[str, Any]:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            base_root = Path(repo_root) if repo_root is not None else _repo_root()
            output_path = (base_root / output_path).resolve()
        candidates = [output_path / "trainer_state.json"]
        checkpoints = sorted(output_path.glob("checkpoint-*/trainer_state.json"))
        if checkpoints:
            candidates.append(checkpoints[-1])
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            return {
                "global_step": payload.get("global_step"),
                "epoch": payload.get("epoch"),
                "best_metric": payload.get("best_metric"),
                "best_model_checkpoint": payload.get("best_model_checkpoint"),
            }
        return {}

    def load_progress_summary(
        self,
        output_dir: str | Path,
        *,
        expected_run_id: str | None = None,
        expected_not_before: str | None = None,
        repo_root: str | Path | None = None,
    ) -> dict[str, Any]:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            base_root = Path(repo_root) if repo_root is not None else _repo_root()
            output_path = (base_root / output_path).resolve()
        path = output_path / PROGRESS_SNAPSHOT_FILENAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return {}
        run_id = str(payload.get("run_id", ""))
        if expected_run_id is not None and run_id != str(expected_run_id):
            return {}
        not_before = _parse_timestamp(expected_not_before)
        if not_before is not None:
            updated_at = _parse_timestamp(payload.get("updated_at"))
            if updated_at is None or updated_at < not_before:
                return {}
        tasks = payload.get("tasks")
        if not isinstance(tasks, dict):
            return {}
        valid_tasks = {
            str(task_id): task for task_id, task in tasks.items() if isinstance(task, dict)
        }
        if not valid_tasks:
            return {
                "progress_status": payload.get("status"),
                "progress_attempt_id": payload.get("attempt_id"),
                "progress_updated_at": payload.get("updated_at"),
            }
        active_task_id = str(payload.get("active_task_id") or "")
        display_task_id = select_progress_display_task_id(
            valid_tasks,
            active_task_id=active_task_id or None,
            status=str(payload.get("status") or ""),
        )
        active_task = valid_tasks[display_task_id] if display_task_id is not None else {}
        train_task = valid_tasks.get("train")
        current = active_task.get("current")
        total = active_task.get("total")
        unit = str(active_task.get("unit") or "it")
        progress_text = f"{current} {unit}"
        progress_percent = None
        if isinstance(current, int) and isinstance(total, int) and total > 0:
            progress_percent = min(max(current / total, 0.0), 1.0)
            progress_text = (
                f"{current}/{total} ({format_progress_percentage(progress_percent)})"
            )
        summary: dict[str, Any] = {
            "progress_status": payload.get("status"),
            "progress_attempt_id": payload.get("attempt_id"),
            "progress_updated_at": payload.get("updated_at"),
            "progress_phase": active_task.get("label"),
            "progress_task_id": active_task.get("task_id"),
            "progress_task_state": active_task.get("state"),
            "progress_current": current,
            "progress_total": total,
            "progress_unit": unit,
            "progress_percent": progress_percent,
            "progress_text": progress_text,
            "progress_message": active_task.get("message"),
            "progress_metrics": _mapping_or_empty(active_task.get("metrics")),
        }
        if train_task is not None:
            summary.update(
                {
                    "global_step": train_task.get("current"),
                    "max_steps": train_task.get("total"),
                    "train_metrics": _mapping_or_empty(train_task.get("metrics")),
                }
            )
        return summary

    def load_finetune_summary(
        self,
        output_dir: str | Path,
        *,
        repo_root: str | Path | None = None,
    ) -> dict[str, Any]:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            base_root = Path(repo_root) if repo_root is not None else _repo_root()
            output_path = (base_root / output_path).resolve()
        path = resolved_finetune_summary_path(output_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def load_optimizer_summary(
        self,
        output_dir: str | Path,
        *,
        repo_root: str | Path | None = None,
    ) -> dict[str, Any]:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            base_root = Path(repo_root) if repo_root is not None else _repo_root()
            output_path = (base_root / output_path).resolve()
        path = resolved_optimizer_summary_path(output_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload
