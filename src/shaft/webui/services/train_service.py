from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

import yaml

from shaft.config import RuntimeConfig
from shaft.webui.types import ShaftRunRecord

from .run_store import ShaftRunStore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("webui-%Y%m%d-%H%M%S")


class ShaftSFTTrainService:
    def __init__(self, *, run_store: ShaftRunStore | None = None, repo_root: str | Path | None = None) -> None:
        self.run_store = run_store or ShaftRunStore()
        self.repo_root = Path(repo_root) if repo_root is not None else _repo_root()
        self._processes: dict[str, subprocess.Popen[str]] = {}

    def start_run(
        self,
        *,
        config_source_path: str | Path,
        resolved_yaml_text: str,
        config: RuntimeConfig,
    ) -> ShaftRunRecord:
        run_id = str(config.experiment.run_id or "").strip() or _build_default_run_id()
        config.experiment.run_id = run_id
        _ = self.run_store.prepare_run_dir(run_id)
        resolved_config_path = self.run_store.get_resolved_config_path(run_id)
        payload = yaml.safe_load(resolved_yaml_text) or {}
        experiment_payload = payload.setdefault("experiment", {})
        experiment_payload["run_id"] = run_id
        resolved_yaml_text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        resolved_config_path.write_text(resolved_yaml_text, encoding="utf-8")
        log_path = self.run_store.get_log_path(run_id)
        command = [
            sys.executable,
            "scripts/train.py",
            "sft",
            "--config",
            str(resolved_config_path),
        ]
        env = os.environ.copy()
        src_path = str((self.repo_root / "src").resolve())
        current_pythonpath = str(env.get("PYTHONPATH", "")).strip()
        env["PYTHONPATH"] = src_path if not current_pythonpath else f"{src_path}:{current_pythonpath}"
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        record = ShaftRunRecord(
            run_id=run_id,
            algorithm="sft",
            status="running",
            command=command,
            config_source_path=str(Path(config_source_path)),
            resolved_config_path=str(resolved_config_path),
            log_path=str(log_path),
            output_dir=str(config.experiment.output_dir),
            pid=int(process.pid),
            created_at=_utc_now(),
            started_at=_utc_now(),
        )
        self._processes[run_id] = process
        return self.run_store.save_record(record)

    def refresh_run(self, run_id: str) -> ShaftRunRecord | None:
        record = self.run_store.load_record(run_id)
        if record is None:
            return None
        if record.is_terminal:
            return record
        process = self._processes.get(run_id)
        if process is not None:
            return_code = process.poll()
        else:
            if self._is_pid_running(record.pid):
                return_code = None
            else:
                return_code = record.return_code if record.return_code is not None else -1
        if return_code is None:
            if record.status != "running":
                record.status = "running"
                self.run_store.save_record(record)
            return record
        record.return_code = int(return_code)
        record.finished_at = record.finished_at or _utc_now()
        if record.status != "stopped":
            record.status = "succeeded" if int(return_code) == 0 else "failed"
        self.run_store.save_record(record)
        self._processes.pop(run_id, None)
        return record

    def stop_run(self, run_id: str) -> ShaftRunRecord | None:
        record = self.run_store.load_record(run_id)
        if record is None:
            return None
        if record.is_terminal:
            return record
        process = self._processes.get(run_id)
        pid = int(process.pid) if process is not None else record.pid
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                if process is not None:
                    process.terminate()
        if process is not None:
            try:
                process.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        record.status = "stopped"
        record.return_code = record.return_code if record.return_code is not None else -15
        record.finished_at = _utc_now()
        self._processes.pop(run_id, None)
        return self.run_store.save_record(record)

    def delete_run(self, run_id: str) -> bool:
        record = self.run_store.load_record(run_id)
        if record is None:
            return False
        refreshed = self.refresh_run(run_id) or record
        if not refreshed.is_terminal:
            raise ValueError("Run is still active. Stop it before deleting the local Web UI record.")
        self._processes.pop(run_id, None)
        return self.run_store.delete_run(run_id)

    def list_runs(self, *, limit: int = 20) -> list[ShaftRunRecord]:
        records = self.run_store.list_records(limit=limit)
        refreshed: list[ShaftRunRecord] = []
        for record in records:
            refreshed.append(self.refresh_run(record.run_id) or record)
        return refreshed

    def read_log(self, run_id: str, *, max_chars: int = 12000) -> str:
        return self.run_store.tail_log(run_id, max_chars=max_chars)

    def read_resolved_config(self, run_id: str) -> str:
        return self.run_store.read_resolved_config(run_id)

    def load_summary(self, run_id: str) -> dict[str, Any]:
        record = self.refresh_run(run_id)
        if record is None:
            return {}
        summary = self.run_store.load_trainer_state_summary(record.output_dir, repo_root=self.repo_root)
        summary["status"] = record.status
        summary["return_code"] = record.return_code
        return summary

    def load_finetune_summary(self, run_id: str) -> dict[str, Any]:
        record = self.refresh_run(run_id)
        if record is None:
            return {}
        return self.run_store.load_finetune_summary(record.output_dir, repo_root=self.repo_root)

    def load_run_snapshot(self, run_id: str) -> dict[str, Any] | None:
        record = self.refresh_run(run_id)
        if record is None:
            return None
        return {
            "record": record,
            "summary": self.load_summary(run_id),
            "finetune_summary": self.load_finetune_summary(run_id),
            "resolved_config": self.read_resolved_config(run_id),
            "log": self.read_log(run_id),
        }

    @staticmethod
    def _is_pid_running(pid: int | None) -> bool:
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
