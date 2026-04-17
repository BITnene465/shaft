from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from shaft.webui.types import ShaftRunRecord


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


class ShaftRunStore:
    def __init__(self, *, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else (_repo_root() / ".tmp" / "webui" / "runs")
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
        temp_path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
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
        records.sort(key=lambda item: (item.started_at or item.created_at or "", item.run_id), reverse=True)
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

    def load_trainer_state_summary(self, output_dir: str | Path) -> dict[str, Any]:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = (_repo_root() / output_path).resolve()
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
