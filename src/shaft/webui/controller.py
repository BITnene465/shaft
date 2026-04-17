from __future__ import annotations

from html import escape
from typing import Any

from shaft.webui.services import ShaftSFTTrainService, ShaftWebUIConfigService
from shaft.webui.types import ShaftRunRecord, ShaftSFTWebUIOverrides


def _as_optional_text(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _as_optional_int(value: str | int | float | None) -> int | None:
    text = str(value).strip() if value is not None else ""
    return int(text) if text else None


def _as_optional_float(value: str | int | float | None) -> float | None:
    text = str(value).strip() if value is not None else ""
    return float(text) if text else None


def _build_overrides(payload: dict[str, Any]) -> ShaftSFTWebUIOverrides:
    return ShaftSFTWebUIOverrides(
        run_id=_as_optional_text(payload.get("run_id")),
        seed=_as_optional_int(payload.get("seed")),
        epochs=_as_optional_int(payload.get("epochs")),
        learning_rate=_as_optional_float(payload.get("learning_rate")),
        train_batch_size=_as_optional_int(payload.get("train_batch_size")),
        eval_batch_size=_as_optional_int(payload.get("eval_batch_size")),
        mix_strategy=_as_optional_text(payload.get("mix_strategy")),
        finetune_mode=_as_optional_text(payload.get("finetune_mode")),
    )


def render_status_html(
    record: ShaftRunRecord | None,
    *,
    summary: dict[str, Any] | None = None,
    message: str | None = None,
    error: str | None = None,
) -> str:
    summary = summary or {}
    status = record.status if record is not None else ("failed" if error else "validated" if message else "idle")
    badge_cls = f"shaft-status-badge shaft-status-{status}"
    secondary_cards = [
        ("PID", str(record.pid) if record is not None and record.pid is not None else "-"),
        ("Global Step", str(summary.get("global_step", "-"))),
        ("Epoch", str(summary.get("epoch", "-"))),
        ("Best Metric", str(summary.get("best_metric", "-"))),
        ("Return Code", str(record.return_code) if record is not None and record.return_code is not None else "-"),
    ]
    parts = [
        '<div class="shaft-card shaft-status-card">',
        '<div class="shaft-status-head">',
        '<div>',
        '<div class="shaft-status-kicker">Run Status</div>',
        '<div class="shaft-status-title">SFT Training</div>',
        '</div>',
        f'<span class="{badge_cls}">{escape(status)}</span>',
        "</div>",
        '<div class="shaft-status-hero-card">',
        '<div class="shaft-summary-label">Current Status</div>',
        f'<div class="shaft-status-hero-value">{escape(status)}</div>',
        "</div>",
        '<div class="shaft-meta-list">',
        '<div class="shaft-meta-row">',
        '<span class="shaft-meta-label">Run ID</span>',
        f'<span class="shaft-meta-value">{escape(record.run_id if record is not None else "-")}</span>',
        "</div>",
        '<div class="shaft-meta-row">',
        '<span class="shaft-meta-label">Output</span>',
        f'<span class="shaft-meta-value shaft-meta-value-path">{escape(record.output_dir if record is not None else "-")}</span>',
        "</div>",
        "</div>",
        '<div class="shaft-status-divider"></div>',
        '<div class="shaft-status-grid-secondary">',
    ]
    for label, value in secondary_cards:
        parts.extend(
            [
                '<div class="shaft-summary-card shaft-summary-card-secondary">',
                f'<span class="shaft-summary-label">{escape(label)}</span>',
                f'<span class="shaft-summary-value shaft-summary-value-secondary">{escape(str(value))}</span>',
                "</div>",
            ]
        )
    parts.append("</div>")
    if message:
        parts.append(f'<div class="shaft-note shaft-note-info">{escape(message)}</div>')
    if error:
        parts.append(f'<div class="shaft-note shaft-note-error">{escape(error)}</div>')
    if summary.get("best_model_checkpoint"):
        parts.append(
            '<div class="shaft-note shaft-note-neutral">'
            f'Latest Best Checkpoint: {escape(str(summary["best_model_checkpoint"]))}</div>'
        )
    parts.append("</div>")
    return "".join(parts)


class ShaftSFTWebUIController:
    def __init__(
        self,
        *,
        config_service: ShaftWebUIConfigService,
        train_service: ShaftSFTTrainService,
    ) -> None:
        self.config_service = config_service
        self.train_service = train_service

    @staticmethod
    def build_runs_table(records: list[ShaftRunRecord]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for record in records:
            rows.append(
                {
                    "run_id": record.run_id,
                    "status": record.status,
                    "pid": str(record.pid or "-"),
                    "return_code": str(record.return_code if record.return_code is not None else "-"),
                    "output_dir": record.output_dir,
                    "started_at": record.started_at or "-",
                    "is_terminal": "true" if record.is_terminal else "false",
                }
            )
        return rows

    @staticmethod
    def build_run_choices(records: list[ShaftRunRecord], selected_run_id: str | None) -> tuple[list[str], str]:
        choices = [record.run_id for record in records]
        value = selected_run_id if selected_run_id in choices else (choices[0] if choices else "")
        return choices, value

    def build_initial_view(self, default_config_path: str, default_yaml_text: str, default_status: str) -> dict[str, Any]:
        records = self.train_service.list_runs()
        choices, selected_run = self.build_run_choices(records, None)
        return {
            "config_path": default_config_path,
            "yaml_text": default_yaml_text,
            "status_html": default_status,
            "resolved_yaml": "",
            "log_text": "",
            "runs": self.build_runs_table(records),
            "run_choices": choices,
            "selected_run": selected_run,
            "current_run_id": "",
        }

    def load_config(self, config_path: str) -> dict[str, Any]:
        records = self.train_service.list_runs()
        try:
            yaml_text = self.config_service.read_config_text(config_path)
            run_choices, selected_run = self.build_run_choices(records, None)
            return {
                "ok": True,
                "config_path": config_path,
                "yaml_text": yaml_text,
                "status_html": render_status_html(None, message=f"Loaded base config: {config_path}"),
                "resolved_yaml": "",
                "log_text": "",
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
                "current_run_id": "",
            }
        except Exception as exc:  # noqa: BLE001
            run_choices, selected_run = self.build_run_choices(records, None)
            return {
                "ok": False,
                "status_html": render_status_html(None, error=str(exc)),
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
            }

    def validate(
        self,
        *,
        config_path: str,
        yaml_text: str,
        form_payload: dict[str, Any],
    ) -> dict[str, Any]:
        records = self.train_service.list_runs()
        try:
            overrides = _build_overrides(form_payload)
            config, resolved_yaml = self.config_service.resolve_sft_config(
                config_path=config_path,
                yaml_text=yaml_text,
                overrides=overrides,
            )
            run_choices, selected_run = self.build_run_choices(records, None)
            message = (
                f"Validated SFT config. datasets={len(config.data.datasets)} "
                f"eval_enabled={config.eval.enabled} model={config.model.model_type}"
            )
            return {
                "ok": True,
                "status_html": render_status_html(None, message=message),
                "resolved_yaml": resolved_yaml,
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
            }
        except Exception as exc:  # noqa: BLE001
            run_choices, selected_run = self.build_run_choices(records, None)
            return {
                "ok": False,
                "status_html": render_status_html(None, error=str(exc)),
                "resolved_yaml": "",
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
            }

    def start(
        self,
        *,
        config_path: str,
        yaml_text: str,
        form_payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            overrides = _build_overrides(form_payload)
            config, resolved_yaml = self.config_service.resolve_sft_config(
                config_path=config_path,
                yaml_text=yaml_text,
                overrides=overrides,
            )
            record = self.train_service.start_run(
                config_source_path=config_path,
                resolved_yaml_text=resolved_yaml,
                config=config,
            )
            records = self.train_service.list_runs()
            run_choices, selected_run = self.build_run_choices(records, record.run_id)
            return {
                "ok": True,
                "status_html": render_status_html(record, message="SFT training started."),
                "resolved_yaml": self.train_service.read_resolved_config(record.run_id),
                "log_text": self.train_service.read_log(record.run_id),
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
                "current_run_id": record.run_id,
            }
        except Exception as exc:  # noqa: BLE001
            records = self.train_service.list_runs()
            run_choices, selected_run = self.build_run_choices(records, None)
            return {
                "ok": False,
                "status_html": render_status_html(None, error=str(exc)),
                "resolved_yaml": "",
                "log_text": "",
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
                "current_run_id": "",
            }

    def refresh(self, current_run_id: str) -> dict[str, Any]:
        run_id = str(current_run_id or "").strip()
        if not run_id:
            records = self.train_service.list_runs()
            run_choices, selected_run = self.build_run_choices(records, None)
            return {
                "ok": True,
                "status_html": render_status_html(None, message="Refreshed recent runs."),
                "resolved_yaml": "",
                "log_text": "",
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
                "current_run_id": "",
            }
        return self.load_run(run_id)

    def stop(self, current_run_id: str) -> dict[str, Any]:
        run_id = str(current_run_id or "").strip()
        if not run_id:
            records = self.train_service.list_runs()
            run_choices, selected_run = self.build_run_choices(records, None)
            return {
                "ok": False,
                "status_html": render_status_html(None, error="No run is selected."),
                "resolved_yaml": "",
                "log_text": "",
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
                "current_run_id": "",
            }
        record = self.train_service.stop_run(run_id)
        records = self.train_service.list_runs()
        if record is None:
            run_choices, selected_run = self.build_run_choices(records, None)
            return {
                "ok": False,
                "status_html": render_status_html(None, error=f"Run not found: {run_id}"),
                "resolved_yaml": "",
                "log_text": "",
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
                "current_run_id": "",
            }
        snapshot = self.train_service.load_run_snapshot(run_id) or {}
        run_choices, selected_run = self.build_run_choices(records, run_id)
        return {
            "ok": True,
            "status_html": render_status_html(record, summary=snapshot.get("summary"), message="Run stopped."),
            "resolved_yaml": str(snapshot.get("resolved_config", "")),
            "log_text": str(snapshot.get("log", "")),
            "runs": self.build_runs_table(records),
            "run_choices": run_choices,
            "selected_run": selected_run,
            "current_run_id": run_id,
        }

    def load_run(self, run_id: str) -> dict[str, Any]:
        run_id = str(run_id or "").strip()
        records = self.train_service.list_runs()
        if not run_id:
            run_choices, selected_run = self.build_run_choices(records, None)
            return {
                "ok": True,
                "status_html": render_status_html(None, message="No run selected."),
                "resolved_yaml": "",
                "log_text": "",
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
                "current_run_id": "",
            }
        snapshot = self.train_service.load_run_snapshot(run_id)
        if snapshot is None:
            run_choices, selected_run = self.build_run_choices(records, None)
            return {
                "ok": False,
                "status_html": render_status_html(None, error=f"Run not found: {run_id}"),
                "resolved_yaml": "",
                "log_text": "",
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
                "current_run_id": "",
            }
        record = snapshot["record"]
        run_choices, selected_run = self.build_run_choices(records, record.run_id)
        return {
            "ok": True,
            "status_html": render_status_html(record, summary=snapshot["summary"]),
            "resolved_yaml": str(snapshot["resolved_config"]),
            "log_text": str(snapshot["log"]),
            "runs": self.build_runs_table(records),
            "run_choices": run_choices,
            "selected_run": selected_run,
            "current_run_id": record.run_id,
        }

    def delete_run(self, run_id: str, current_run_id: str) -> dict[str, Any]:
        run_id = str(run_id or "").strip()
        records_before = self.train_service.list_runs()
        if not run_id:
            run_choices, selected_run = self.build_run_choices(records_before, current_run_id or None)
            return {
                "ok": False,
                "status_html": render_status_html(None, error="No run is selected for deletion."),
                "runs": self.build_runs_table(records_before),
                "run_choices": run_choices,
                "selected_run": selected_run,
                "current_run_id": current_run_id if current_run_id and current_run_id != run_id else "",
            }
        try:
            deleted = self.train_service.delete_run(run_id)
        except Exception as exc:  # noqa: BLE001
            records = self.train_service.list_runs()
            run_choices, selected_run = self.build_run_choices(records, current_run_id or None)
            return {
                "ok": False,
                "status_html": render_status_html(None, error=str(exc)),
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
                "current_run_id": current_run_id,
            }
        records = self.train_service.list_runs()
        next_current_run_id = "" if current_run_id == run_id else current_run_id
        run_choices, selected_run = self.build_run_choices(records, next_current_run_id or None)
        if not deleted:
            return {
                "ok": False,
                "status_html": render_status_html(None, error=f"Run not found: {run_id}"),
                "runs": self.build_runs_table(records),
                "run_choices": run_choices,
                "selected_run": selected_run,
                "current_run_id": next_current_run_id,
            }
        return {
            "ok": True,
            "status_html": render_status_html(None, message=f"Deleted local Web UI run entry: {run_id}"),
            "resolved_yaml": "" if current_run_id == run_id else None,
            "log_text": "" if current_run_id == run_id else None,
            "runs": self.build_runs_table(records),
            "run_choices": run_choices,
            "selected_run": selected_run,
            "current_run_id": next_current_run_id,
        }
