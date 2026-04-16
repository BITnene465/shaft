from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import gradio as gr

from shaft.webui.services import ShaftRunStore, ShaftSFTTrainService, ShaftWebUIConfigService
from shaft.webui.theme import THEME_INIT_JS, WEBUI_CSS, build_theme
from shaft.webui.types import ShaftRunRecord, ShaftSFTWebUIOverrides


DEFAULT_SFT_CONFIG = "configs/train/train_sft_4b.yaml"


def _as_optional_text(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _as_optional_int(value: str | int | float | None) -> int | None:
    text = str(value).strip() if value is not None else ""
    return int(text) if text else None


def _as_optional_float(value: str | int | float | None) -> float | None:
    text = str(value).strip() if value is not None else ""
    return float(text) if text else None


def _as_optional_bool(value: str | bool | None) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {value!r}")


def _build_overrides(
    run_id: str,
    seed: str,
    epochs: str,
    max_steps: str,
    learning_rate: str,
    train_batch_size: str,
    eval_batch_size: str,
    mix_strategy: str,
    optimizer_name: str,
    scheduler_name: str,
    finetune_mode: str,
    init_from: str,
    resume_from: str,
    use_cpu: str,
) -> ShaftSFTWebUIOverrides:
    return ShaftSFTWebUIOverrides(
        run_id=_as_optional_text(run_id),
        seed=_as_optional_int(seed),
        epochs=_as_optional_int(epochs),
        max_steps=_as_optional_int(max_steps),
        learning_rate=_as_optional_float(learning_rate),
        train_batch_size=_as_optional_int(train_batch_size),
        eval_batch_size=_as_optional_int(eval_batch_size),
        mix_strategy=_as_optional_text(mix_strategy),
        optimizer_name=_as_optional_text(optimizer_name),
        scheduler_name=_as_optional_text(scheduler_name),
        finetune_mode=_as_optional_text(finetune_mode),
        init_from=_as_optional_text(init_from),
        resume_from=_as_optional_text(resume_from),
        use_cpu=_as_optional_bool(use_cpu),
    )


def _render_status_html(
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
        '<div class="shaft-card" style="padding: 18px 18px 14px;">',
        '<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:14px;">',
        '<div>',
        '<div class="shaft-status-kicker">Run Status</div>',
        '<div class="shaft-status-title">SFT Training</div>',
        '</div>',
        f'<span class="{badge_cls}">{escape(status)}</span>',
        '</div>',
        '<div class="shaft-status-hero-card">',
        '<div class="shaft-summary-label">Current Status</div>',
        f'<div class="shaft-status-hero-value">{escape(status)}</div>',
        '</div>',
        '<div class="shaft-meta-list">',
        '<div class="shaft-meta-row">',
        '<span class="shaft-meta-label">Run ID</span>',
        f'<span class="shaft-meta-value">{escape(record.run_id if record is not None else "-")}</span>',
        '</div>',
        '<div class="shaft-meta-row">',
        '<span class="shaft-meta-label">Output</span>',
        f'<span class="shaft-meta-value shaft-meta-value-path">{escape(record.output_dir if record is not None else "-")}</span>',
        '</div>',
        '</div>',
    ]
    parts.append('<div class="shaft-status-divider"></div>')
    parts.append('<div class="shaft-status-grid-secondary">')
    for label, value in secondary_cards:
        parts.extend(
            [
                '<div class="shaft-summary-card shaft-summary-card-secondary">',
                f'<span class="shaft-summary-label">{escape(label)}</span>',
                f'<span class="shaft-summary-value shaft-summary-value-secondary">{escape(str(value))}</span>',
                '</div>',
            ]
        )
    parts.append("</div>")
    if message:
        parts.append(
            f'<div class="shaft-note shaft-note-info">{escape(message)}</div>'
        )
    if error:
        parts.append(
            f'<div class="shaft-note shaft-note-error">{escape(error)}</div>'
        )
    if summary.get("best_model_checkpoint"):
        parts.append(
            '<div class="shaft-note shaft-note-neutral">'
            f'Latest Best Checkpoint: {escape(str(summary["best_model_checkpoint"]))}</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _build_runs_table(records: list[ShaftRunRecord]) -> list[list[str]]:
    rows: list[list[str]] = []
    for record in records:
        rows.append(
            [
                record.run_id,
                record.status,
                str(record.pid or "-"),
                str(record.return_code if record.return_code is not None else "-"),
                record.output_dir,
                record.started_at or "-",
            ]
        )
    return rows


def _field_label(label: str) -> str:
    return f'<div class="shaft-field-label">{escape(label)}</div>'


def _section_title_html(*, kicker: str, title: str, icon_svg: str) -> str:
    return (
        '<div class="shaft-section-title">'
        f'<span class="shaft-section-icon">{icon_svg}</span>'
        "<div>"
        f"<small>{escape(kicker)}</small>"
        f"<strong>{escape(title)}</strong>"
        "</div>"
        "</div>"
    )


def _hero_markup() -> str:
    art_svg = """
    <svg viewBox="0 0 360 220" aria-hidden="true" role="img">
      <defs>
        <linearGradient id="shaftHeroStroke" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="currentColor" stop-opacity="0.88" />
          <stop offset="100%" stop-color="currentColor" stop-opacity="0.42" />
        </linearGradient>
        <radialGradient id="shaftHeroNode" cx="50%" cy="38%" r="70%">
          <stop offset="0%" stop-color="currentColor" stop-opacity="0.95" />
          <stop offset="100%" stop-color="currentColor" stop-opacity="0.58" />
        </radialGradient>
      </defs>
      <g fill="none" stroke="url(#shaftHeroStroke)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <rect x="26" y="60" width="126" height="82" rx="22" />
        <rect x="202" y="26" width="144" height="110" rx="24" />
        <path d="M152 72 L184 48 L212 60" stroke-opacity="0.72" />
        <path d="M152 94 L202 84" stroke-opacity="0.60" />
        <path d="M74 142 L104 178 L168 178 L196 124" />
        <path d="M86 142 L74 142" stroke-opacity="0.42" />
        <path d="M216 76 L244 92 L274 82 L308 104" />
        <path d="M220 106 L248 126 L276 122 L324 142" />
        <path d="M200 124 L220 106" stroke-opacity="0.66" />
        <path d="M168 178 L208 178 L226 152" stroke-opacity="0.42" />
        <path d="M34 194 H332" stroke-opacity="0.18" stroke-dasharray="5 11" />
      </g>
      <g fill="url(#shaftHeroNode)" stroke="none">
        <circle cx="74" cy="142" r="6.2" />
        <circle cx="104" cy="178" r="6.0" />
        <circle cx="168" cy="178" r="5.8" />
        <circle cx="196" cy="124" r="5.9" />
        <circle cx="216" cy="76" r="5.1" />
        <circle cx="244" cy="92" r="5.2" />
        <circle cx="274" cy="82" r="5.1" />
        <circle cx="308" cy="104" r="5.2" />
        <circle cx="220" cy="106" r="5.0" />
        <circle cx="248" cy="126" r="5.1" />
        <circle cx="276" cy="122" r="5.0" />
        <circle cx="324" cy="142" r="5.2" />
      </g>
    </svg>
    """
    return (
        '<div class="shaft-shell">'
        '<div class="shaft-hero">'
        '<div class="shaft-hero-copy">'
        '<div class="shaft-hero-kicker">SFT Research Console</div>'
        '<div class="shaft-hero-title">Shaft</div>'
        '<div class="shaft-hero-subtitle">HF-first Multimodal Training Workspace</div>'
        '<div class="shaft-hero-meta">'
        '<span class="shaft-chip">YAML-first</span>'
        '<span class="shaft-chip">CLI-backed</span>'
        '<span class="shaft-chip">Engineer-facing</span>'
        '</div>'
        '<button type="button" class="shaft-theme-toggle" onclick="window.__shaftToggleTheme && window.__shaftToggleTheme()">'
        '<svg viewBox="0 0 24 24" aria-hidden="true">'
        '<circle cx="12" cy="12" r="4.2"></circle>'
        '<path d="M12 2.8V5.4M12 18.6V21.2M21.2 12H18.6M5.4 12H2.8M18.3 5.7L16.4 7.6M7.6 16.4L5.7 18.3M18.3 18.3L16.4 16.4M7.6 7.6L5.7 5.7"></path>'
        '</svg>'
        '<span>Theme · <span data-shaft-theme-value>Light</span></span>'
        '</button>'
        '</div>'
        f'<div class="shaft-hero-art">{art_svg}</div>'
        '</div>'
        '</div>'
    )


def create_app(
    *,
    default_config_path: str = DEFAULT_SFT_CONFIG,
    config_service: ShaftWebUIConfigService | None = None,
    train_service: ShaftSFTTrainService | None = None,
) -> gr.Blocks:
    config_service = config_service or ShaftWebUIConfigService()
    train_service = train_service or ShaftSFTTrainService(run_store=ShaftRunStore())
    try:
        default_yaml_text = config_service.read_config_text(default_config_path)
        default_status = _render_status_html(
            None,
            message="Loaded default SFT config. Edit YAML directly for advanced fields, or use quick overrides.",
        )
    except Exception as exc:  # noqa: BLE001
        default_yaml_text = ""
        default_status = _render_status_html(None, error=str(exc))

    def _load_config(config_path: str):
        try:
            yaml_text = config_service.read_config_text(config_path)
            return (
                yaml_text,
                _render_status_html(None, message=f"Loaded base config: {config_path}"),
                "",
                "",
                "",
                _build_runs_table(train_service.list_runs()),
                "",
            )
        except Exception as exc:  # noqa: BLE001
            return (
                gr.update(),
                _render_status_html(None, error=str(exc)),
                "",
                "",
                "",
                _build_runs_table(train_service.list_runs()),
                "",
            )

    def _validate(
        config_path: str,
        yaml_text: str,
        run_id: str,
        seed: str,
        epochs: str,
        max_steps: str,
        learning_rate: str,
        train_batch_size: str,
        eval_batch_size: str,
        mix_strategy: str,
        optimizer_name: str,
        scheduler_name: str,
        finetune_mode: str,
        init_from: str,
        resume_from: str,
        use_cpu: str,
    ):
        try:
            overrides = _build_overrides(
                run_id,
                seed,
                epochs,
                max_steps,
                learning_rate,
                train_batch_size,
                eval_batch_size,
                mix_strategy,
                optimizer_name,
                scheduler_name,
                finetune_mode,
                init_from,
                resume_from,
                use_cpu,
            )
            config, resolved_yaml = config_service.resolve_sft_config(
                config_path=config_path,
                yaml_text=yaml_text,
                overrides=overrides,
            )
            message = (
                f"Validated SFT config. datasets={len(config.data.datasets)} "
                f"eval_enabled={config.eval.enabled} model={config.model.model_type}"
            )
            return (
                _render_status_html(None, message=message),
                resolved_yaml,
                "python scripts/train.py sft --config <resolved_config.yaml>",
                _build_runs_table(train_service.list_runs()),
            )
        except Exception as exc:  # noqa: BLE001
            return (
                _render_status_html(None, error=str(exc)),
                "",
                "",
                _build_runs_table(train_service.list_runs()),
            )

    def _start(
        config_path: str,
        yaml_text: str,
        run_id: str,
        seed: str,
        epochs: str,
        max_steps: str,
        learning_rate: str,
        train_batch_size: str,
        eval_batch_size: str,
        mix_strategy: str,
        optimizer_name: str,
        scheduler_name: str,
        finetune_mode: str,
        init_from: str,
        resume_from: str,
        use_cpu: str,
    ):
        try:
            overrides = _build_overrides(
                run_id,
                seed,
                epochs,
                max_steps,
                learning_rate,
                train_batch_size,
                eval_batch_size,
                mix_strategy,
                optimizer_name,
                scheduler_name,
                finetune_mode,
                init_from,
                resume_from,
                use_cpu,
            )
            config, resolved_yaml = config_service.resolve_sft_config(
                config_path=config_path,
                yaml_text=yaml_text,
                overrides=overrides,
            )
            record = train_service.start_run(
                config_source_path=config_path,
                resolved_yaml_text=resolved_yaml,
                config=config,
            )
            resolved_yaml = Path(record.resolved_config_path).read_text(encoding="utf-8")
            return (
                record.run_id,
                _render_status_html(record, message="SFT training started."),
                resolved_yaml,
                " ".join(record.command),
                train_service.read_log(record.run_id),
                _build_runs_table(train_service.list_runs()),
            )
        except Exception as exc:  # noqa: BLE001
            return (
                "",
                _render_status_html(None, error=str(exc)),
                "",
                "",
                "",
                _build_runs_table(train_service.list_runs()),
            )

    def _refresh(current_run_id: str):
        run_id = str(current_run_id or "").strip()
        records = train_service.list_runs()
        if not run_id:
            return (
                current_run_id,
                _render_status_html(None, message="Refreshed recent runs."),
                "",
                "",
                _build_runs_table(records),
            )
        record = train_service.refresh_run(run_id)
        if record is None:
            return (
                "",
                _render_status_html(None, error=f"Run not found: {run_id}"),
                "",
                "",
                _build_runs_table(records),
            )
        summary = train_service.load_summary(run_id)
        return (
            record.run_id,
            _render_status_html(record, summary=summary),
            " ".join(record.command),
            train_service.read_log(run_id),
            _build_runs_table(records),
        )

    def _stop(current_run_id: str):
        run_id = str(current_run_id or "").strip()
        if not run_id:
            return (
                "",
                _render_status_html(None, error="No active run is selected."),
                "",
                "",
                _build_runs_table(train_service.list_runs()),
            )
        record = train_service.stop_run(run_id)
        records = train_service.list_runs()
        if record is None:
            return (
                "",
                _render_status_html(None, error=f"Run not found: {run_id}"),
                "",
                "",
                _build_runs_table(records),
            )
        summary = train_service.load_summary(run_id)
        return (
            record.run_id,
            _render_status_html(record, summary=summary, message="Run stopped."),
            " ".join(record.command),
            train_service.read_log(run_id),
            _build_runs_table(records),
        )

    with gr.Blocks(
        title="Shaft Web UI",
    ) as demo:
        current_run_id = gr.State(value="")
        gr.HTML(_hero_markup())
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="shaft-pane"):
                gr.HTML(
                    _section_title_html(
                        kicker="Compose",
                        title="Training Configuration",
                        icon_svg=(
                            '<svg viewBox="0 0 24 24" aria-hidden="true">'
                            '<path d="M5 6.5H19M5 12H15M5 17.5H13"></path>'
                            '<path d="M16.5 15.5L19 18L22 14"></path>'
                            '</svg>'
                        ),
                    )
                )
                with gr.Column(elem_classes="shaft-launch-strip"):
                    gr.HTML(
                        """
                        <div class="shaft-subsection-copy shaft-launch-copy">
                          Start from a base YAML, validate the resolved runtime config, then launch or inspect the run from the monitor pane.
                        </div>
                        """
                    )
                    with gr.Row(elem_classes="shaft-inline-bar shaft-base-row"):
                        with gr.Column(scale=6, elem_classes="shaft-field-stack"):
                            gr.HTML(_field_label("Base Config Path"))
                            config_path = gr.Textbox(
                                value=default_config_path,
                                show_label=False,
                                container=False,
                                elem_classes="shaft-control shaft-input-control shaft-base-config",
                            )
                        load_config_btn = gr.Button("Load Base Config", variant="secondary", scale=2)
                    with gr.Row(elem_classes="shaft-action-row shaft-action-strip"):
                        validate_btn = gr.Button("Validate", variant="secondary")
                        start_btn = gr.Button("Start SFT Run", elem_id="shaft-start-btn")
                        stop_btn = gr.Button("Stop Run", elem_id="shaft-stop-btn")
                        refresh_btn = gr.Button("Refresh", variant="secondary")
                with gr.Tabs(elem_classes="shaft-tabs"):
                    with gr.Tab("Quick Overrides"):
                        with gr.Column(elem_classes="shaft-overrides-surface"):
                            gr.HTML(
                                """
                                <div class="shaft-subsection-copy">
                                  High-signal overrides for fast iteration. Leave any field blank to preserve the base YAML.
                                </div>
                                """
                            )
                            gr.HTML('<div class="shaft-override-group-title">Run Context</div>')
                            with gr.Row(elem_classes="shaft-override-row"):
                                with gr.Column(elem_classes="shaft-field-stack"):
                                    gr.HTML(_field_label("Run ID"))
                                    run_id = gr.Textbox(
                                        placeholder="Optional",
                                        show_label=False,
                                        container=False,
                                        elem_classes="shaft-control shaft-input-control",
                                    )
                                with gr.Column(elem_classes="shaft-field-stack"):
                                    gr.HTML(_field_label("Seed"))
                                    seed = gr.Textbox(
                                        placeholder="Optional",
                                        show_label=False,
                                        container=False,
                                        elem_classes="shaft-control shaft-input-control",
                                    )
                                with gr.Column(elem_classes="shaft-field-stack"):
                                    gr.HTML(_field_label("Finetune Mode"))
                                    finetune_mode = gr.Dropdown(
                                        choices=["", "full", "lora", "dora", "qlora"],
                                        value="",
                                        show_label=False,
                                        container=False,
                                        elem_classes="shaft-control shaft-input-control",
                                    )
                            gr.HTML('<div class="shaft-override-group-title">Optimization Schedule</div>')
                            with gr.Row(elem_classes="shaft-override-row"):
                                with gr.Column(elem_classes="shaft-field-stack"):
                                    gr.HTML(_field_label("Epochs"))
                                    epochs = gr.Textbox(
                                        placeholder="Optional",
                                        show_label=False,
                                        container=False,
                                        elem_classes="shaft-control shaft-input-control",
                                    )
                                with gr.Column(elem_classes="shaft-field-stack"):
                                    gr.HTML(_field_label("Max Steps"))
                                    max_steps = gr.Textbox(
                                        placeholder="Optional",
                                        show_label=False,
                                        container=False,
                                        elem_classes="shaft-control shaft-input-control",
                                    )
                                with gr.Column(elem_classes="shaft-field-stack"):
                                    gr.HTML(_field_label("Learning Rate"))
                                    learning_rate = gr.Textbox(
                                        placeholder="Optional",
                                        show_label=False,
                                        container=False,
                                        elem_classes="shaft-control shaft-input-control",
                                    )
                            with gr.Row(elem_classes="shaft-override-row"):
                                with gr.Column(elem_classes="shaft-field-stack"):
                                    gr.HTML(_field_label("Train Batch Size"))
                                    train_batch_size = gr.Textbox(
                                        placeholder="Optional",
                                        show_label=False,
                                        container=False,
                                        elem_classes="shaft-control shaft-input-control",
                                    )
                                with gr.Column(elem_classes="shaft-field-stack"):
                                    gr.HTML(_field_label("Eval Batch Size"))
                                    eval_batch_size = gr.Textbox(
                                        placeholder="Optional",
                                        show_label=False,
                                        container=False,
                                        elem_classes="shaft-control shaft-input-control",
                                    )
                                with gr.Column(elem_classes="shaft-field-stack"):
                                    gr.HTML(_field_label("Mix Strategy"))
                                    mix_strategy = gr.Dropdown(
                                        choices=["", "concat", "interleave_under", "interleave_over"],
                                        value="",
                                        show_label=False,
                                        container=False,
                                    elem_classes="shaft-control shaft-input-control",
                                )
                            with gr.Accordion("Advanced Runtime and Checkpointing", open=False, elem_classes="shaft-inline-accordion"):
                                with gr.Row(elem_classes="shaft-override-row"):
                                    with gr.Column(elem_classes="shaft-field-stack"):
                                        gr.HTML(_field_label("Optimizer"))
                                        optimizer_name = gr.Textbox(
                                            placeholder="Optional",
                                            show_label=False,
                                            container=False,
                                            elem_classes="shaft-control shaft-input-control",
                                        )
                                    with gr.Column(elem_classes="shaft-field-stack"):
                                        gr.HTML(_field_label("Scheduler"))
                                        scheduler_name = gr.Textbox(
                                            placeholder="Optional",
                                            show_label=False,
                                            container=False,
                                            elem_classes="shaft-control shaft-input-control",
                                        )
                                    with gr.Column(elem_classes="shaft-field-stack"):
                                        gr.HTML(_field_label("Use CPU"))
                                        use_cpu = gr.Dropdown(
                                            choices=["", "true", "false"],
                                            value="",
                                            show_label=False,
                                            container=False,
                                            elem_classes="shaft-control shaft-input-control",
                                        )
                                with gr.Row(elem_classes="shaft-override-row"):
                                    with gr.Column(elem_classes="shaft-field-stack"):
                                        gr.HTML(_field_label("Init From"))
                                        init_from = gr.Textbox(
                                            placeholder="Optional",
                                            show_label=False,
                                            container=False,
                                            elem_classes="shaft-control shaft-input-control",
                                        )
                                    with gr.Column(elem_classes="shaft-field-stack"):
                                        gr.HTML(_field_label("Resume From"))
                                        resume_from = gr.Textbox(
                                            placeholder="Optional",
                                            show_label=False,
                                            container=False,
                                            elem_classes="shaft-control shaft-input-control",
                                        )
                    with gr.Tab("Editable YAML"):
                        with gr.Column(elem_classes="shaft-yaml-surface"):
                            gr.HTML(
                                """
                                <div class="shaft-subsection-copy">
                                  Full-fidelity source editing. <strong>The source file is never overwritten.</strong>
                                  Each launch writes a run-scoped config to <code>.tmp/webui/runs/&lt;run_id&gt;/resolved_config.yaml</code>.
                                </div>
                                """
                            )
                            gr.HTML(_field_label("Editable YAML"))
                            yaml_editor = gr.Code(
                                language="yaml",
                                value=default_yaml_text,
                                lines=28,
                                elem_classes="shaft-control shaft-code-control",
                                show_label=False,
                                container=False,
                            )
            with gr.Column(scale=4, elem_classes="shaft-pane"):
                gr.HTML(
                    _section_title_html(
                        kicker="Observe",
                        title="Status and Outputs",
                        icon_svg=(
                            '<svg viewBox="0 0 24 24" aria-hidden="true">'
                            '<path d="M4 7.5H20V16.5H4z"></path>'
                            '<path d="M8 19H16"></path>'
                            '<path d="M9 12L11.5 9.5L14 12L16.5 8.5"></path>'
                            '</svg>'
                        ),
                    )
                )
                status_html = gr.HTML(value=default_status)
                with gr.Tabs(elem_classes="shaft-tabs shaft-output-tabs"):
                    with gr.Tab("Resolved Config"):
                        with gr.Column(elem_classes="shaft-output-surface"):
                            gr.HTML(
                                """
                                <div class="shaft-subsection-copy">
                                  Final runtime config after expansion, normalization, and UI overrides. This is the exact config passed to the CLI.
                                </div>
                                """
                            )
                            gr.HTML(_field_label("Resolved Runtime Config"))
                            resolved_yaml = gr.Code(
                                language="yaml",
                                lines=24,
                                elem_classes="shaft-control shaft-code-control",
                                show_label=False,
                                container=False,
                            )
                    with gr.Tab("Command"):
                        with gr.Column(elem_classes="shaft-output-surface"):
                            gr.HTML(
                                """
                                <div class="shaft-subsection-copy">
                                  Canonical CLI launch command for the current run.
                                </div>
                                """
                            )
                            gr.HTML(_field_label("Command Preview"))
                            command_preview = gr.Textbox(
                                interactive=False,
                                lines=3,
                                show_label=False,
                                container=False,
                                elem_classes="shaft-control shaft-preview-control",
                            )
                    with gr.Tab("Logs"):
                        with gr.Column(elem_classes="shaft-output-surface"):
                            gr.HTML(
                                """
                                <div class="shaft-subsection-copy">
                                  Recent stdout and stderr emitted by the active training process.
                                </div>
                                """
                            )
                            gr.HTML(_field_label("Training Log Tail"))
                            log_viewer = gr.Textbox(
                                interactive=False,
                                lines=24,
                                max_lines=32,
                                elem_classes="shaft-control shaft-log-control",
                                show_label=False,
                                container=False,
                            )
                    with gr.Tab("Runs"):
                        with gr.Column(elem_classes="shaft-output-surface"):
                            gr.HTML(
                                """
                                <div class="shaft-subsection-copy">
                                  Recent Web UI launches tracked in the local run store.
                                </div>
                                """
                            )
                            gr.HTML(_field_label("Recent Runs"))
                            runs_table = gr.Dataframe(
                                headers=["run_id", "status", "pid", "return_code", "output_dir", "started_at"],
                                interactive=False,
                                value=_build_runs_table(train_service.list_runs()),
                                row_count=8,
                                column_count=6,
                                column_widths=["18%", "12%", "10%", "10%", "32%", "18%"],
                                elem_classes="shaft-runs",
                                show_label=False,
                            )

        load_config_btn.click(
            fn=_load_config,
            inputs=[config_path],
            outputs=[yaml_editor, status_html, resolved_yaml, command_preview, log_viewer, runs_table, current_run_id],
        )
        validate_btn.click(
            fn=_validate,
            inputs=[
                config_path,
                yaml_editor,
                run_id,
                seed,
                epochs,
                max_steps,
                learning_rate,
                train_batch_size,
                eval_batch_size,
                mix_strategy,
                optimizer_name,
                scheduler_name,
                finetune_mode,
                init_from,
                resume_from,
                use_cpu,
            ],
            outputs=[status_html, resolved_yaml, command_preview, runs_table],
        )
        start_btn.click(
            fn=_start,
            inputs=[
                config_path,
                yaml_editor,
                run_id,
                seed,
                epochs,
                max_steps,
                learning_rate,
                train_batch_size,
                eval_batch_size,
                mix_strategy,
                optimizer_name,
                scheduler_name,
                finetune_mode,
                init_from,
                resume_from,
                use_cpu,
            ],
            outputs=[current_run_id, status_html, resolved_yaml, command_preview, log_viewer, runs_table],
        )
        refresh_btn.click(
            fn=_refresh,
            inputs=[current_run_id],
            outputs=[current_run_id, status_html, command_preview, log_viewer, runs_table],
        )
        stop_btn.click(
            fn=_stop,
            inputs=[current_run_id],
            outputs=[current_run_id, status_html, command_preview, log_viewer, runs_table],
        )
        demo.load(fn=None, js=THEME_INIT_JS, show_progress="hidden")

    return demo


def main(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    base_config_path: str = DEFAULT_SFT_CONFIG,
    share: bool = False,
) -> None:
    demo = create_app(default_config_path=base_config_path)
    launch_kwargs = {
        "server_name": host,
        "share": bool(share),
        "css": WEBUI_CSS,
        "theme": build_theme(),
    }
    if port is not None:
        launch_kwargs["server_port"] = int(port)
    demo.launch(**launch_kwargs)
