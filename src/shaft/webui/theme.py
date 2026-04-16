from __future__ import annotations

import gradio as gr


THEME_INIT_JS = """
() => {
  const storageKey = "shaft-webui-theme";

  const applyTheme = (theme) => {
    const resolved = theme === "dark" || theme === "light"
      ? theme
      : (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

    document.documentElement.setAttribute("data-shaft-theme", resolved);
    document.body.setAttribute("data-shaft-theme", resolved);
    document.querySelectorAll(".gradio-container").forEach((node) => {
      node.setAttribute("data-shaft-theme", resolved);
    });
    document.querySelectorAll("[data-shaft-theme-value]").forEach((node) => {
      node.textContent = resolved === "dark" ? "Dark" : "Light";
    });
    return resolved;
  };

  window.__shaftApplyTheme = applyTheme;
  window.__shaftToggleTheme = () => {
    const current = document.documentElement.getAttribute("data-shaft-theme") === "dark" ? "light" : "dark";
    window.localStorage.setItem(storageKey, current);
    return applyTheme(current);
  };

  return applyTheme(window.localStorage.getItem(storageKey));
}
"""


WEBUI_CSS = """
:root {
  --shaft-font-sans: "Segoe UI", "Avenir Next", "Helvetica Neue", Arial, sans-serif;
  --shaft-font-serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  --shaft-font-mono: "SFMono-Regular", "JetBrains Mono", "Consolas", "Liberation Mono", monospace;

  --shaft-bg: #f2ebde;
  --shaft-bg-soft: #e7decd;
  --shaft-shell: rgba(255, 251, 245, 0.96);
  --shaft-panel: rgba(254, 250, 243, 0.98);
  --shaft-panel-soft: rgba(248, 242, 232, 0.90);
  --shaft-control-bg: rgba(255, 252, 247, 0.96);
  --shaft-control-shell: rgba(243, 235, 223, 0.88);
  --shaft-code-bg: rgba(246, 239, 228, 0.98);
  --shaft-code-bg-soft: rgba(239, 231, 217, 0.92);
  --shaft-line: rgba(57, 68, 77, 0.12);
  --shaft-line-strong: rgba(22, 33, 43, 0.16);
  --shaft-ink: #17212b;
  --shaft-muted: #5a6874;
  --shaft-soft-text: #7a8792;
  --shaft-accent: #15706d;
  --shaft-accent-strong: #0f5a57;
  --shaft-accent-soft: rgba(21, 112, 109, 0.12);
  --shaft-gold: #a06a16;
  --shaft-danger: #9e3557;
  --shaft-shadow: 0 18px 42px rgba(22, 33, 43, 0.08);
  --shaft-glow-a: rgba(21, 112, 109, 0.10);
  --shaft-glow-b: rgba(160, 106, 22, 0.08);
}

html[data-shaft-theme="dark"] {
  --shaft-bg: #101924;
  --shaft-bg-soft: #172230;
  --shaft-shell: rgba(19, 29, 40, 0.94);
  --shaft-panel: rgba(20, 30, 42, 0.96);
  --shaft-panel-soft: rgba(24, 36, 50, 0.92);
  --shaft-control-bg: rgba(23, 34, 48, 0.96);
  --shaft-control-shell: rgba(18, 28, 40, 0.96);
  --shaft-code-bg: rgba(17, 27, 39, 0.98);
  --shaft-code-bg-soft: rgba(21, 31, 45, 0.96);
  --shaft-line: rgba(193, 209, 219, 0.10);
  --shaft-line-strong: rgba(193, 209, 219, 0.18);
  --shaft-ink: #edf3f7;
  --shaft-muted: #a8b8c3;
  --shaft-soft-text: #8da1af;
  --shaft-accent: #67d5ca;
  --shaft-accent-strong: #8ae1d8;
  --shaft-accent-soft: rgba(103, 213, 202, 0.12);
  --shaft-gold: #e4b05f;
  --shaft-danger: #ef8bab;
  --shaft-shadow: 0 24px 48px rgba(0, 0, 0, 0.32);
  --shaft-glow-a: rgba(103, 213, 202, 0.10);
  --shaft-glow-b: rgba(228, 176, 95, 0.08);
}

html,
body,
gradio-app,
.gradio-container {
  color-scheme: light;
  min-height: 100%;
  margin: 0;
  background:
    radial-gradient(circle at top left, var(--shaft-glow-a), transparent 26%),
    radial-gradient(circle at top right, var(--shaft-glow-b), transparent 24%),
    linear-gradient(180deg, var(--shaft-bg) 0%, var(--shaft-bg-soft) 100%);
  color: var(--shaft-ink) !important;
  transition: background 180ms ease, color 180ms ease, border-color 180ms ease;
}

html[data-shaft-theme="dark"],
body[data-shaft-theme="dark"],
.gradio-container[data-shaft-theme="dark"] {
  color-scheme: dark;
}

.gradio-container {
  font-family: var(--shaft-font-sans);
  color: var(--shaft-ink) !important;
}

.gradio-container *,
.gradio-container .prose,
.gradio-container .prose * {
  color: inherit;
}

.gradio-container .gr-block,
.gradio-container .gr-box,
.gradio-container .gr-group,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container form,
.gradio-container fieldset {
  background: transparent !important;
  box-shadow: none !important;
}

.shaft-shell {
  max-width: 1500px;
  margin: 0 auto 14px;
  padding: 20px 20px 8px;
}

.shaft-hero {
  position: relative;
  overflow: hidden;
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(240px, 0.9fr);
  gap: 18px;
  align-items: stretch;
  border-radius: 30px;
  padding: 28px 30px 24px;
  border: 1px solid var(--shaft-line);
  background: linear-gradient(180deg, var(--shaft-panel), var(--shaft-panel-soft));
  box-shadow: var(--shaft-shadow);
}

.shaft-hero::before,
.shaft-hero::after {
  content: "";
  position: absolute;
  border-radius: 999px;
  pointer-events: none;
  filter: blur(2px);
}

.shaft-hero::before {
  inset: auto auto -80px -40px;
  width: 220px;
  height: 220px;
  background: radial-gradient(circle, var(--shaft-glow-b), transparent 68%);
}

.shaft-hero::after {
  inset: -60px -20px auto auto;
  width: 260px;
  height: 260px;
  background: radial-gradient(circle, var(--shaft-glow-a), transparent 68%);
}

.shaft-hero-copy {
  position: relative;
  z-index: 1;
}

.shaft-hero-kicker {
  color: var(--shaft-gold);
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-weight: 800;
}

.shaft-hero-title {
  margin: 12px 0 0;
  font-family: var(--shaft-font-serif);
  font-size: 3rem;
  line-height: 0.94;
  letter-spacing: -0.05em;
  color: var(--shaft-ink);
  font-weight: 700;
}

.shaft-hero-subtitle {
  margin: 12px 0 0;
  color: var(--shaft-muted);
  font-size: 1.02rem;
  font-weight: 700;
}

.shaft-hero-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}

.shaft-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 34px;
  padding: 0 14px;
  border-radius: 999px;
  border: 1px solid var(--shaft-line);
  background: rgba(255, 255, 255, 0.30);
  color: var(--shaft-muted);
  font-size: 0.84rem;
  font-weight: 700;
}

.shaft-theme-toggle {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  min-height: 42px;
  margin-top: 18px;
  padding: 0 16px 0 14px;
  border-radius: 999px;
  border: 1px solid var(--shaft-line-strong);
  background: var(--shaft-control-bg);
  color: var(--shaft-ink);
  font: inherit;
  font-size: 0.9rem;
  font-weight: 700;
  cursor: pointer;
  transition: transform 140ms ease, background 140ms ease, border-color 140ms ease;
}

.shaft-theme-toggle:hover {
  transform: translateY(-1px);
  border-color: var(--shaft-accent);
}

.shaft-theme-toggle svg {
  width: 18px;
  height: 18px;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.8;
}

.shaft-hero-art {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 190px;
}

.shaft-hero-art svg {
  width: 100%;
  max-width: 380px;
  height: auto;
}

.shaft-pane {
  border-radius: 30px;
  border: 1px solid var(--shaft-line);
  background: linear-gradient(180deg, var(--shaft-panel), var(--shaft-panel-soft));
  box-shadow: var(--shaft-shadow);
  padding: 14px !important;
}

.shaft-section-title {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0 2px 12px;
}

.shaft-section-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 42px;
  height: 42px;
  border-radius: 16px;
  border: 1px solid var(--shaft-line);
  background: rgba(255, 255, 255, 0.32);
  color: var(--shaft-accent);
}

.shaft-section-icon svg {
  width: 22px;
  height: 22px;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.8;
}

.shaft-section-title small {
  display: block;
  margin-bottom: 2px;
  color: var(--shaft-gold);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-weight: 800;
}

.shaft-section-title strong {
  display: block;
  color: var(--shaft-ink);
  font-size: 1.32rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  font-family: var(--shaft-font-serif);
}

.shaft-inline-bar,
.shaft-action-row {
  gap: 12px;
}

.shaft-launch-strip {
  margin-bottom: 14px;
  padding: 16px;
  border-radius: 24px;
  border: 1px solid var(--shaft-line);
  background: linear-gradient(180deg, rgba(255,255,255,0.18), rgba(255,255,255,0.08));
}

.shaft-action-row {
  margin-top: 10px;
}

.shaft-field-stack {
  gap: 6px;
}

.shaft-field-label {
  margin: 0 2px 4px;
  color: var(--shaft-muted);
  font-size: 0.79rem;
  line-height: 1.1;
  font-weight: 700;
}

.shaft-subsection-copy {
  margin: 2px 2px 14px;
  color: var(--shaft-muted);
  font-size: 0.93rem;
  line-height: 1.56;
}

.shaft-subsection-copy strong {
  color: var(--shaft-ink);
}

.shaft-subsection-copy code {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.26);
  border: 1px solid var(--shaft-line);
  color: var(--shaft-ink);
  font-family: var(--shaft-font-mono);
  font-size: 0.84rem;
}

.shaft-override-group-title {
  margin: 14px 2px 8px;
  color: var(--shaft-gold);
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-weight: 800;
}

.shaft-base-row {
  align-items: end;
}

.shaft-launch-copy {
  margin-bottom: 10px;
}

.shaft-action-strip button {
  min-height: 46px !important;
}

.shaft-overrides-surface,
.shaft-yaml-surface,
.shaft-output-surface {
  padding: 16px;
  border-radius: 24px;
  border: 1px solid var(--shaft-line);
  background: linear-gradient(180deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06));
}

.shaft-inline-accordion {
  margin-top: 10px;
}

.shaft-inline-accordion .label-wrap,
.shaft-inline-accordion summary {
  border-radius: 16px !important;
  background: rgba(255,255,255,0.12) !important;
  border: 1px solid var(--shaft-line) !important;
}

.shaft-override-row {
  margin-bottom: 10px;
  gap: 12px;
  padding: 12px 12px 8px;
  border-radius: 20px;
  border: 1px solid rgba(22, 33, 43, 0.08);
  background: rgba(255,255,255,0.10);
}

.shaft-control,
.shaft-control > div,
.shaft-control > div > div,
.shaft-control .wrap,
.shaft-control .wrap-inner,
.shaft-control .scroll-hide,
.shaft-control label,
.shaft-control form,
.shaft-control fieldset {
  background: transparent !important;
  box-shadow: none !important;
}

.shaft-input-control input,
.shaft-input-control textarea,
.shaft-input-control .wrap,
.shaft-input-control .wrap.default,
.shaft-input-control .inner,
.shaft-input-control [data-testid="textbox"],
.shaft-input-control [data-testid="textbox"] > div,
.shaft-input-control [data-testid="dropdown"],
.shaft-input-control [data-testid="dropdown"] > div,
.shaft-input-control button,
.shaft-input-control .choices {
  border-radius: 18px !important;
  border: 1px solid var(--shaft-line-strong) !important;
  background: var(--shaft-control-bg) !important;
  color: var(--shaft-ink) !important;
}

.shaft-input-control input::placeholder,
.shaft-input-control textarea::placeholder {
  color: var(--shaft-soft-text) !important;
}

.shaft-input-control input,
.shaft-input-control textarea {
  font-size: 0.96rem !important;
}

.shaft-code-control .cm-editor,
.shaft-code-control .cm-editor .cm-scroller,
.shaft-code-control .cm-editor .cm-content,
.shaft-code-control .cm-editor .cm-line,
.shaft-code-control .cm-editor .cm-gutters,
.shaft-code-control .cm-editor .cm-gutter,
.shaft-code-control .cm-editor .cm-activeLine,
.shaft-code-control .cm-editor .cm-activeLineGutter,
.shaft-code-control textarea,
.shaft-code-control .wrap,
.shaft-code-control .wrap textarea,
.shaft-preview-control textarea,
.shaft-preview-control .wrap,
.shaft-log-control textarea,
.shaft-log-control .wrap {
  background: linear-gradient(180deg, var(--shaft-code-bg), var(--shaft-code-bg-soft)) !important;
  color: var(--shaft-ink) !important;
  border: 1px solid var(--shaft-line-strong) !important;
  border-radius: 22px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
}

.shaft-code-control .cm-gutters,
.shaft-code-control .cm-activeLine,
.shaft-code-control .cm-activeLineGutter {
  background: transparent !important;
}

.shaft-code-control .cm-editor,
.shaft-preview-control textarea,
.shaft-log-control textarea,
.shaft-code-control textarea {
  font-family: var(--shaft-font-mono) !important;
}

.shaft-runs table {
  border-radius: 22px;
  overflow: hidden;
  border: 1px solid var(--shaft-line) !important;
}

.shaft-runs table thead th {
  background: rgba(255, 255, 255, 0.18) !important;
  color: var(--shaft-muted) !important;
  border-bottom: 1px solid var(--shaft-line) !important;
  font-weight: 700 !important;
}

.shaft-runs table tbody td {
  background: var(--shaft-control-bg) !important;
  color: var(--shaft-ink) !important;
}

.shaft-card {
  border-radius: 28px;
  border: 1px solid var(--shaft-line);
  background: linear-gradient(180deg, var(--shaft-panel), var(--shaft-panel-soft));
  box-shadow: 0 14px 34px rgba(22, 33, 43, 0.08);
}

.shaft-summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.shaft-status-hero-card {
  margin-bottom: 12px;
  padding: 16px 18px;
  border-radius: 20px;
  border: 1px solid var(--shaft-line);
  background: linear-gradient(180deg, rgba(255,255,255,0.28), rgba(255,255,255,0.12));
}

.shaft-status-hero-value {
  margin-top: 6px;
  color: var(--shaft-ink);
  font-size: 1.8rem;
  font-weight: 700;
  letter-spacing: -0.03em;
}

.shaft-status-grid-primary {
  display: grid;
  grid-template-columns: 1.2fr 1.8fr 0.8fr;
  gap: 12px;
}

.shaft-status-grid-secondary {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.shaft-status-divider {
  height: 1px;
  margin: 14px 0 12px;
  background: linear-gradient(90deg, transparent, var(--shaft-line-strong), transparent);
}

.shaft-meta-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 12px;
}

.shaft-meta-row {
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr);
  gap: 12px;
  align-items: start;
  padding: 12px 14px;
  border-radius: 18px;
  border: 1px solid var(--shaft-line);
  background: var(--shaft-control-bg);
}

.shaft-meta-label {
  color: var(--shaft-muted);
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
}

.shaft-meta-value {
  color: var(--shaft-ink);
  font-size: 0.98rem;
  font-weight: 700;
  word-break: break-word;
}

.shaft-meta-value-path {
  font-family: var(--shaft-font-mono);
  font-size: 0.88rem;
}

.shaft-summary-card {
  border-radius: 18px;
  border: 1px solid var(--shaft-line);
  background: var(--shaft-control-bg);
  padding: 14px 16px;
}

.shaft-status-kicker {
  color: var(--shaft-muted);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
}

.shaft-status-title {
  margin-top: 6px;
  color: var(--shaft-ink);
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: -0.02em;
}

.shaft-summary-label {
  display: block;
  color: var(--shaft-muted);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
}

.shaft-summary-value {
  display: block;
  margin-top: 6px;
  color: var(--shaft-ink);
  font-size: 1.08rem;
  font-weight: 700;
  word-break: break-word;
}

.shaft-summary-card-secondary {
  padding: 12px 14px;
}

.shaft-summary-value-secondary {
  font-size: 1rem;
}

.shaft-status-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 98px;
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 0.84rem;
  font-weight: 800;
}

.shaft-status-running { background: rgba(21, 112, 109, 0.14); color: var(--shaft-accent-strong); }
.shaft-status-succeeded { background: rgba(36, 132, 90, 0.14); color: #17724b; }
.shaft-status-failed { background: rgba(158, 53, 87, 0.12); color: var(--shaft-danger); }
.shaft-status-stopped { background: rgba(160, 106, 22, 0.14); color: var(--shaft-gold); }
.shaft-status-idle,
.shaft-status-validated { background: rgba(104, 118, 130, 0.14); color: var(--shaft-muted); }

.shaft-note {
  margin-top: 14px;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid transparent;
}

.shaft-note-info {
  background: var(--shaft-accent-soft);
  border-color: rgba(21, 112, 109, 0.18);
  color: var(--shaft-accent-strong);
}

.shaft-note-error {
  background: rgba(158, 53, 87, 0.10);
  border-color: rgba(158, 53, 87, 0.20);
  color: var(--shaft-danger);
}

.shaft-note-neutral {
  background: rgba(22, 33, 43, 0.06);
  border-color: rgba(22, 33, 43, 0.12);
  color: var(--shaft-muted);
}

#shaft-start-btn button {
  min-height: 52px;
  border-radius: 18px;
  border: 1px solid transparent !important;
  background: linear-gradient(135deg, var(--shaft-accent), var(--shaft-accent-strong)) !important;
  color: #ffffff !important;
  font-weight: 800 !important;
  box-shadow: 0 14px 28px rgba(18, 78, 87, 0.20) !important;
}

#shaft-stop-btn button {
  min-height: 52px;
  border-radius: 18px;
  border: 1px solid rgba(158, 53, 87, 0.22) !important;
  background: rgba(255, 244, 247, 0.96) !important;
  color: var(--shaft-danger) !important;
  font-weight: 800 !important;
}

.shaft-pane .gr-button:not(#shaft-start-btn button):not(#shaft-stop-btn button),
.shaft-pane .secondary-wrap button {
  min-height: 50px;
  border-radius: 18px !important;
  background: var(--shaft-control-bg) !important;
  color: var(--shaft-ink) !important;
  border: 1px solid var(--shaft-line-strong) !important;
  box-shadow: none !important;
  font-weight: 700 !important;
}

.shaft-tabs {
  margin-top: 8px;
}

.shaft-tabs .tab-nav {
  gap: 10px;
  padding: 4px 2px 12px;
  border-bottom: 1px solid var(--shaft-line) !important;
}

.shaft-tabs .tab-nav button {
  border-radius: 999px !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  color: var(--shaft-soft-text) !important;
  padding: 8px 12px !important;
  font-weight: 700 !important;
}

.shaft-tabs .tab-nav button.selected {
  background: var(--shaft-accent-soft) !important;
  border-color: rgba(21, 112, 109, 0.18) !important;
  color: var(--shaft-accent-strong) !important;
}

.shaft-tabs .tabitem {
  border-radius: 24px;
  border: 1px solid var(--shaft-line);
  background: rgba(255, 255, 255, 0.18);
  padding: 18px;
}

.shaft-output-tabs .tabitem {
  min-height: 580px;
}

.shaft-output-surface {
  min-height: 520px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.shaft-yaml-surface .shaft-code-control,
.shaft-output-surface .shaft-code-control,
.shaft-output-surface .shaft-preview-control,
.shaft-output-surface .shaft-log-control,
.shaft-output-surface .shaft-runs {
  margin-top: 2px;
}

@media (max-width: 1200px) {
  .shaft-hero {
    grid-template-columns: 1fr;
  }

  .shaft-summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .shaft-status-grid-primary,
  .shaft-status-grid-secondary {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .shaft-shell {
    padding: 14px 12px 4px;
  }

  .shaft-hero {
    padding: 22px 20px 18px;
    border-radius: 24px;
  }

  .shaft-hero-title {
    font-size: 2.3rem;
  }

  .shaft-summary-grid {
    grid-template-columns: 1fr;
  }

  .shaft-status-grid-primary,
  .shaft-status-grid-secondary {
    grid-template-columns: 1fr;
  }

  .shaft-overrides-surface,
  .shaft-yaml-surface,
  .shaft-output-surface {
    padding: 14px;
  }
}
"""


def build_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue="teal",
        secondary_hue="amber",
        neutral_hue="slate",
    ).set(
        body_background_fill="#f2ebde",
        body_text_color="#17212b",
        block_background_fill="rgba(255,251,245,0.96)",
        block_border_color="rgba(57,68,77,0.12)",
        block_radius="24px",
        block_shadow="none",
        background_fill_primary="rgba(255,251,245,0.96)",
        background_fill_secondary="rgba(248,242,232,0.90)",
        input_background_fill="rgba(255,252,247,0.96)",
        input_border_color="rgba(22,33,43,0.16)",
        input_border_color_focus="#15706d",
        input_shadow="none",
        button_primary_background_fill="#15706d",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="rgba(255,252,247,0.96)",
        button_secondary_border_color="rgba(22,33,43,0.16)",
        button_secondary_text_color="#17212b",
        code_background_fill="rgba(246,239,228,0.98)",
        prose_text_size="16px",
    )
