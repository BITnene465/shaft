from __future__ import annotations

from pathlib import Path

import gradio as gr

from shaft.webui import create_app
from shaft.webui.theme import THEME_INIT_JS, WEBUI_CSS


def test_create_webui_app_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
algorithm:
  name: sft
data:
  datasets:
    - dataset_name: ds
      train_path: train.jsonl
      val_path: val.jsonl
train:
  report_to: ["none"]
eval:
  enabled: true
""",
        encoding="utf-8",
    )
    app = create_app(default_config_path=str(config_path))
    assert isinstance(app, gr.Blocks)


def test_webui_theme_is_local_and_toggleable() -> None:
    assert "googleapis" not in WEBUI_CSS
    assert "__shaftToggleTheme" in THEME_INIT_JS
    assert "localStorage" in THEME_INIT_JS
