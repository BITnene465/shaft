from __future__ import annotations

from pathlib import Path

from shaft.infer import load_infer_config


def test_load_minimal_infer_config(tmp_path: Path) -> None:
    payload = """
engines:
  e1:
    model_type: smoke_vlm
    model_name_or_path: unused
    device: cpu
stages:
  - name: stage1
    engine: e1
    user_prompt_template: "hello"
"""
    cfg_path = tmp_path / "infer.yaml"
    cfg_path.write_text(payload, encoding="utf-8")
    cfg = load_infer_config(cfg_path)
    assert "e1" in cfg.engines
    assert len(cfg.stages) == 1
    assert cfg.stages[0].name == "stage1"
