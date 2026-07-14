from __future__ import annotations

from pathlib import Path

import pytest

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


def test_load_infer_config_accepts_shared_prompt_arguments(tmp_path: Path) -> None:
    payload = """
engines:
  e1:
    model_type: smoke_vlm
    model_name_or_path: unused
stages:
  - name: stage1
    engine: e1
    arguments:
      payload:
        type: json
    user_prompt_template: "payload={{ payload | json }}"
"""
    cfg_path = tmp_path / "infer_dynamic.yaml"
    cfg_path.write_text(payload, encoding="utf-8")

    cfg = load_infer_config(cfg_path)

    assert cfg.stages[0].arguments == {"payload": {"type": "json"}}


@pytest.mark.parametrize(
    "legacy",
    [
        "{det_out}",
        "{det_out!r}",
        "{det_out:s}",
        "{det_out.value}",
        "{det_out[key]}",
        "{det_out:{width}}",
    ],
)
def test_load_infer_config_rejects_legacy_python_format_placeholder(
    tmp_path: Path,
    legacy: str,
) -> None:
    payload = f"""
engines:
  e1:
    model_type: smoke_vlm
    model_name_or_path: unused
stages:
  - name: stage1
    engine: e1
    user_prompt_template: "legacy {legacy}"
"""
    cfg_path = tmp_path / "infer_legacy.yaml"
    cfg_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="legacy.*det_out.*double braces"):
        load_infer_config(cfg_path)


def test_load_infer_config_rejects_invalid_unicode_in_system_prompt(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "infer_invalid_unicode.yaml"
    cfg_path.write_text(
        """
engines:
  mock:
    backend: hf_local
stages:
  - name: invalid
    engine: mock
    system_prompt: "\\uD800"
    user_prompt_template: static
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="UTF-8"):
        load_infer_config(cfg_path)


def test_load_infer_config_rejects_unused_prompt_arguments(tmp_path: Path) -> None:
    payload = """
engines:
  e1:
    model_type: smoke_vlm
    model_name_or_path: unused
stages:
  - name: stage1
    engine: e1
    arguments:
      unused: {type: string}
    user_prompt_template: static
"""
    cfg_path = tmp_path / "infer_unused.yaml"
    cfg_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="unused fields.*unused"):
        load_infer_config(cfg_path)


def test_load_infer_config_rejects_unknown_codec(tmp_path: Path) -> None:
    payload = """
engines:
  e1:
    model_type: smoke_vlm
    model_name_or_path: unused
    device: cpu
stages:
  - name: stage1
    engine: e1
    codec: not_exists
    user_prompt_template: "hello"
"""
    cfg_path = tmp_path / "infer_bad_codec.yaml"
    cfg_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered"):
        load_infer_config(cfg_path)


def test_load_infer_config_rejects_unknown_stage_engine(tmp_path: Path) -> None:
    payload = """
engines:
  e1:
    model_type: smoke_vlm
    model_name_or_path: unused
    device: cpu
stages:
  - name: stage1
    engine: missing
    user_prompt_template: "hello"
"""
    cfg_path = tmp_path / "infer_bad_engine.yaml"
    cfg_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="not found in engines"):
        load_infer_config(cfg_path)


def test_load_infer_config_rejects_vllm_backend_without_endpoint(tmp_path: Path) -> None:
    payload = """
engines:
  e1:
    backend: vllm_openai
    model_type: qwen3vl
    model_name_or_path: arrow_mixed_4b
stages:
  - name: stage1
    engine: e1
    user_prompt_template: "hello"
"""
    cfg_path = tmp_path / "infer_bad_vllm_endpoint.yaml"
    cfg_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="endpoint is required"):
        load_infer_config(cfg_path)


def test_load_infer_config_accepts_vllm_backend(tmp_path: Path) -> None:
    payload = """
engines:
  e1:
    backend: vllm_openai
    endpoint: http://127.0.0.1:8001
    model_type: qwen3vl
    served_model_name: arrow_mixed_4b
    request_timeout_seconds: 30
stages:
  - name: stage1
    engine: e1
    user_prompt_template: "hello"
"""
    cfg_path = tmp_path / "infer_vllm_ok.yaml"
    cfg_path.write_text(payload, encoding="utf-8")
    cfg = load_infer_config(cfg_path)
    assert cfg.engines["e1"].backend == "vllm_openai"
    assert cfg.engines["e1"].endpoint == "http://127.0.0.1:8001"


def test_load_qwen36_vllm_example_config() -> None:
    cfg = load_infer_config(Path("configs/infer/qwen36_vllm_example.yaml"))
    engine = cfg.engines["qwen36_vllm"]
    assert engine.backend == "vllm_openai"
    assert engine.model_type == "qwen36vl"
    assert engine.model_name_or_path == "models/Qwen3.6-27B"
    assert engine.served_model_name == "qwen3_6_27b"
    assert cfg.stages[0].max_pixels == 1_000_000


def test_load_infer_config_rejects_invalid_stage_pixel_budget(tmp_path: Path) -> None:
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
    min_pixels: 200
    max_pixels: 100
"""
    cfg_path = tmp_path / "infer_bad_stage_pixels.yaml"
    cfg_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="min_pixels must be <= max_pixels"):
        load_infer_config(cfg_path)
