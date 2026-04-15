from __future__ import annotations

import copy
import json
from pathlib import Path

from peft import PeftModel, load_peft_weights, set_peft_model_state_dict

from shaft.config import RuntimeConfig

from . import qwen3vl as _qwen3vl  # noqa: F401
from . import smoke_vlm as _smoke_vlm  # noqa: F401
from .registry import build_model_meta
from .types import ModelArtifacts


def _is_adapter_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "adapter_config.json").exists():
        return False
    if (path / "adapter_model.safetensors").exists():
        return True
    if (path / "adapter_model.bin").exists():
        return True
    return False


def _load_adapter_config(path: Path) -> dict:
    raw = (path / "adapter_config.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid adapter_config.json in {path}")
    return payload


def _normalize_target_modules(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def _validate_adapter_compatibility(config: RuntimeConfig, adapter_config: dict[str, object], path: Path) -> None:
    mode = str(config.model.finetune.mode).strip().lower()
    if mode == "full":
        raise ValueError(f"init_from={path} is a PEFT adapter checkpoint, but finetune.mode is 'full'.")
    if mode not in {"lora", "dora", "qlora"}:
        raise ValueError(f"Unsupported finetune mode for adapter init: {mode!r}")

    use_dora = bool(adapter_config.get("use_dora", False))
    if mode == "dora" and not use_dora:
        raise ValueError(f"Adapter at {path} is LoRA, but finetune.mode='dora'.")
    if mode != "dora" and use_dora:
        raise ValueError(f"Adapter at {path} is DoRA, but finetune.mode={mode!r}.")

    expected_r = int(config.model.finetune.lora_r)
    adapter_r = int(adapter_config.get("r", expected_r))
    if adapter_r != expected_r:
        raise ValueError(f"LoRA rank mismatch: adapter={adapter_r}, config={expected_r}.")

    expected_alpha = int(config.model.finetune.lora_alpha)
    adapter_alpha = int(adapter_config.get("lora_alpha", expected_alpha))
    if adapter_alpha != expected_alpha:
        raise ValueError(f"LoRA alpha mismatch: adapter={adapter_alpha}, config={expected_alpha}.")

    expected_bias = str(config.model.finetune.lora_bias).strip().lower()
    adapter_bias = str(adapter_config.get("bias", expected_bias)).strip().lower()
    if adapter_bias != expected_bias:
        raise ValueError(f"LoRA bias mismatch: adapter={adapter_bias!r}, config={expected_bias!r}.")

    expected_target_modules = _normalize_target_modules(config.model.finetune.target_modules)
    adapter_target_modules = _normalize_target_modules(adapter_config.get("target_modules"))
    if expected_target_modules not in (["all-linear"], ["auto"]) and adapter_target_modules:
        if sorted(expected_target_modules) != sorted(adapter_target_modules):
            raise ValueError("LoRA target_modules mismatch between adapter and current config.")

    expected_rslora = bool(config.model.finetune.use_rslora)
    adapter_rslora = bool(adapter_config.get("use_rslora", expected_rslora))
    if adapter_rslora != expected_rslora:
        raise ValueError(
            f"LoRA use_rslora mismatch: adapter={adapter_rslora}, config={expected_rslora}."
        )


def _build_artifacts_from_runtime_config(config: RuntimeConfig, *, model_meta) -> ModelArtifacts:
    runtime_config = copy.deepcopy(config)
    model_adapter = model_meta.resolve_adapter(
        model_name_or_path=runtime_config.model.model_name_or_path,
        template_type=runtime_config.model.template,
    )
    model_adapter.check_requires()
    assert model_meta.loader is not None
    return model_meta.loader.build(runtime_config, model_meta=model_meta, model_adapter=model_adapter)


def build_model_tokenizer_processor(
    config: RuntimeConfig,
    *,
    init_from_checkpoint: str | None = None,
) -> ModelArtifacts:
    model_type = str(config.model.model_type).strip().lower()
    model_meta = build_model_meta(model_type)
    if init_from_checkpoint is None:
        return _build_artifacts_from_runtime_config(config, model_meta=model_meta)

    init_path = Path(init_from_checkpoint)
    if not init_path.exists():
        raise FileNotFoundError(f"init_from checkpoint path not found: {init_path}")

    if _is_adapter_checkpoint(init_path):
        adapter_cfg = _load_adapter_config(init_path)
        _validate_adapter_compatibility(config, adapter_cfg, init_path)
        artifacts = _build_artifacts_from_runtime_config(config, model_meta=model_meta)
        if not isinstance(artifacts.model, PeftModel):
            raise TypeError("Adapter init requires a PEFT model, but current mode did not create one.")
        peft_state = load_peft_weights(str(init_path), device="cpu")
        set_peft_model_state_dict(artifacts.model, peft_state, adapter_name="default")
        return artifacts

    override_config = copy.deepcopy(config)
    override_config.model.model_name_or_path = str(init_path)
    return _build_artifacts_from_runtime_config(override_config, model_meta=model_meta)
