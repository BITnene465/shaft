from __future__ import annotations

import copy
import json
from pathlib import Path

from peft import PeftModel, load_peft_weights, set_peft_model_state_dict

from shaft.config import RuntimeConfig

from . import qwen35vl as _qwen35vl  # noqa: F401
from . import qwen3vl as _qwen3vl  # noqa: F401
from . import smoke_vlm as _smoke_vlm  # noqa: F401
from .registry import build_model_meta
from .types import ModelArtifacts, ShaftModelAdapter


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


def _validate_hf_sharded_checkpoint_files(path: Path) -> None:
    if not path.is_dir():
        return

    index_path: Path | None = None
    for candidate in (
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    ):
        if candidate.exists():
            index_path = candidate
            break
    if index_path is None:
        return

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid HF checkpoint index JSON: {index_path}") from exc

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"HF checkpoint index has no object weight_map: {index_path}")

    expected_files = sorted(
        {
            str(value).strip()
            for value in weight_map.values()
            if isinstance(value, str) and str(value).strip()
        }
    )
    missing = [name for name in expected_files if not (path / name).is_file()]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... (+{len(missing) - 8} more)"
        temp_dir = path / "._____temp"
        temp_hint = ""
        if temp_dir.exists():
            temp_matches = [name for name in missing if (temp_dir / name).exists()]
            if temp_matches:
                temp_hint = (
                    f" {len(temp_matches)} missing shard(s) are still in {temp_dir}; "
                    "the model download is likely incomplete."
                )
        raise FileNotFoundError(
            f"HF sharded checkpoint is incomplete under {path}: missing {len(missing)} "
            f"file(s): {preview}{suffix}.{temp_hint}"
        )


def _validate_local_hf_config_model_type(path: Path, *, model_meta) -> None:
    expected = tuple(str(item).strip() for item in getattr(model_meta, "hf_model_types", ()) if str(item).strip())
    if not expected or not path.is_dir():
        return
    config_path = path / "config.json"
    if not config_path.exists():
        return
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid HF config JSON: {config_path}") from exc
    actual = str(payload.get("model_type") or "").strip()
    if actual and actual not in expected:
        raise ValueError(
            f"HF config model_type={actual!r} under {path} does not match "
            f"model.model_type={model_meta.model_type!r}; expected one of {expected}."
        )


def _load_adapter_config(path: Path) -> dict:
    raw = (path / "adapter_config.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid adapter_config.json in {path}")
    return payload


def _normalize_name_list(value) -> list[str]:
    if isinstance(value, str):
        return [str(value)]
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    return []


def _expected_adapter_names_from_artifacts(artifacts: ModelArtifacts) -> tuple[list[str] | None, list[str] | None]:
    finetune_plan = getattr(artifacts, "finetune_plan", None)
    if finetune_plan is None or getattr(finetune_plan, "adapter_plan", None) is None:
        return None, None
    adapter_plan = finetune_plan.adapter_plan
    return (
        list(getattr(adapter_plan, "resolved_target_modules", ()) or ()),
        list(getattr(adapter_plan, "modules_to_save", ()) or ()),
    )


def _resolve_default_peft_config(model: PeftModel):
    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict):
        if "default" in peft_config:
            return peft_config["default"]
        if peft_config:
            return next(iter(peft_config.values()))
        return None
    return peft_config


def _validate_adapter_compatibility(
    config: RuntimeConfig,
    adapter_config: dict[str, object],
    path: Path,
    *,
    expected_target_modules: list[str] | None = None,
    expected_modules_to_save: list[str] | None = None,
) -> None:
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

    adapter_target_modules = _normalize_name_list(adapter_config.get("target_modules"))
    if expected_target_modules is not None and adapter_target_modules:
        if sorted(expected_target_modules) != sorted(adapter_target_modules):
            raise ValueError("LoRA target_modules mismatch between adapter and current config.")

    if expected_modules_to_save is not None:
        adapter_modules_to_save = _normalize_name_list(adapter_config.get("modules_to_save"))
        if sorted(expected_modules_to_save) != sorted(adapter_modules_to_save):
            raise ValueError("LoRA modules_to_save mismatch between adapter and current config.")

    expected_rslora = bool(config.model.finetune.use_rslora)
    adapter_rslora = bool(adapter_config.get("use_rslora", expected_rslora))
    if adapter_rslora != expected_rslora:
        raise ValueError(
            f"LoRA use_rslora mismatch: adapter={adapter_rslora}, config={expected_rslora}."
        )


def _build_artifacts_from_runtime_config(config: RuntimeConfig, *, model_meta) -> ModelArtifacts:
    runtime_config = copy.deepcopy(config)
    model_path = Path(runtime_config.model.model_name_or_path)
    _validate_hf_sharded_checkpoint_files(model_path)
    _validate_local_hf_config_model_type(model_path, model_meta=model_meta)
    model_adapter = resolve_model_adapter_from_config(
        runtime_config,
        model_meta=model_meta,
    )
    model_adapter.check_requires()
    assert model_meta.loader is not None
    return model_meta.loader.build(runtime_config, model_meta=model_meta, model_adapter=model_adapter)


def resolve_model_adapter_from_config(
    config: RuntimeConfig,
    *,
    model_meta=None,
) -> ShaftModelAdapter:
    resolved_model_meta = model_meta or build_model_meta(
        str(config.model.model_type).strip().lower()
    )
    return resolved_model_meta.resolve_adapter(
        model_name_or_path=config.model.model_name_or_path,
        template_type=config.model.template,
    )


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
        expected_target_modules, expected_modules_to_save = _expected_adapter_names_from_artifacts(artifacts)
        if expected_target_modules is None:
            peft_config = _resolve_default_peft_config(artifacts.model)
            expected_target_modules = _normalize_name_list(
                getattr(peft_config, "target_modules", config.model.finetune.target_modules)
            )
            expected_modules_to_save = _normalize_name_list(getattr(peft_config, "modules_to_save", None))
        _validate_adapter_compatibility(
            config,
            adapter_cfg,
            init_path,
            expected_target_modules=expected_target_modules,
            expected_modules_to_save=expected_modules_to_save,
        )
        peft_state = load_peft_weights(str(init_path), device="cpu")
        set_peft_model_state_dict(artifacts.model, peft_state, adapter_name="default")
        return artifacts

    override_config = copy.deepcopy(config)
    override_config.model.model_name_or_path = str(init_path)
    return _build_artifacts_from_runtime_config(override_config, model_meta=model_meta)
