from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vlm_structgen.core.config import ExperimentRuntimeConfig, _from_dict, apply_model_scale_tag, load_config
from vlm_structgen.core.modeling.builder import (
    build_model_tokenizer_processor,
    build_model_tokenizer_processor_from_checkpoint,
)
from vlm_structgen.core.utils.checkpoint import load_checkpoint_meta, load_training_checkpoint
from vlm_structgen.core.utils.distributed import unwrap_model
from vlm_structgen.core.utils.io import ensure_dir, write_json


@dataclass
class MergeResult:
    output_dir: Path
    checkpoint_dir: Path
    used_checkpoint_meta_config: bool
    model_source: str
    merged_state_dict_pt: Path | None
    merged_full_model_pt: Path | None


def _resolve_runtime_config(
    *,
    checkpoint_dir: Path,
    config_path: str | Path | None,
    prefer_checkpoint_meta: bool,
) -> tuple[ExperimentRuntimeConfig, bool]:
    if prefer_checkpoint_meta:
        meta = load_checkpoint_meta(checkpoint_dir)
        payload = meta.get("config") if isinstance(meta, dict) else None
        if isinstance(payload, dict) and payload:
            return apply_model_scale_tag(_from_dict(ExperimentRuntimeConfig, payload)), True

    if config_path is None:
        raise ValueError(
            "No runtime config available. Pass --config, or keep --prefer-checkpoint-meta enabled "
            "when checkpoint meta includes config."
        )
    return load_config(config_path), False


def _resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sanitize_tokenizer_config(path: Path) -> None:
    tokenizer_config_path = path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return
    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
    except json.JSONDecodeError:
        return
    if not isinstance(tokenizer_config.get("extra_special_tokens"), list):
        return
    tokenizer_config.pop("extra_special_tokens", None)
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n")


def merge_lora_checkpoint(
    *,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    prefer_checkpoint_meta: bool = True,
    device_name: str | None = None,
    safe_serialization: bool = True,
    export_state_dict_pt: bool = False,
    export_full_model_pt: bool = False,
    save_checkpoint_compat: bool = False,
) -> MergeResult:
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)

    runtime_config, used_meta = _resolve_runtime_config(
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
        prefer_checkpoint_meta=prefer_checkpoint_meta,
    )

    # NOTE: Merge is inference/export only. Disable training-time runtime hooks
    # to keep serialization predictable (especially full-model torch.save).
    runtime_config.train.gradient_checkpointing = False

    if runtime_config.finetune.mode != "lora":
        raise ValueError(
            f"Expected finetune.mode='lora' for merge, got {runtime_config.finetune.mode!r}."
        )

    if (checkpoint_dir / "config.json").exists() or (checkpoint_dir / "model" / "config.json").exists():
        artifacts = build_model_tokenizer_processor_from_checkpoint(
            runtime_config,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        artifacts = build_model_tokenizer_processor(runtime_config)
    device = _resolve_device(device_name)
    artifacts.model = artifacts.model.to(device)

    load_training_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        processor=artifacts.processor,
        strict=True,
        resume_training_state=False,
    )

    peft_or_model = unwrap_model(artifacts.model)
    merge_and_unload = getattr(peft_or_model, "merge_and_unload", None)
    if not callable(merge_and_unload):
        raise ValueError(
            "Loaded model does not expose merge_and_unload(). "
            "Please verify checkpoint/config correspond to a LoRA training run."
        )

    merged_model = merge_and_unload()
    merged_model = merged_model.to("cpu")
    merged_model.eval()

    ensure_dir(output_dir)
    merged_model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    artifacts.tokenizer.save_pretrained(output_dir)
    artifacts.processor.save_pretrained(output_dir)
    _sanitize_tokenizer_config(output_dir)

    merged_state_dict_pt: Path | None = None
    merged_full_model_pt: Path | None = None

    if export_state_dict_pt:
        merged_state_dict_pt = output_dir / "merged_state_dict.pt"
        torch.save(merged_model.state_dict(), merged_state_dict_pt)

    full_model_export_error: str | None = None
    if export_full_model_pt:
        merged_full_model_pt = output_dir / "merged_model_full.pt"
        try:
            torch.save(merged_model, merged_full_model_pt)
        except Exception as exc:  # noqa: BLE001
            full_model_export_error = str(exc)
            merged_full_model_pt = None

    if save_checkpoint_compat:
        model_dir = ensure_dir(output_dir / "model")
        tokenizer_dir = ensure_dir(output_dir / "tokenizer")
        processor_dir = ensure_dir(output_dir / "processor")
        torch.save(merged_model.state_dict(), model_dir / "state_dict.pt")
        if hasattr(merged_model, "config"):
            merged_model.config.to_json_file(model_dir / "config.json")
        artifacts.tokenizer.save_pretrained(tokenizer_dir)
        artifacts.processor.save_pretrained(processor_dir)
        _sanitize_tokenizer_config(tokenizer_dir)
        _sanitize_tokenizer_config(processor_dir)

    write_json(
        output_dir / "merge_meta.json",
        {
            "source_checkpoint": str(checkpoint_dir),
            "used_checkpoint_meta_config": bool(used_meta),
            "model_source": runtime_config.model.model_name_or_path,
            "finetune_mode": runtime_config.finetune.mode,
            "safe_serialization": bool(safe_serialization),
            "export_state_dict_pt": bool(export_state_dict_pt),
            "export_full_model_pt": bool(export_full_model_pt),
            "save_checkpoint_compat": bool(save_checkpoint_compat),
            "merged_state_dict_pt": str(merged_state_dict_pt) if merged_state_dict_pt is not None else None,
            "merged_full_model_pt": str(merged_full_model_pt) if merged_full_model_pt is not None else None,
            "full_model_export_error": full_model_export_error,
        },
    )

    return MergeResult(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        used_checkpoint_meta_config=used_meta,
        model_source=runtime_config.model.model_name_or_path,
        merged_state_dict_pt=merged_state_dict_pt,
        merged_full_model_pt=merged_full_model_pt,
    )
