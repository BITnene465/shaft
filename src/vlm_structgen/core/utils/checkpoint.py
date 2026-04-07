from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from vlm_structgen.core.utils.distributed import get_rng_state, set_rng_state, unwrap_model
from vlm_structgen.core.utils.io import ensure_dir, write_json


def _torch_load(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool,
):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_adapter_dir(checkpoint_dir: Path) -> Path:
    root_adapter_config = checkpoint_dir / "adapter_config.json"
    root_adapter_weights = checkpoint_dir / "adapter_model.safetensors"
    root_adapter_weights_bin = checkpoint_dir / "adapter_model.bin"
    if root_adapter_config.exists() and (root_adapter_weights.exists() or root_adapter_weights_bin.exists()):
        return checkpoint_dir
    raise FileNotFoundError(f"Missing adapter files in checkpoint: {checkpoint_dir}")


def _resolve_base_model_dir(checkpoint_dir: Path) -> Path:
    base_model_dir = checkpoint_dir / "base_model"
    if not (base_model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Missing bundled base model in checkpoint: {checkpoint_dir}."
        )
    return base_model_dir


def _load_base_model_snapshot(checkpoint_dir: Path, model: torch.nn.Module) -> None:
    base_model_dir = _resolve_base_model_dir(checkpoint_dir)
    base_model_getter = getattr(unwrap_model(model), "get_base_model", None)
    if not callable(base_model_getter):
        raise ValueError(
            "LoRA checkpoint detected, but the model does not expose `get_base_model()`."
        )
    base_model = base_model_getter()
    base_model_class = base_model.__class__
    snapshot = base_model_class.from_pretrained(
        base_model_dir,
        local_files_only=True,
    )
    base_model.load_state_dict(snapshot.state_dict(), strict=False)


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    tokenizer,
    processor,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    trainer_state: dict[str, Any],
    config_dict: dict[str, Any],
) -> None:
    checkpoint_dir = ensure_dir(checkpoint_dir)

    unwrapped = unwrap_model(model)
    tokenizer.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)

    save_pretrained = getattr(unwrapped, "save_pretrained", None)
    if not callable(save_pretrained):
        raise ValueError(
            "LoRA checkpoint saving requires a PEFT model with `save_pretrained()`."
        )
    save_pretrained(
        checkpoint_dir,
        safe_serialization=True,
        save_embedding_layers=False,
    )

    base_model_getter = getattr(unwrapped, "get_base_model", None)
    if not callable(base_model_getter):
        raise ValueError(
            "LoRA checkpoint saving requires a PEFT model with `get_base_model()`."
        )
    base_model = base_model_getter()
    base_model_dir = ensure_dir(checkpoint_dir / "base_model")
    base_model_save_pretrained = getattr(base_model, "save_pretrained", None)
    if not callable(base_model_save_pretrained):
        raise ValueError(
            "LoRA checkpoint saving requires a base model with `save_pretrained()`."
        )
    base_model_save_pretrained(base_model_dir, safe_serialization=False)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    torch.save(get_rng_state(), checkpoint_dir / "rng_state.pt")
    write_json(checkpoint_dir / "trainer_state.json", trainer_state)
    write_json(
        checkpoint_dir / "meta.json",
        {
            "experiment_name": config_dict["experiment"]["name"],
            "protocol_version": "arrow_v2_json",
            "config": config_dict,
            "trainer_state": trainer_state,
            "checkpoint_layout": "peft_adapter_with_base_model",
            "has_base_model": True,
        },
    )


def load_training_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    tokenizer=None,
    processor=None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    strict: bool = True,
    resume_training_state: bool = True,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    _ = strict
    _load_base_model_snapshot(checkpoint_dir, model)
    adapter_dir = _resolve_adapter_dir(checkpoint_dir)
    peft_model = unwrap_model(model)
    load_adapter = getattr(peft_model, "load_adapter", None)
    if not callable(load_adapter):
        raise ValueError(
            "LoRA checkpoint detected, but the model does not expose `load_adapter()`."
        )
    load_adapter(
        adapter_dir,
        adapter_name="default",
        is_trainable=resume_training_state,
    )

    trainer_state = {}
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        with trainer_state_path.open("r", encoding="utf-8") as handle:
            trainer_state = json.load(handle)

    if not resume_training_state:
        return trainer_state

    if optimizer is not None and (checkpoint_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(
            _torch_load(
                checkpoint_dir / "optimizer.pt",
                map_location="cpu",
                weights_only=False,
            )
        )
    if scheduler is not None and (checkpoint_dir / "scheduler.pt").exists():
        scheduler.load_state_dict(
            _torch_load(
                checkpoint_dir / "scheduler.pt",
                map_location="cpu",
                weights_only=False,
            )
        )
    if (checkpoint_dir / "rng_state.pt").exists():
        set_rng_state(
            _torch_load(
                checkpoint_dir / "rng_state.pt",
                map_location="cpu",
                weights_only=False,
            )
        )
    return trainer_state


def load_initial_model_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    _load_base_model_snapshot(checkpoint_dir, model)
    adapter_dir = _resolve_adapter_dir(checkpoint_dir)
    peft_model = unwrap_model(model)
    load_adapter = getattr(peft_model, "load_adapter", None)
    if not callable(load_adapter):
        raise ValueError(
            "LoRA checkpoint detected, but the target model does not expose `load_adapter()`."
        )
    load_adapter(
        adapter_dir,
        adapter_name="default",
        is_trainable=True,
    )
    return load_checkpoint_meta(checkpoint_dir)


def load_checkpoint_meta(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
