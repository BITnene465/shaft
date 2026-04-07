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


def _resolve_finetune_mode(config_dict: dict[str, Any] | None) -> str:
    if not isinstance(config_dict, dict):
        return "full"
    finetune = config_dict.get("finetune")
    if not isinstance(finetune, dict):
        return "full"
    mode = finetune.get("mode")
    if mode is None:
        return "full"
    return str(mode)


def _resolve_state_dict_path(checkpoint_dir: Path) -> Path:
    flat_path = checkpoint_dir / "state_dict.pt"
    if flat_path.exists():
        return flat_path
    legacy_path = checkpoint_dir / "model" / "state_dict.pt"
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(f"Missing state dict in checkpoint: {checkpoint_dir}")


def _resolve_adapter_dir(checkpoint_dir: Path) -> Path | None:
    root_adapter_config = checkpoint_dir / "adapter_config.json"
    root_adapter_weights = checkpoint_dir / "adapter_model.safetensors"
    root_adapter_weights_bin = checkpoint_dir / "adapter_model.bin"
    if root_adapter_config.exists() and (root_adapter_weights.exists() or root_adapter_weights_bin.exists()):
        return checkpoint_dir

    nested_dir = checkpoint_dir / "adapter"
    nested_adapter_config = nested_dir / "adapter_config.json"
    nested_adapter_weights = nested_dir / "adapter_model.safetensors"
    nested_adapter_weights_bin = nested_dir / "adapter_model.bin"
    if nested_adapter_config.exists() and (nested_adapter_weights.exists() or nested_adapter_weights_bin.exists()):
        return nested_dir
    return None


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
    if hasattr(unwrapped, "config"):
        unwrapped.config.to_json_file(checkpoint_dir / "config.json")

    tokenizer.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)

    checkpoint_mode = _resolve_finetune_mode(config_dict)
    if checkpoint_mode == "lora":
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
        base_model_save_pretrained(base_model_dir, safe_serialization=True)
    else:
        torch.save(unwrapped.state_dict(), checkpoint_dir / "state_dict.pt")

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
            "checkpoint_layout": "peft_adapter" if checkpoint_mode == "lora" else "full_state_dict",
            "has_base_model": bool(checkpoint_mode == "lora"),
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
    meta = load_checkpoint_meta(checkpoint_dir)
    checkpoint_mode = _resolve_finetune_mode(meta.get("config") if isinstance(meta, dict) else None)

    if checkpoint_mode == "lora":
        adapter_dir = _resolve_adapter_dir(checkpoint_dir)
        if adapter_dir is not None:
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
        else:
            state_dict_path = _resolve_state_dict_path(checkpoint_dir)
            state_dict = _torch_load(
                state_dict_path,
                map_location="cpu",
                weights_only=True,
            )
            unwrap_model(model).load_state_dict(state_dict, strict=strict)
    else:
        state_dict_path = _resolve_state_dict_path(checkpoint_dir)
        state_dict = _torch_load(
            state_dict_path,
            map_location="cpu",
            weights_only=True,
        )
        unwrap_model(model).load_state_dict(state_dict, strict=strict)

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
    meta = load_checkpoint_meta(checkpoint_dir)
    checkpoint_mode = (
        meta.get("config", {})
        .get("finetune", {})
        .get("mode")
    )

    target_model = unwrap_model(model)
    base_model_getter = getattr(target_model, "get_base_model", None)
    candidate_models: list[torch.nn.Module] = []

    if checkpoint_mode == "lora" and not callable(base_model_getter):
        raise ValueError(
            "Cannot initialize a non-LoRA model from a LoRA checkpoint with `init-from`. "
            "Use a matching LoRA config with `init-from`, or use `resume-from` on the original training setup."
        )

    if checkpoint_mode == "full" and callable(base_model_getter):
        # Common stage-2 case: initialize a fresh LoRA-wrapped model from a full-FT
        # checkpoint by loading weights into the underlying base model only.
        candidate_models.append(base_model_getter())
    else:
        candidate_models.append(target_model)
        if callable(base_model_getter):
            base_model = base_model_getter()
            if base_model is not target_model:
                candidate_models.append(base_model)

    load_error: RuntimeError | None = None
    for candidate_model in candidate_models:
        try:
            if checkpoint_mode == "lora":
                adapter_dir = _resolve_adapter_dir(checkpoint_dir)
                if adapter_dir is not None:
                    load_adapter = getattr(candidate_model, "load_adapter", None)
                    if callable(load_adapter):
                        load_adapter(
                            adapter_dir,
                            adapter_name="default",
                            is_trainable=True,
                        )
                    else:
                        raise ValueError(
                            "LoRA checkpoint detected, but the target model does not expose `load_adapter()`."
                        )
                else:
                    state_dict_path = _resolve_state_dict_path(checkpoint_dir)
                    state_dict = _torch_load(
                        state_dict_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    candidate_model.load_state_dict(state_dict, strict=strict)
            else:
                state_dict_path = _resolve_state_dict_path(checkpoint_dir)
                state_dict = _torch_load(
                    state_dict_path,
                    map_location="cpu",
                    weights_only=True,
                )
                candidate_state_dict = candidate_model.state_dict()
                remapped_state_dict = _maybe_remap_full_checkpoint_for_lora_base(
                    source_state_dict=state_dict,
                    target_state_dict=candidate_state_dict,
                    checkpoint_mode=checkpoint_mode,
                )
                if remapped_state_dict is not None:
                    merged_state_dict = dict(candidate_state_dict)
                    merged_state_dict.update(remapped_state_dict)
                    candidate_model.load_state_dict(merged_state_dict, strict=True)
                else:
                    candidate_model.load_state_dict(state_dict, strict=strict)
            load_error = None
            break
        except RuntimeError as exc:
            load_error = exc

    if load_error is not None:
        raise RuntimeError(
            f"Failed to initialize model weights from checkpoint {checkpoint_dir}."
        ) from load_error
    return meta


def _maybe_remap_full_checkpoint_for_lora_base(
    source_state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
    checkpoint_mode: str | None,
) -> dict[str, torch.Tensor] | None:
    if checkpoint_mode != "full":
        return None
    if not any(".base_layer." in key for key in target_state_dict):
        return None

    remapped_state_dict: dict[str, torch.Tensor] = {}
    for source_key, value in source_state_dict.items():
        if source_key in target_state_dict:
            remapped_state_dict[source_key] = value
            continue

        base_layer_key = source_key.replace(".weight", ".base_layer.weight")
        base_layer_key = base_layer_key.replace(".bias", ".base_layer.bias")
        if base_layer_key in target_state_dict:
            remapped_state_dict[base_layer_key] = value

    return remapped_state_dict


def load_checkpoint_meta(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
