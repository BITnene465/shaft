from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch

from vlm_structgen.core.registry import get_adapter_for_route, resolve_route_binding
from vlm_structgen.core.routing import normalize_route_key, parse_route_key
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


def _has_adapter_assets(checkpoint_dir: Path) -> bool:
    root_adapter_config = checkpoint_dir / "adapter_config.json"
    root_adapter_weights = checkpoint_dir / "adapter_model.safetensors"
    root_adapter_weights_bin = checkpoint_dir / "adapter_model.bin"
    return root_adapter_config.exists() and (root_adapter_weights.exists() or root_adapter_weights_bin.exists())


def _has_full_model_assets(checkpoint_dir: Path) -> bool:
    config_file = checkpoint_dir / "config.json"
    if not config_file.exists():
        return False
    full_weight_candidates = (
        checkpoint_dir / "model.safetensors",
        checkpoint_dir / "model.safetensors.index.json",
        checkpoint_dir / "pytorch_model.bin",
        checkpoint_dir / "pytorch_model.bin.index.json",
        checkpoint_dir / "weights.json",  # lightweight dummy model tests
    )
    return any(path.exists() for path in full_weight_candidates)


def _resolve_checkpoint_layout(checkpoint_dir: Path) -> Literal["adapter", "full_model"]:
    if _has_adapter_assets(checkpoint_dir):
        return "adapter"
    if _has_full_model_assets(checkpoint_dir):
        return "full_model"
    raise FileNotFoundError(
        "Unsupported checkpoint layout. Expected PEFT adapter files or full model files. "
        f"checkpoint={checkpoint_dir}"
    )


def _resolve_dense_target_model(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = unwrap_model(model)
    # Full-FT checkpoints should load against the top-level transformers model
    # (e.g. Qwen3VLForConditionalGeneration). Do not descend to `.base_model`
    # unless this is explicitly a PEFT wrapper.
    if not hasattr(unwrapped, "peft_config"):
        return unwrapped
    get_base_model = getattr(unwrapped, "get_base_model", None)
    if callable(get_base_model):
        try:
            base_model = get_base_model()
            if isinstance(base_model, torch.nn.Module):
                return base_model
        except Exception:  # noqa: BLE001
            pass
    base_model_attr = getattr(unwrapped, "base_model", None)
    if isinstance(base_model_attr, torch.nn.Module):
        return base_model_attr
    return unwrapped


def _load_full_model_weights(
    *,
    checkpoint_dir: Path,
    model: torch.nn.Module,
    strict: bool,
) -> None:
    target_model = _resolve_dense_target_model(model)
    from_pretrained = getattr(type(target_model), "from_pretrained", None)
    if not callable(from_pretrained):
        raise ValueError(
            "Dense checkpoint detected, but target model class does not expose from_pretrained(). "
            f"target_model_class={type(target_model).__name__!r}."
        )

    try:
        loaded_model = type(target_model).from_pretrained(checkpoint_dir, local_files_only=True)
    except TypeError:
        loaded_model = type(target_model).from_pretrained(checkpoint_dir)

    load_result = target_model.load_state_dict(loaded_model.state_dict(), strict=strict)
    if not strict and hasattr(load_result, "unexpected_keys") and hasattr(load_result, "missing_keys"):
        # Keep best-effort behavior for non-strict load while still surfacing mismatch context.
        _ = load_result.unexpected_keys, load_result.missing_keys
    del loaded_model


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
        raise ValueError("Checkpoint saving requires a model with `save_pretrained()`.")
    save_pretrained(
        checkpoint_dir,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    layout = _resolve_checkpoint_layout(checkpoint_dir)
    checkpoint_layout = "peft_adapter_only" if layout == "adapter" else "full_model"
    has_base_model = layout == "full_model"

    if optimizer is not None:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    torch.save(get_rng_state(), checkpoint_dir / "rng_state.pt")
    write_json(checkpoint_dir / "trainer_state.json", trainer_state)
    write_json(
        checkpoint_dir / "protocol.json",
        _build_protocol_artifact(config_dict),
    )
    write_json(
        checkpoint_dir / "meta.json",
        {
            "experiment_name": config_dict["experiment"]["name"],
            "protocol_version": "vlm_structgen_v1",
            "config": config_dict,
            "trainer_state": trainer_state,
            "checkpoint_layout": checkpoint_layout,
            "has_base_model": has_base_model,
        },
    )


def _build_protocol_artifact(config_dict: dict[str, Any]) -> dict[str, Any]:
    model_cfg = dict(config_dict.get("model", {}))
    finetune_cfg = dict(config_dict.get("finetune", {}))
    tokenizer_cfg = dict(config_dict.get("tokenizer", {}))
    task_cfg = dict(config_dict.get("task", {}))
    prompt_cfg = dict(config_dict.get("prompt", {}))
    eval_cfg = dict(config_dict.get("eval", {}))

    route_options = _normalize_route_mapping(task_cfg.get("route_options"))
    route_prompts = _normalize_route_mapping(prompt_cfg.get("route_prompts"))
    routes = sorted(_collect_routes(task_cfg=task_cfg, route_options=route_options, route_prompts=route_prompts))

    return {
        "protocol_version": "1.0.0",
        "model_family": "qwen3_vl",
        "model_name_or_path": model_cfg.get("model_name_or_path"),
        "finetune_mode": str(finetune_cfg.get("mode", "")),
        "tokenizer": {
            "num_bins": tokenizer_cfg.get("num_bins"),
            "add_eos_token": tokenizer_cfg.get("add_eos_token"),
        },
        "routes": [
            _build_route_protocol_entry(
                route_key=route_key,
                route_options=route_options.get(route_key, {}),
                route_prompt=route_prompts.get(route_key, {}),
                num_bins=tokenizer_cfg.get("num_bins"),
            )
            for route_key in routes
        ],
        "global_evaluation": {
            "best_metric": eval_cfg.get("best_metric"),
            "monitor_mode": eval_cfg.get("monitor_mode"),
        },
        "compatibility": {
            "legacy_task_domain_fallback": True,
        },
    }


def _normalize_route_mapping(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for raw_route, raw_value in payload.items():
        if not isinstance(raw_value, dict):
            continue
        route_key = normalize_route_key(str(raw_route))
        normalized[route_key] = dict(raw_value)
    return normalized


def _collect_routes(
    *,
    task_cfg: dict[str, Any],
    route_options: dict[str, dict[str, Any]],
    route_prompts: dict[str, dict[str, Any]],
) -> set[str]:
    routes: set[str] = set(route_options.keys()) | set(route_prompts.keys())
    default_route = task_cfg.get("route")
    if default_route is not None and str(default_route).strip():
        routes.add(normalize_route_key(str(default_route)))
    return routes


def _build_route_protocol_entry(
    *,
    route_key: str,
    route_options: dict[str, Any],
    route_prompt: dict[str, Any],
    num_bins: Any,
) -> dict[str, Any]:
    task_name: str | None = None
    domain_name: str | None = None
    adapter_name: str | None = None
    codec_name: str | None = None
    default_metric_name: str | None = None

    try:
        binding = resolve_route_binding(route_key)
        task_name = binding.task_type
        domain_name = binding.domain_type
    except Exception:
        try:
            task_name, domain_name = parse_route_key(route_key)
        except Exception:
            task_name, domain_name = None, None

    try:
        if num_bins is not None:
            adapter = get_adapter_for_route(
                route_key=route_key,
                num_bins=int(num_bins),
                task_options_key=(),
            )
            adapter_name = type(adapter).__name__
            codec = getattr(adapter, "codec", None)
            codec_name = type(codec).__name__ if codec is not None else None
            if hasattr(adapter, "default_eval_primary_metric"):
                default_metric_name = str(adapter.default_eval_primary_metric())
    except Exception:
        pass

    primary_metric = route_options.get("eval_primary_metric")
    if primary_metric is None or str(primary_metric).strip() == "":
        primary_metric = default_metric_name

    return {
        "route": route_key,
        "task_name": task_name,
        "domain_name": domain_name,
        "adapter": {
            "name": adapter_name,
            "codec": codec_name,
            "codec_version": None,
        },
        "prompt": {
            "profile": route_prompt.get("profile"),
        },
        "evaluation": {
            "primary_metric": primary_metric,
            "normalizer": route_options.get("eval_metric_normalizer", "identity"),
            "weight": float(route_options.get("eval_metric_weight", 1.0)),
            "metric_min": route_options.get("eval_metric_min"),
            "metric_max": route_options.get("eval_metric_max"),
        },
        "route_options": dict(route_options),
    }


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
    layout = _resolve_checkpoint_layout(checkpoint_dir)
    if layout == "adapter":
        peft_model = unwrap_model(model)
        load_adapter = getattr(peft_model, "load_adapter", None)
        if not callable(load_adapter):
            raise ValueError(
                "LoRA checkpoint detected, but the model does not expose `load_adapter()`."
            )
        load_adapter(
            checkpoint_dir,
            adapter_name="default",
            is_trainable=resume_training_state,
        )
    else:
        _load_full_model_weights(
            checkpoint_dir=checkpoint_dir,
            model=model,
            strict=strict,
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
    layout = _resolve_checkpoint_layout(checkpoint_dir)
    if layout == "adapter":
        peft_model = unwrap_model(model)
        load_adapter = getattr(peft_model, "load_adapter", None)
        if not callable(load_adapter):
            raise ValueError(
                "LoRA checkpoint detected, but the target model does not expose `load_adapter()`."
            )
        load_adapter(
            checkpoint_dir,
            adapter_name="default",
            is_trainable=True,
        )
    else:
        _load_full_model_weights(
            checkpoint_dir=checkpoint_dir,
            model=model,
            strict=strict,
        )
    return load_checkpoint_meta(checkpoint_dir)


def load_checkpoint_meta(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
