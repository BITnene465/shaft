from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def _resolve_adapter_dir(checkpoint_dir: Path) -> Path:
    root_adapter_config = checkpoint_dir / "adapter_config.json"
    root_adapter_weights = checkpoint_dir / "adapter_model.safetensors"
    root_adapter_weights_bin = checkpoint_dir / "adapter_model.bin"
    if root_adapter_config.exists() and (root_adapter_weights.exists() or root_adapter_weights_bin.exists()):
        return checkpoint_dir
    raise FileNotFoundError(f"Missing adapter files in checkpoint: {checkpoint_dir}")


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
            "checkpoint_layout": "peft_adapter_only",
            "has_base_model": False,
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
    _ = strict
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
