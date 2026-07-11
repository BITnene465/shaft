from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from transformers import TrainingArguments

from shaft.config import RuntimeConfig, resolve_effective_gradient_checkpointing
from shaft.data import ShaftDynamicBatchPlanningContract


def _resolve_fsdp_transformer_layers(config: RuntimeConfig) -> list[str]:
    configured = list(config.train.distributed.fsdp.transformer_layer_cls_to_wrap)
    from shaft.model import build_model_meta

    model_type = str(config.model.model_type).strip().lower()
    try:
        model_meta = build_model_meta(model_type)
    except KeyError as exc:
        raise ValueError(
            "train.distributed.fsdp.transformer_layer_cls_to_wrap=['auto'] is not available "
            f"for model.model_type={model_type!r}. Configure explicit transformer layer class names."
        ) from exc
    model_adapter = model_meta.resolve_adapter(
        model_name_or_path=config.model.model_name_or_path,
        template_type=config.model.template,
    )
    return model_adapter.resolve_fsdp_transformer_layer_cls_to_wrap(configured)


def _resolve_fsdp_auto_wrap_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized == "transformer":
        return "TRANSFORMER_BASED_WRAP"
    if normalized == "size":
        return "SIZE_BASED_WRAP"
    if normalized in {"none", "no_wrap"}:
        return "NO_WRAP"
    return str(policy)


def _resolve_fsdp_reshard_after_forward(sharding_strategy: str) -> bool | str:
    normalized = str(sharding_strategy).strip().lower()
    if normalized == "full_shard":
        return True
    if normalized in {"no_shard", "none"}:
        return False
    return normalized


def _build_fsdp_args(config: RuntimeConfig) -> tuple[bool | None, dict[str, Any] | None]:
    distributed = config.train.distributed
    if distributed.strategy != "fsdp":
        return None, None

    fsdp_cfg = distributed.fsdp
    fsdp_config: dict[str, Any] = {
        "version": 2,
        "activation_checkpointing": bool(fsdp_cfg.activation_checkpointing),
        "cpu_offload": bool(fsdp_cfg.cpu_offload),
        "use_orig_params": bool(fsdp_cfg.use_orig_params),
        "forward_prefetch": bool(fsdp_cfg.forward_prefetch),
        "limit_all_gathers": bool(fsdp_cfg.limit_all_gathers),
        "state_dict_type": str(fsdp_cfg.state_dict_type),
        "sync_module_states": bool(fsdp_cfg.sync_module_states),
        "reshard_after_forward": _resolve_fsdp_reshard_after_forward(fsdp_cfg.sharding_strategy),
        "auto_wrap_policy": _resolve_fsdp_auto_wrap_policy(fsdp_cfg.auto_wrap_policy),
    }
    if fsdp_cfg.backward_prefetch is not None:
        fsdp_config["backward_prefetch"] = str(fsdp_cfg.backward_prefetch)

    if fsdp_cfg.auto_wrap_policy == "transformer":
        fsdp_config["transformer_layer_cls_to_wrap"] = _resolve_fsdp_transformer_layers(config)
    elif fsdp_cfg.auto_wrap_policy == "size":
        fsdp_config["min_num_params"] = int(fsdp_cfg.min_num_params)

    return True, fsdp_config


def _build_deepspeed_arg(config: RuntimeConfig) -> dict[str, Any] | str | None:
    distributed = config.train.distributed
    if distributed.strategy != "deepspeed":
        return None
    deepspeed_cfg = distributed.deepspeed
    if deepspeed_cfg.config:
        return dict(deepspeed_cfg.config)
    if deepspeed_cfg.config_path:
        return str(deepspeed_cfg.config_path)
    raise ValueError(
        "train.distributed.strategy='deepspeed' requires either "
        "train.distributed.deepspeed.config_path or train.distributed.deepspeed.config."
    )


def _reset_deepspeed_runtime_state() -> None:
    """Clear HF/Accelerate DeepSpeed globals when Shaft is not using DeepSpeed."""
    os.environ.pop("ACCELERATE_USE_DEEPSPEED", None)
    try:
        from transformers.integrations.deepspeed import unset_hf_deepspeed_config
    except Exception:  # pragma: no cover - defensive fallback for optional integration drift
        return
    unset_hf_deepspeed_config()


def resolve_hf_train_duration(train_cfg: Any) -> tuple[float, int]:
    """Resolve Shaft's single duration value into the two HF compatibility fields."""

    unit = str(train_cfg.duration.unit).strip().lower()
    value = float(train_cfg.duration.value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("train.duration.value must be finite and > 0.")
    if unit == "steps":
        if not value.is_integer():
            raise ValueError("train.duration.value must be an integer when unit='steps'.")
        return 1.0, int(value)
    if unit == "epochs":
        return value, -1
    raise ValueError(f"Unsupported train duration unit: {unit!r}.")


def resolve_step_sample_budget(
    config: RuntimeConfig,
    *,
    world_size: int,
) -> int | None:
    """Return the global sample plan size for a step-bounded run."""

    _, max_steps = resolve_hf_train_duration(config.train)
    if max_steps < 0:
        return None
    if config.data.batching.strategy == "dynamic_cost_aware":
        return resolve_dynamic_batch_planning_contract(
            config,
            world_size=world_size,
            optimizer_step_count=max_steps,
        ).sample_plan_horizon
    return (
        max_steps
        * int(config.train.per_device_train_batch_size)
        * int(config.train.gradient_accumulation_steps)
        * max(int(world_size), 1)
    )


def resolve_dynamic_batch_planning_contract(
    config: RuntimeConfig,
    *,
    world_size: int,
    optimizer_step_count: int,
) -> ShaftDynamicBatchPlanningContract:
    max_padded_tokens = config.data.batching.max_padded_tokens
    if max_padded_tokens is None:
        raise ValueError(
            "data.batching.max_padded_tokens is required for dynamic batching."
        )
    optimizer_batch = config.train.optimizer_batch
    return ShaftDynamicBatchPlanningContract.resolve(
        optimizer_step_count=optimizer_step_count,
        per_device_train_batch_size=config.train.per_device_train_batch_size,
        data_world_size=max(int(world_size), 1),
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        max_samples_per_microbatch=(
            config.data.batching.max_samples_per_microbatch
        ),
        max_padded_tokens=max_padded_tokens,
        max_vision_patches=config.data.batching.max_vision_patches,
        target_samples=optimizer_batch.target_samples,
        target_supervised_tokens=optimizer_batch.target_supervised_tokens,
        planning_window=config.data.batching.planning_window,
        seed=config.experiment.seed,
        rank_balance=(
            True
            if config.data.batching.rank_balance is None
            else bool(config.data.batching.rank_balance)
        ),
    )


def _build_warmup_kwargs(train_cfg: Any) -> dict[str, float | int]:
    warmup_ratio = float(train_cfg.warmup_ratio)
    if warmup_ratio <= 0:
        return {"warmup_steps": 0}
    _, max_steps = resolve_hf_train_duration(train_cfg)
    if max_steps > 0:
        return {"warmup_steps": max(1, int(round(max_steps * warmup_ratio)))}
    return {"warmup_ratio": warmup_ratio}


def build_hf_training_args(config: RuntimeConfig) -> TrainingArguments:
    train_cfg = config.train
    eval_cfg = config.eval
    eval_strategy = "no" if not eval_cfg.enabled else eval_cfg.eval_strategy
    use_bf16 = bool(train_cfg.bf16) and torch.cuda.is_available()
    dataloader_num_workers = int(config.data.num_workers)
    fsdp, fsdp_config = _build_fsdp_args(config)
    deepspeed = _build_deepspeed_arg(config)
    gradient_checkpointing = resolve_effective_gradient_checkpointing(config)
    warmup_kwargs = _build_warmup_kwargs(train_cfg)
    num_train_epochs, max_steps = resolve_hf_train_duration(train_cfg)
    if deepspeed is None:
        _reset_deepspeed_runtime_state()
    return TrainingArguments(
        output_dir=str(Path(config.experiment.output_dir)),
        run_name=config.experiment.run_id or config.experiment.name,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=int(train_cfg.per_device_train_batch_size),
        per_device_eval_batch_size=int(eval_cfg.per_device_eval_batch_size),
        gradient_accumulation_steps=int(train_cfg.gradient_accumulation_steps),
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
        lr_scheduler_type=str(train_cfg.lr_scheduler_type),
        max_grad_norm=float(train_cfg.max_grad_norm),
        bf16=use_bf16,
        use_cpu=bool(train_cfg.use_cpu),
        full_determinism=bool(train_cfg.full_determinism),
        logging_steps=int(train_cfg.logging_steps),
        save_strategy=str(train_cfg.save_strategy),
        save_steps=int(train_cfg.save_steps),
        save_total_limit=int(train_cfg.save_total_limit),
        load_best_model_at_end=bool(train_cfg.load_best_model_at_end),
        eval_strategy=eval_strategy,
        eval_steps=int(eval_cfg.eval_steps),
        metric_for_best_model=str(eval_cfg.metric_for_best_model),
        greater_is_better=bool(eval_cfg.greater_is_better),
        ddp_find_unused_parameters=bool(train_cfg.ddp_find_unused_parameters),
        fsdp=fsdp,
        fsdp_config=fsdp_config,
        deepspeed=deepspeed,
        save_on_each_node=False,
        log_on_each_node=False,
        report_to=list(train_cfg.report_to),
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=(
            int(config.data.prefetch_factor)
            if dataloader_num_workers > 0 and config.data.prefetch_factor is not None
            else None
        ),
        dataloader_pin_memory=bool(config.data.pin_memory),
        dataloader_persistent_workers=bool(config.data.persistent_workers and dataloader_num_workers > 0),
        disable_tqdm=True,
        remove_unused_columns=False,
        average_tokens_across_devices=True,
        accelerator_config={"even_batches": True},
        **warmup_kwargs,
    )
