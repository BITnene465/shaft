from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import TrainingArguments

from shaft.config import RuntimeConfig

_AUTO_FSDP_TRANSFORMER_LAYERS: dict[str, tuple[str, ...]] = {
    "qwen3vl": ("Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"),
}


def _resolve_fsdp_transformer_layers(config: RuntimeConfig) -> list[str]:
    configured = list(config.train.distributed.fsdp.transformer_layer_cls_to_wrap)
    normalized = [str(item).strip() for item in configured if str(item).strip()]
    if normalized != ["auto"]:
        return normalized
    model_type = str(config.model.model_type).strip().lower()
    resolved = _AUTO_FSDP_TRANSFORMER_LAYERS.get(model_type)
    if not resolved:
        raise ValueError(
            "train.distributed.fsdp.transformer_layer_cls_to_wrap=['auto'] is not available "
            f"for model.model_type={model_type!r}. Configure explicit transformer layer class names."
        )
    return list(resolved)


def _build_fsdp_args(config: RuntimeConfig) -> tuple[str | None, dict[str, Any] | None]:
    distributed = config.train.distributed
    if distributed.strategy != "fsdp":
        return None, None

    fsdp_cfg = distributed.fsdp
    fsdp_options = [str(fsdp_cfg.sharding_strategy)]
    fsdp_config: dict[str, Any] = {
        "activation_checkpointing": bool(fsdp_cfg.activation_checkpointing),
        "cpu_offload": bool(fsdp_cfg.cpu_offload),
        "use_orig_params": bool(fsdp_cfg.use_orig_params),
        "forward_prefetch": bool(fsdp_cfg.forward_prefetch),
        "limit_all_gathers": bool(fsdp_cfg.limit_all_gathers),
        "state_dict_type": str(fsdp_cfg.state_dict_type),
        "sync_module_states": bool(fsdp_cfg.sync_module_states),
    }
    if fsdp_cfg.backward_prefetch is not None:
        fsdp_config["backward_prefetch"] = str(fsdp_cfg.backward_prefetch)

    if fsdp_cfg.auto_wrap_policy == "transformer":
        fsdp_options.append("auto_wrap")
        fsdp_config["transformer_layer_cls_to_wrap"] = _resolve_fsdp_transformer_layers(config)
    elif fsdp_cfg.auto_wrap_policy == "size":
        fsdp_options.append("auto_wrap")
        fsdp_config["min_num_params"] = int(fsdp_cfg.min_num_params)

    return " ".join(fsdp_options), fsdp_config


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


def build_hf_training_args(config: RuntimeConfig) -> TrainingArguments:
    train_cfg = config.train
    eval_cfg = config.eval
    eval_strategy = "no" if not eval_cfg.enabled else eval_cfg.eval_strategy
    use_bf16 = bool(train_cfg.bf16) and torch.cuda.is_available()
    dataloader_num_workers = int(config.data.num_workers)
    fsdp, fsdp_config = _build_fsdp_args(config)
    deepspeed = _build_deepspeed_arg(config)
    return TrainingArguments(
        output_dir=str(Path(config.experiment.output_dir)),
        run_name=config.experiment.run_id or config.experiment.name,
        num_train_epochs=float(train_cfg.epochs),
        max_steps=int(train_cfg.max_steps),
        per_device_train_batch_size=int(train_cfg.per_device_train_batch_size),
        per_device_eval_batch_size=int(eval_cfg.per_device_eval_batch_size),
        gradient_accumulation_steps=int(train_cfg.gradient_accumulation_steps),
        gradient_checkpointing=bool(train_cfg.gradient_checkpointing),
        learning_rate=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
        warmup_ratio=float(train_cfg.warmup_ratio),
        lr_scheduler_type=str(train_cfg.lr_scheduler_type),
        max_grad_norm=float(train_cfg.max_grad_norm),
        bf16=use_bf16,
        use_cpu=bool(train_cfg.use_cpu),
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
        dataloader_pin_memory=bool(config.data.pin_memory),
        dataloader_persistent_workers=bool(config.data.persistent_workers and dataloader_num_workers > 0),
        disable_tqdm=True,
        remove_unused_columns=False,
    )
