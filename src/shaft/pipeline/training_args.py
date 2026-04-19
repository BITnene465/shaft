from __future__ import annotations

from pathlib import Path

import torch
from transformers import TrainingArguments

from shaft.config import RuntimeConfig


def build_hf_training_args(config: RuntimeConfig) -> TrainingArguments:
    train_cfg = config.train
    eval_cfg = config.eval
    eval_strategy = "no" if not eval_cfg.enabled else eval_cfg.eval_strategy
    use_bf16 = bool(train_cfg.bf16) and torch.cuda.is_available()
    dataloader_num_workers = int(config.data.num_workers)
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
        save_on_each_node=False,
        log_on_each_node=False,
        report_to=list(train_cfg.report_to),
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=bool(config.data.pin_memory),
        dataloader_persistent_workers=bool(config.data.persistent_workers and dataloader_num_workers > 0),
        disable_tqdm=True,
        remove_unused_columns=False,
    )
