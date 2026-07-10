from __future__ import annotations

import argparse
import logging
from typing import Any

from shaft.config import RuntimeConfig, load_config
from shaft.observability import bind_log_context, configure_logging, set_log_context
from shaft.training.distributed import destroy_process_group_if_initialized


def run_sft(config: RuntimeConfig) -> dict[str, Any]:
    from shaft.pipeline import run_sft as _run_sft

    return _run_sft(config)


def run_rlhf(config: RuntimeConfig) -> dict[str, Any]:
    from shaft.pipeline import run_rlhf as _run_rlhf

    return _run_rlhf(config)


def _as_bool(text: str) -> bool:
    normalized = str(text).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {text!r}")


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to YAML train config.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    duration_group = parser.add_mutually_exclusive_group()
    duration_group.add_argument("--epochs", type=float, default=None)
    duration_group.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", type=_as_bool, default=None)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--mix-strategy", choices=["concat", "weighted"], default=None)
    parser.add_argument("--optimizer", dest="optimizer_name", default=None)
    parser.add_argument("--scheduler", dest="scheduler_name", default=None)
    parser.add_argument("--scheduler-num-cycles", type=float, default=None)
    parser.add_argument("--scheduler-power", type=float, default=None)
    parser.add_argument("--loss", dest="loss_name", default=None)
    parser.add_argument("--loss-scale", dest="loss_scale", default=None)
    parser.add_argument("--finetune-mode", choices=["full", "lora", "dora", "qlora"], default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--qlora-load-in-4bit", type=_as_bool, default=None)
    parser.add_argument("--use-cpu", type=_as_bool, default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--resume-from", default=None)


def apply_common_overrides(config: RuntimeConfig, args: argparse.Namespace) -> RuntimeConfig:
    train_config = config.train
    eval_config = config.eval
    run_id = getattr(args, "run_id", None)
    seed = getattr(args, "seed", None)
    epochs = getattr(args, "epochs", None)
    max_steps = getattr(args, "max_steps", None)
    gradient_checkpointing = getattr(args, "gradient_checkpointing", None)
    learning_rate = getattr(args, "learning_rate", None)
    train_batch_size = getattr(args, "train_batch_size", None)
    eval_batch_size = getattr(args, "eval_batch_size", None)
    mix_strategy = getattr(args, "mix_strategy", None)
    optimizer_name = getattr(args, "optimizer_name", None)
    scheduler_name = getattr(args, "scheduler_name", None)
    scheduler_num_cycles = getattr(args, "scheduler_num_cycles", None)
    scheduler_power = getattr(args, "scheduler_power", None)
    loss_name = getattr(args, "loss_name", None)
    loss_scale = getattr(args, "loss_scale", None)
    finetune_mode = getattr(args, "finetune_mode", None)
    lora_r = getattr(args, "lora_r", None)
    lora_alpha = getattr(args, "lora_alpha", None)
    lora_dropout = getattr(args, "lora_dropout", None)
    qlora_load_in_4bit = getattr(args, "qlora_load_in_4bit", None)
    use_cpu = getattr(args, "use_cpu", None)
    init_from = getattr(args, "init_from", None)
    resume_from = getattr(args, "resume_from", None)

    if run_id is not None:
        config.experiment.run_id = str(run_id)
    if seed is not None:
        config.experiment.seed = int(seed)
    if epochs is not None and max_steps is not None:
        raise ValueError("--epochs and --max-steps are mutually exclusive duration overrides.")
    if epochs is not None:
        if float(epochs) <= 0:
            raise ValueError("--epochs must be > 0.")
        train_config.duration.unit = "epochs"
        train_config.duration.value = float(epochs)
    if max_steps is not None:
        if int(max_steps) <= 0:
            raise ValueError("--max-steps must be > 0.")
        train_config.duration.unit = "steps"
        train_config.duration.value = float(max_steps)
    if gradient_checkpointing is not None:
        train_config.gradient_checkpointing = bool(gradient_checkpointing)
    if learning_rate is not None:
        train_config.learning_rate = float(learning_rate)
    if train_batch_size is not None:
        train_config.per_device_train_batch_size = int(train_batch_size)
    if eval_batch_size is not None:
        eval_config.per_device_eval_batch_size = int(eval_batch_size)
    if mix_strategy is not None:
        config.data.mix_strategy = str(mix_strategy)
    if optimizer_name is not None:
        train_config.optimizer_name = str(optimizer_name)
    if scheduler_name is not None:
        train_config.scheduler_name = str(scheduler_name)
    if scheduler_num_cycles is not None:
        train_config.scheduler_num_cycles = float(scheduler_num_cycles)
    if scheduler_power is not None:
        train_config.scheduler_power = float(scheduler_power)
    if loss_name is not None:
        train_config.loss_name = str(loss_name)
    if loss_scale is not None:
        train_config.loss_scale = str(loss_scale)
    if finetune_mode is not None:
        config.model.finetune.mode = str(finetune_mode)
    if lora_r is not None:
        config.model.finetune.lora_r = int(lora_r)
    if lora_alpha is not None:
        config.model.finetune.lora_alpha = int(lora_alpha)
    if lora_dropout is not None:
        config.model.finetune.lora_dropout = float(lora_dropout)
    if qlora_load_in_4bit is not None:
        config.model.finetune.qlora_load_in_4bit = bool(qlora_load_in_4bit)
    if use_cpu is not None:
        train_config.use_cpu = bool(use_cpu)
    if init_from is not None:
        train_config.init_from_checkpoint = str(init_from)
    if resume_from is not None:
        train_config.resume_from_checkpoint = str(resume_from)
    return config


def run_from_args(
    args: argparse.Namespace,
    *,
    forced_algorithm: str | None = None,
    allowed_algorithms: set[str] | None = None,
) -> dict[str, Any]:
    config = load_config(args.config)
    config = apply_common_overrides(config, args)
    configure_logging(config.logging, run_id=config.experiment.run_id)
    set_log_context(algorithm=config.algorithm.name)
    logger = logging.getLogger(__name__)
    logger.info("[startup] config loaded")

    algorithm_name = forced_algorithm
    if algorithm_name is None and hasattr(args, "algorithm") and args.algorithm is not None:
        algorithm_name = str(args.algorithm)
    if algorithm_name is not None:
        if allowed_algorithms is not None and algorithm_name not in allowed_algorithms:
            allowed_sorted = ",".join(sorted(allowed_algorithms))
            raise ValueError(f"Unsupported algorithm={algorithm_name!r}. Allowed: {allowed_sorted}")
        config.algorithm.name = algorithm_name

    try:
        with bind_log_context(algorithm=config.algorithm.name):
            logger.info("[startup] start training (algorithm=%s)...", config.algorithm.name)
            if config.algorithm.name == "sft":
                metrics = run_sft(config)
            elif config.algorithm.name in {"dpo", "ppo", "grpo"}:
                metrics = run_rlhf(config)
            else:
                raise ValueError(f"Unsupported algorithm={config.algorithm.name!r}.")
        logger.info("[done] train metrics: %s", metrics)
        return metrics
    finally:
        destroy_process_group_if_initialized()
