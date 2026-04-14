from __future__ import annotations

import argparse
import logging
from typing import Any

from shaft.config import RuntimeConfig, load_config
from shaft.observability import bind_log_context, configure_logging, set_log_context
from shaft.pipeline import run_rlhf, run_train


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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--mix-strategy", choices=["concat", "interleave_under", "interleave_over"], default=None)
    parser.add_argument("--optimizer", dest="optimizer_name", default=None)
    parser.add_argument("--scheduler", dest="scheduler_name", default=None)
    parser.add_argument("--scheduler-num-cycles", type=float, default=None)
    parser.add_argument("--scheduler-power", type=float, default=None)
    parser.add_argument("--loss", dest="loss_name", default=None)
    parser.add_argument("--finetune-mode", choices=["full", "lora", "dora", "qlora"], default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--qlora-load-in-4bit", type=_as_bool, default=None)
    parser.add_argument("--use-cpu", type=_as_bool, default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--resume-from", default=None)


def apply_common_overrides(config: RuntimeConfig, args: argparse.Namespace) -> RuntimeConfig:
    sft_train = config.sft.train
    sft_eval = config.sft.eval
    run_id = getattr(args, "run_id", None)
    seed = getattr(args, "seed", None)
    epochs = getattr(args, "epochs", None)
    max_steps = getattr(args, "max_steps", None)
    learning_rate = getattr(args, "learning_rate", None)
    train_batch_size = getattr(args, "train_batch_size", None)
    eval_batch_size = getattr(args, "eval_batch_size", None)
    mix_strategy = getattr(args, "mix_strategy", None)
    optimizer_name = getattr(args, "optimizer_name", None)
    scheduler_name = getattr(args, "scheduler_name", None)
    scheduler_num_cycles = getattr(args, "scheduler_num_cycles", None)
    scheduler_power = getattr(args, "scheduler_power", None)
    loss_name = getattr(args, "loss_name", None)
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
    if epochs is not None:
        sft_train.epochs = int(epochs)
    if max_steps is not None:
        sft_train.max_steps = int(max_steps)
    if learning_rate is not None:
        sft_train.learning_rate = float(learning_rate)
    if train_batch_size is not None:
        sft_train.per_device_train_batch_size = int(train_batch_size)
    if eval_batch_size is not None:
        sft_eval.per_device_eval_batch_size = int(eval_batch_size)
    if mix_strategy is not None:
        config.data.mix_strategy = str(mix_strategy)
    if optimizer_name is not None:
        sft_train.optimizer_name = str(optimizer_name)
    if scheduler_name is not None:
        sft_train.scheduler_name = str(scheduler_name)
    if scheduler_num_cycles is not None:
        sft_train.scheduler_num_cycles = float(scheduler_num_cycles)
    if scheduler_power is not None:
        sft_train.scheduler_power = float(scheduler_power)
    if loss_name is not None:
        sft_train.loss_name = str(loss_name)
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
        sft_train.use_cpu = bool(use_cpu)
    if init_from is not None:
        sft_train.init_from_checkpoint = str(init_from)
    if resume_from is not None:
        sft_train.resume_from_checkpoint = str(resume_from)
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

    with bind_log_context(algorithm=config.algorithm.name):
        logger.info("[startup] start training (algorithm=%s)...", config.algorithm.name)
        if config.algorithm.name == "sft":
            metrics = run_train(config)
        elif config.algorithm.name in {"dpo", "ppo"}:
            metrics = run_rlhf(config)
        else:
            raise ValueError(f"Unsupported algorithm={config.algorithm.name!r}.")
    logger.info("[done] train metrics: %s", metrics)
    return metrics
