from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import TrainingArguments
from transformers.trainer_callback import PrinterCallback

from shaft.model.finetune_plan import ShaftResolvedFinetunePlan
from shaft.model.types import ShaftModelAdapter
from shaft.utils.distributed import is_rank_zero

from .optimizer import build_optimizer_and_plan, build_optimizer_from_plan
from .optimizer_plan import (
    ShaftResolvedOptimizerPlan,
    build_resolved_optimizer_plan,
    summarize_resolved_optimizer_plan,
    write_resolved_optimizer_summary,
)
from .scheduler import build_scheduler

logger = logging.getLogger(__name__)


class ShaftOptimizerMixin:
    def __init__(
        self,
        *args: Any,
        optimizer_name: str = "adamw_torch",
        scheduler_name: str = "cosine",
        scheduler_num_cycles: float = 0.5,
        scheduler_power: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        model_adapter: ShaftModelAdapter | None = None,
        finetune_plan: ShaftResolvedFinetunePlan | None = None,
        resolved_optimizer_plan: ShaftResolvedOptimizerPlan | None = None,
        param_group_lrs: dict[str, float] | None = None,
        no_decay_name_patterns: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.optimizer_name = str(optimizer_name).strip().lower()
        self.scheduler_name = str(scheduler_name).strip().lower()
        self.scheduler_num_cycles = float(scheduler_num_cycles)
        self.scheduler_power = float(scheduler_power)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_epsilon = float(adam_epsilon)
        self.model_adapter = model_adapter
        self.finetune_plan = finetune_plan
        self.param_group_lrs = {
            str(key).strip().lower(): float(value)
            for key, value in dict(param_group_lrs or {}).items()
        }
        self.no_decay_name_patterns = [
            str(pattern).strip().lower()
            for pattern in list(no_decay_name_patterns or [])
            if str(pattern).strip()
        ]
        self.resolved_optimizer_plan = resolved_optimizer_plan
        self.resolved_optimizer_summary = None
        super().__init__(*args, **kwargs)
        # Shaft uses a progress adapter; the default HF PrinterCallback only
        # duplicates step logs and breaks progress-bar readability.
        self.remove_callback(PrinterCallback)

    @property
    def train_args(self) -> TrainingArguments:
        return self.args

    def create_optimizer(self, model: torch.nn.Module | None = None):
        if self.optimizer is None:
            optimizer_model = self.model if model is None else model
            if model is not None:
                wrapped_plan = build_resolved_optimizer_plan(
                    model=optimizer_model,
                    args=self.train_args,
                    finetune_plan=self.finetune_plan,
                    model_adapter=self.model_adapter,
                    param_group_lrs=self.param_group_lrs,
                    no_decay_name_patterns=self.no_decay_name_patterns,
                )
                if (
                    self.resolved_optimizer_plan is not None
                    and wrapped_plan.fingerprint
                    != self.resolved_optimizer_plan.fingerprint
                ):
                    raise ValueError(
                        "Wrapped-model optimizer plan differs from the resolved "
                        "exact-resume optimizer plan. FSDP requires use_orig_params=true "
                        "and stable trainable parameter names/groups; otherwise start a "
                        "new schedule."
                    )
                self.resolved_optimizer_plan = wrapped_plan
            if self.resolved_optimizer_plan is None:
                self.optimizer, self.resolved_optimizer_plan = build_optimizer_and_plan(
                    model=optimizer_model,
                    args=self.train_args,
                    optimizer_name=self.optimizer_name,
                    adam_beta1=self.adam_beta1,
                    adam_beta2=self.adam_beta2,
                    adam_epsilon=self.adam_epsilon,
                    finetune_plan=self.finetune_plan,
                    model_adapter=self.model_adapter,
                    param_group_lrs=self.param_group_lrs,
                    no_decay_name_patterns=self.no_decay_name_patterns,
                )
            else:
                self.optimizer = build_optimizer_from_plan(
                    plan=self.resolved_optimizer_plan,
                    args=self.train_args,
                    optimizer_name=self.optimizer_name,
                    adam_beta1=self.adam_beta1,
                    adam_beta2=self.adam_beta2,
                    adam_epsilon=self.adam_epsilon,
                )
            self.resolved_optimizer_summary = summarize_resolved_optimizer_plan(self.resolved_optimizer_plan)
            if is_rank_zero():
                write_resolved_optimizer_summary(
                    self.train_args.output_dir,
                    self.resolved_optimizer_summary,
                )
                logger.info(
                    "[startup] resolved optimizer groups: %s",
                    self.resolved_optimizer_summary.to_log_dict(),
                )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None):
        if self.lr_scheduler is None:
            if optimizer is None:
                optimizer = self.optimizer
            if optimizer is None:
                raise ValueError("Optimizer must be created before scheduler.")
            self.lr_scheduler = build_scheduler(
                scheduler_name=self.scheduler_name,
                optimizer=optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                num_cycles=self.scheduler_num_cycles,
                power=self.scheduler_power,
            )
        return self.lr_scheduler
