from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import TrainingArguments

from shaft.model.finetune_plan import ShaftResolvedFinetunePlan
from shaft.model.types import ShaftModelAdapter

from .optimizer import build_optimizer_and_plan
from .optimizer_plan import summarize_resolved_optimizer_plan, write_resolved_optimizer_summary
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
        param_group_lrs: dict[str, float] | None = None,
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
        self.resolved_optimizer_plan = None
        self.resolved_optimizer_summary = None
        super().__init__(*args, **kwargs)

    @property
    def train_args(self) -> TrainingArguments:
        return self.args

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer, self.resolved_optimizer_plan = build_optimizer_and_plan(
                model=self.model,
                args=self.train_args,
                optimizer_name=self.optimizer_name,
                adam_beta1=self.adam_beta1,
                adam_beta2=self.adam_beta2,
                adam_epsilon=self.adam_epsilon,
                finetune_plan=self.finetune_plan,
                model_adapter=self.model_adapter,
                param_group_lrs=self.param_group_lrs,
            )
            self.resolved_optimizer_summary = summarize_resolved_optimizer_plan(self.resolved_optimizer_plan)
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
