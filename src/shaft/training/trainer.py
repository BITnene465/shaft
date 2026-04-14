from __future__ import annotations

from typing import Any

import torch
from transformers import Trainer, TrainingArguments

from .loss import build_loss
from .optimizer import build_optimizer
from .scheduler import build_scheduler


class ShaftSFTTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        loss_name: str = "auto",
        optimizer_name: str = "adamw_torch",
        scheduler_name: str = "cosine",
        scheduler_num_cycles: float = 0.5,
        scheduler_power: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        ignore_index: int = -100,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_name = str(loss_name).strip().lower()
        self.loss_fn = build_loss(self.loss_name)
        self.optimizer_name = str(optimizer_name).strip().lower()
        self.scheduler_name = str(scheduler_name).strip().lower()
        self.scheduler_num_cycles = float(scheduler_num_cycles)
        self.scheduler_power = float(scheduler_power)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_epsilon = float(adam_epsilon)
        self.ignore_index = int(ignore_index)

    @property
    def train_args(self) -> TrainingArguments:
        return self.args

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        _ = num_items_in_batch
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(
            outputs=outputs,
            labels=labels,
            ignore_index=self.ignore_index,
            model=model,
            inputs=inputs,
        )
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = build_optimizer(
                model=self.model,
                args=self.train_args,
                optimizer_name=self.optimizer_name,
                adam_beta1=self.adam_beta1,
                adam_beta2=self.adam_beta2,
                adam_epsilon=self.adam_epsilon,
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
