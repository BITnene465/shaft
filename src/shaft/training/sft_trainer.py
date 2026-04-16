from __future__ import annotations

import math
import time
from typing import Any

import torch
import transformers.trainer as hf_trainer_module
from transformers.debug_utils import DebugOption
from transformers import Trainer, TrainingArguments

from shaft.utils.distributed import barrier_if_distributed
from .loss import build_loss
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .online_eval import ShaftOnlineEvalRunner


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
        online_eval_runner: ShaftOnlineEvalRunner | None = None,
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
        self.online_eval_runner = online_eval_runner

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

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ):
        barrier_if_distributed()
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, dataset_value in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=dataset_value if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            barrier_if_distributed()
            return metrics

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = hf_trainer_module.tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            hf_trainer_module.speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size) if total_batch_size > 0 else 0,
            )
        )

        metrics = dict(output.metrics)
        loss_key = f"{metric_key_prefix}_loss"
        report_metrics: dict[str, float] = {}
        if loss_key in output.metrics:
            report_metrics[loss_key] = float(output.metrics[loss_key])
        if self.online_eval_runner is not None and eval_dataset is not None:
            online_metrics = self.online_eval_runner.evaluate(
                self,
                eval_dataset=eval_dataset,
                metric_key_prefix=metric_key_prefix,
            )
            metrics.update(online_metrics)
            final_score_key = f"{metric_key_prefix}_final_score"
            if final_score_key in online_metrics:
                report_metrics[final_score_key] = float(online_metrics[final_score_key])

        self.log(report_metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            hf_trainer_module.xm.master_print(hf_trainer_module.met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, report_metrics)
        self._memory_tracker.stop_and_update_metrics(report_metrics)

        barrier_if_distributed()
        return metrics

    def _save_checkpoint(self, model, trial) -> None:
        barrier_if_distributed()
        super()._save_checkpoint(model, trial)
        barrier_if_distributed()

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        barrier_if_distributed()
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        barrier_if_distributed()
