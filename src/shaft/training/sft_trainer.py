from __future__ import annotations

import math
import time
from typing import Any

import torch
import transformers.trainer as hf_trainer_module
from transformers.debug_utils import DebugOption
from transformers import Trainer

from shaft.config.training import EvalConfig
from shaft.utils.distributed import barrier_if_distributed
from .eval_policy import aggregate_weighted_dataset_values
from .loss import build_loss
from .optimizer_mixin import ShaftOptimizerMixin
from .online_eval import ShaftOnlineEvalRunner
from .train_sampler_mixin import ShaftTrainSamplerMixin


class ShaftSFTTrainer(ShaftOptimizerMixin, ShaftTrainSamplerMixin, Trainer):
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
        eval_config: EvalConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            scheduler_num_cycles=scheduler_num_cycles,
            scheduler_power=scheduler_power,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            **kwargs,
        )
        self.loss_name = str(loss_name).strip().lower()
        self.loss_fn = build_loss(self.loss_name)
        self.ignore_index = int(ignore_index)
        self.online_eval_runner = online_eval_runner
        self.eval_config = eval_config

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        _ = num_items_in_batch
        model_inputs = dict(inputs)
        labels = model_inputs.get("labels")
        loss_scale = model_inputs.pop("loss_scale", None)
        outputs = model(**model_inputs)
        loss = self.loss_fn(
            outputs=outputs,
            labels=labels,
            ignore_index=self.ignore_index,
            loss_scale=loss_scale,
            model=model,
            inputs=model_inputs,
        )
        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ):
        barrier_if_distributed()
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset

        self._memory_tracker.start()
        report_metrics: dict[str, float] = {}
        metrics: dict[str, float] = {}
        if isinstance(eval_dataset, dict):
            loss_metrics, loss_report_metrics = self._evaluate_named_datasets(
                eval_datasets=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            metrics.update(loss_metrics)
            report_metrics.update(loss_report_metrics)
        else:
            merged_metrics = self._evaluate_single_dataset(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            metrics.update(merged_metrics)
            loss_key = f"{metric_key_prefix}_loss"
            if loss_key in merged_metrics:
                report_metrics[loss_key] = float(merged_metrics[loss_key])
        if self.online_eval_runner is not None and eval_dataset is not None:
            online_metrics = self.online_eval_runner.evaluate(
                self,
                eval_dataset=eval_dataset,
                metric_key_prefix=metric_key_prefix,
            )
            metrics.update(online_metrics)
            report_metrics.update({key: float(value) for key, value in online_metrics.items()})

        self.log(report_metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            hf_trainer_module.xm.master_print(hf_trainer_module.met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, report_metrics)
        self._memory_tracker.stop_and_update_metrics(report_metrics)

        barrier_if_distributed()
        return metrics

    def _evaluate_single_dataset(
        self,
        *,
        eval_dataset: Any,
        ignore_keys: list[str] | None,
        metric_key_prefix: str,
    ) -> dict[str, float]:
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
        return {key: float(value) for key, value in output.metrics.items()}

    def _evaluate_named_datasets(
        self,
        *,
        eval_datasets: dict[str, Any],
        ignore_keys: list[str] | None,
        metric_key_prefix: str,
    ) -> tuple[dict[str, float], dict[str, float]]:
        if self.eval_config is not None and not self.eval_config.loss_metrics_enabled:
            return {}, {}
        metrics: dict[str, float] = {}
        report_metrics: dict[str, float] = {}
        loss_values: dict[str, float] = {}
        for dataset_name in sorted(eval_datasets):
            dataset_metrics = self._evaluate_single_dataset(
                eval_dataset=eval_datasets[dataset_name],
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_{dataset_name}",
            )
            metrics.update(dataset_metrics)
            loss_key = f"{metric_key_prefix}_{dataset_name}_loss"
            if loss_key in dataset_metrics:
                loss_value = float(dataset_metrics[loss_key])
                report_metrics[loss_key] = loss_value
                loss_values[dataset_name] = loss_value
        final_loss = self._aggregate_final_loss(loss_values)
        if final_loss is not None:
            metrics[f"{metric_key_prefix}_final_loss"] = final_loss
            report_metrics[f"{metric_key_prefix}_final_loss"] = final_loss
        return metrics, report_metrics

    def _aggregate_final_loss(self, loss_values: dict[str, float]) -> float | None:
        if not loss_values:
            return None
        if self.eval_config is None or not self.eval_config.datasets:
            return None
        return aggregate_weighted_dataset_values(
            values_by_dataset=loss_values,
            eval_config=self.eval_config,
            metric_name="loss",
        )

    def _save_checkpoint(self, model, trial) -> None:
        barrier_if_distributed()
        super()._save_checkpoint(model, trial)
        barrier_if_distributed()

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        barrier_if_distributed()
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        barrier_if_distributed()
