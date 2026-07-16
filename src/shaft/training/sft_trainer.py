from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any
import warnings

import torch
import transformers.trainer as hf_trainer_module
from transformers.debug_utils import DebugOption
from transformers import Trainer

from shaft.config.training import EvalConfig
from shaft.utils.distributed import barrier_if_distributed
from .checkpointing import ShaftCheckpointCommitMixin
from .eval_policy import aggregate_weighted_dataset_values
from .efficiency import (
    ShaftTrainingEfficiencyMonitor,
    prepare_training_efficiency_checkpoint,
)
from .eval_dataloader import ShaftEvalDataLoaderMixin
from .loss import build_loss
from .optimizer_mixin import ShaftOptimizerMixin
from .online_eval import ShaftOnlineEvalRunner
from .reproducibility import isolate_training_rng_during_eval
from .train_sampler_mixin import ShaftTrainSamplerMixin


class ShaftSFTTrainer(
    ShaftCheckpointCommitMixin,
    ShaftEvalDataLoaderMixin,
    ShaftOptimizerMixin,
    ShaftTrainSamplerMixin,
    Trainer,
):
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
        eval_data_collator: Any | None = None,
        efficiency_monitor: ShaftTrainingEfficiencyMonitor | None = None,
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
        self.efficiency_monitor = efficiency_monitor
        self._configure_eval_data_collator(eval_data_collator)
        # HF uses this flag to collect one optimizer batch before backward and pass
        # its global normalization denominator into compute_loss.
        self.model_accepts_loss_kwargs = True

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if "_shaft_batch_stats" not in inputs and "_shaft_varlen_layout" not in inputs:
            return super()._prepare_inputs(inputs)
        prepared = dict(inputs)
        prepared.pop("_shaft_batch_stats", None)
        if "_shaft_varlen_layout" in prepared:
            if self.model_adapter is None:
                raise ValueError(
                    "Shaft varlen inputs require a model adapter sequence execution policy."
                )
            prepared = self.model_adapter.prepare_sequence_training_inputs(
                model=self.model,
                inputs=prepared,
            )
        return super()._prepare_inputs(prepared)

    def get_batch_samples(
        self,
        epoch_iterator: Any,
        num_batches: int,
        device: torch.device,
    ) -> tuple[list[Any], torch.Tensor | int | None]:
        if self.efficiency_monitor is None or not self.efficiency_monitor.enabled:
            return super().get_batch_samples(epoch_iterator, num_batches, device)
        acquire_started_at = time.perf_counter()
        batch_samples: list[Any] = []
        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break
        acquire_seconds = time.perf_counter() - acquire_started_at

        prepare_started_at = time.perf_counter()
        num_items = self._get_num_items_in_batch(batch_samples, device)
        prepare_seconds = time.perf_counter() - prepare_started_at
        if self.efficiency_monitor is not None and batch_samples:
            self.efficiency_monitor.stage(
                batch_samples,
                host_batch_acquire_seconds=acquire_seconds,
                batch_prepare_seconds=prepare_seconds,
            )
        return batch_samples, num_items

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        if self.efficiency_monitor is None or not self.efficiency_monitor.enabled:
            return super().training_step(model, inputs, num_items_in_batch)
        started_at = time.perf_counter()
        if (
            self.efficiency_monitor is not None
            and self.efficiency_monitor.device_timing
            and self.args.device.type == "cuda"
        ):
            self.efficiency_monitor.start_device_frame()
        result = super().training_step(model, inputs, num_items_in_batch)
        if self.efficiency_monitor is not None:
            self.efficiency_monitor.record_training_step(
                time.perf_counter() - started_at,
            )
        return result

    def finalize_training_efficiency(self) -> dict[str, float]:
        if self.efficiency_monitor is None:
            return {}
        if not self.efficiency_monitor.enabled:
            return {}
        _, metrics = self.efficiency_monitor.finalize(
            final_global_step=int(self.state.global_step),
            device=self.args.device,
        )
        return metrics

    def _get_num_items_in_batch(
        self,
        batch_samples: list[dict[str, Any]],
        device: torch.device,
    ) -> torch.Tensor | int | None:
        if not batch_samples or "labels" not in batch_samples[0]:
            return None
        labels_device = batch_samples[0]["labels"].device
        denominator = torch.zeros((), dtype=torch.float32, device=labels_device)
        for batch in batch_samples:
            labels = batch["labels"]
            shifted_labels = labels[..., 1:]
            valid = shifted_labels.ne(self.ignore_index)
            loss_scale = batch.get("loss_scale")
            if loss_scale is None:
                denominator = denominator + valid.sum().to(dtype=torch.float32)
            else:
                shifted_scale = loss_scale[..., 1:].to(
                    device=labels.device,
                    dtype=torch.float32,
                )
                denominator = denominator + (
                    shifted_scale * valid.to(dtype=torch.float32)
                ).sum()

        denominator = denominator.to(device)
        if self.args.average_tokens_across_devices:
            if self.args.world_size > 1:
                denominator = self.accelerator.gather(denominator).sum()
        elif self.args.n_gpu > 1:
            denominator = denominator / self.args.n_gpu
        parallelism_config = getattr(self.accelerator, "parallelism_config", None)
        if parallelism_config is not None:
            denominator = denominator / parallelism_config.non_data_parallel_size
        return denominator

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        model_inputs = dict(inputs)
        labels = model_inputs.get("labels")
        loss_scale = model_inputs.pop("loss_scale", None)
        if self.loss_name in {"auto", "causal_lm"}:
            # Shaft's built-in SFT losses own shifted CE and its GA/DP denominator.
            # Forwarding labels would make HF compute a second full-vocabulary CE.
            model_inputs.pop("labels", None)
        outputs = model(**model_inputs)
        loss = self.loss_fn(
            outputs=outputs,
            labels=labels,
            ignore_index=self.ignore_index,
            loss_scale=loss_scale,
            model=model,
            inputs=model_inputs,
            normalization_denominator=num_items_in_batch,
        )
        if num_items_in_batch is not None and self.args.average_tokens_across_devices:
            data_parallel_scale = self.accelerator.num_processes
            parallelism_config = getattr(self.accelerator, "parallelism_config", None)
            if parallelism_config is not None:
                data_parallel_scale //= parallelism_config.tp_size
            loss = loss * (data_parallel_scale if self.args.n_gpu <= 1 else self.args.n_gpu)
        return (loss, outputs) if return_outputs else loss

    @isolate_training_rng_during_eval
    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ):
        return self._evaluate_impl(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

    def _evaluate_impl(
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
        dataloader_key: str | None = None,
    ) -> dict[str, float]:
        if dataloader_key is None:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
        else:
            eval_dataloader = self._build_shaft_eval_loader(
                dataset=eval_dataset,
                data_collator=self.eval_data_collator or self.data_collator,
                cache_key=dataloader_key,
                description="Evaluation",
            )
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = hf_trainer_module.tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()
        use_legacy_prediction_loop = bool(getattr(self.args, "use_legacy_prediction_loop", False))
        eval_loop = self.prediction_loop if use_legacy_prediction_loop else self.evaluation_loop
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
                dataloader_key=dataset_name,
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

    def _prepare_shaft_checkpoint_save(self, checkpoint_path: Path) -> None:
        efficiency_generation: str | None = None
        if (
            self.efficiency_monitor is not None
            and self.efficiency_monitor.enabled
            and self.efficiency_monitor.persist
        ):
            efficiency_generation = self.efficiency_monitor.snapshot_generation
        prepare_training_efficiency_checkpoint(
            checkpoint_path,
            global_step=int(self.state.global_step),
            generation=efficiency_generation,
        )

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        if not self._requires_cpu_distributed_state_restore(checkpoint):
            super()._load_optimizer_and_scheduler(checkpoint)
            return
        assert checkpoint is not None
        checkpoint_path = Path(checkpoint)
        optimizer_path = checkpoint_path / hf_trainer_module.OPTIMIZER_NAME
        if not optimizer_path.is_file():
            optimizer_path = checkpoint_path / hf_trainer_module.OPTIMIZER_NAME_BIN
        scheduler_path = checkpoint_path / hf_trainer_module.SCHEDULER_NAME
        if not optimizer_path.is_file() or not scheduler_path.is_file():
            super()._load_optimizer_and_scheduler(checkpoint)
            return

        # Transformers maps distributed optimizer state directly to args.device.
        # torchrun CPU devices are tagged cpu:0/cpu:1, which torch.load cannot restore.
        hf_trainer_module.check_torch_load_is_safe()
        self.optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=True)
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            hf_trainer_module.check_torch_load_is_safe()
            self.lr_scheduler.load_state_dict(
                torch.load(scheduler_path, map_location="cpu", weights_only=True)
            )
        hf_trainer_module.reissue_pt_warnings(caught_warnings)

    def _requires_cpu_distributed_state_restore(
        self,
        checkpoint: str | None,
    ) -> bool:
        return bool(
            checkpoint is not None
            and self.args.world_size > 1
            and self.args.device.type == "cpu"
            and not self.is_deepspeed_enabled
            and not self.is_fsdp_enabled
            and not hf_trainer_module.is_torch_xla_available()
            and not hf_trainer_module.is_sagemaker_mp_enabled()
        )

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        barrier_if_distributed()
        local_error: Exception | None = None
        try:
            super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
            local_error = exc
        self._raise_synchronized_checkpoint_error("model save", local_error)
