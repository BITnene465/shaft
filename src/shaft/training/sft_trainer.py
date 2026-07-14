from __future__ import annotations

import math
from pathlib import Path
import random
import time
from typing import Any
import warnings

import numpy as np
import torch
import transformers.trainer as hf_trainer_module
from transformers.debug_utils import DebugOption
from transformers import Trainer

from shaft.config.training import EvalConfig
from shaft.utils.distributed import all_gather_objects, barrier_if_distributed
from .batch_planning import (
    BATCH_PLANNING_CALLBACK_NAME,
    BATCH_PLANNING_CHECKPOINT_COMPLETION_FILENAME,
    write_batch_planning_checkpoint_completion,
)
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
        eval_data_collator: Any | None = None,
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
        self.eval_data_collator = eval_data_collator
        self._shaft_train_data_collator = self.data_collator
        # HF uses this flag to collect one optimizer batch before backward and pass
        # its global normalization denominator into compute_loss.
        self.model_accepts_loss_kwargs = True

    def get_eval_dataloader(self, eval_dataset: Any = None):
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        cache_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        resolved_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator
        if (
            self.eval_data_collator is not None
            and data_collator is self._shaft_train_data_collator
        ):
            data_collator = self.eval_data_collator
        return self._build_eval_loader(
            dataset=resolved_dataset,
            data_collator=data_collator,
            cache_key=str(cache_key),
            description="Evaluation",
        )

    def get_online_eval_dataloader(
        self,
        eval_dataset: Any,
        *,
        data_collator: Any,
        dataset_key: str,
    ):
        resolved_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
        )
        normalized_key = str(dataset_key).strip()
        if not normalized_key:
            raise ValueError("Online eval dataset_key must not be empty.")
        return self._build_eval_loader(
            dataset=resolved_dataset,
            data_collator=data_collator,
            cache_key=f"shaft-online:{normalized_key}",
            description="Online Evaluation",
        )

    def _build_eval_loader(
        self,
        *,
        dataset: Any,
        data_collator: Any,
        cache_key: str,
        description: str,
    ):
        if dataset is None:
            raise ValueError(f"Trainer: {description.lower()} requires an eval_dataset.")
        cached_loaders = getattr(self, "_eval_dataloaders", {})
        if self.args.dataloader_persistent_workers and cache_key in cached_loaders:
            return cached_loaders[cache_key]
        previous_collator = self.data_collator
        self.data_collator = data_collator
        try:
            return self._get_dataloader(
                dataset=dataset,
                description=description,
                batch_size=self.args.eval_batch_size,
                sampler_fn=self._get_eval_sampler,
                dataloader_key=cache_key,
            )
        finally:
            self.data_collator = previous_collator

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        prepared = inputs
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

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ):
        if not self.is_in_train:
            return self._evaluate_impl(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_cpu_rng_state = torch.random.get_rng_state()
        cuda_device: int | None = None
        cuda_rng_state: torch.Tensor | None = None
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            cuda_device = torch.cuda.current_device()
            cuda_rng_state = torch.cuda.get_rng_state(cuda_device)
        try:
            return self._evaluate_impl(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            # A persistent eval loader is created only on its first evaluation.
            # Iterator construction consumes host RNG and would otherwise make a
            # resumed run's later RNG snapshot differ from an uninterrupted run.
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_cpu_rng_state)
            if cuda_device is not None and cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, cuda_device)

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
            eval_dataloader = self._build_eval_loader(
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

    def _save_checkpoint(self, model, trial) -> None:
        barrier_if_distributed()
        has_planning_callback = any(
            callback.__class__.__name__ == BATCH_PLANNING_CALLBACK_NAME
            for callback in self.callback_handler.callbacks
        )
        checkpoint_path = (
            Path(self._get_output_dir(trial=trial))
            / f"checkpoint-{int(self.state.global_step)}"
        )
        revoke_error: Exception | None = None
        if has_planning_callback and self.is_world_process_zero():
            try:
                (checkpoint_path / BATCH_PLANNING_CHECKPOINT_COMPLETION_FILENAME).unlink(
                    missing_ok=True
                )
            except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
                revoke_error = exc
        if has_planning_callback:
            self._raise_synchronized_operation_error(
                "batch-planning checkpoint begin",
                revoke_error,
            )
        save_total_limit = self.args.save_total_limit
        local_error: Exception | None = None
        # HF rotates on rank zero before a peer-rank RNG write failure can be
        # observed. Defer rotation until every rank confirms the full save.
        if has_planning_callback:
            self.args.save_total_limit = None
        try:
            super()._save_checkpoint(model, trial)
        except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
            local_error = exc
        finally:
            if has_planning_callback:
                self.args.save_total_limit = save_total_limit
        self._raise_synchronized_operation_error("checkpoint save", local_error)
        if not has_planning_callback:
            return

        completion_error: Exception | None = None
        if self.is_world_process_zero():
            try:
                write_batch_planning_checkpoint_completion(checkpoint_path)
            except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
                completion_error = exc
        self._raise_synchronized_operation_error(
            "batch-planning checkpoint commit",
            completion_error,
        )

        rotation_error: Exception | None = None
        if self.args.should_save:
            try:
                hf_trainer_module.rotate_checkpoints(
                    output_dir=self._get_output_dir(trial=trial),
                    save_total_limit=save_total_limit,
                    best_model_checkpoint=self.state.best_model_checkpoint,
                    use_mtime=True,
                )
            except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
                rotation_error = exc
        self._raise_synchronized_operation_error(
            "checkpoint rotation",
            rotation_error,
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
        self._raise_synchronized_operation_error("model save", local_error)

    @staticmethod
    def _raise_synchronized_operation_error(
        operation: str,
        local_error: Exception | None,
    ) -> None:
        statuses = all_gather_objects(
            {
                "ok": local_error is None,
                "error_type": (
                    None if local_error is None else type(local_error).__name__
                ),
                "error": None if local_error is None else str(local_error),
            }
        )
        failures = [status for status in statuses if not bool(status.get("ok"))]
        if not failures:
            return
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"Distributed {operation} failed on a peer rank: {failures!r}.")
