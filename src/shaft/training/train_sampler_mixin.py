from __future__ import annotations

from functools import partial
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available


if is_datasets_available():
    import datasets


class ShaftTrainSamplerMixin:
    """Inject Shaft sample or batch plans through HF's DataLoader extension points."""

    def __init__(
        self,
        *args: Any,
        train_sampler: Any = None,
        train_batch_sampler: Any = None,
        **kwargs: Any,
    ) -> None:
        self.train_sampler = train_sampler
        self.train_batch_sampler = train_batch_sampler
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if self.train_sampler is not None:
            return self.train_sampler
        return super()._get_train_sampler(train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_batch_sampler is None:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset,
                description="Training",
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator,
                description="Training",
            )

        should_fork = (
            torch.backends.mps.is_available()
            and self.args.dataloader_num_workers > 1
        )
        dataloader_params: dict[str, Any] = {
            "batch_sampler": self.train_batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": "fork" if should_fork else None,
        }
        if not isinstance(train_dataset, IterableDataset):
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
        dataloader = DataLoader(train_dataset, **dataloader_params)
        previous_even_batches = self.accelerator.even_batches
        self.accelerator.even_batches = False
        try:
            return self.accelerator.prepare(dataloader)
        finally:
            # Dynamic cardinality is train-only. Distributed eval must keep equal
            # per-rank step counts or its prediction collectives can deadlock.
            self.accelerator.even_batches = previous_even_batches

    def set_initial_training_values(
        self,
        args: Any,
        dataloader: DataLoader,
        total_train_batch_size: int | None = None,
    ) -> tuple[Any, ...]:
        if total_train_batch_size is None:
            values = list(super().set_initial_training_values(args, dataloader))
        else:
            # Transformers 4.57 passes this third positional argument; 5.x
            # resolves it inside Trainer and calls the two-argument form.
            values = list(
                super().set_initial_training_values(
                    args,
                    dataloader,
                    total_train_batch_size,
                )
            )
        if self.train_batch_sampler is None:
            return tuple(values)
        planned_sample_count = getattr(
            self.train_batch_sampler,
            "planned_sample_count",
            None,
        )
        optimizer_batch_samples = getattr(
            self.train_batch_sampler,
            "planned_optimizer_batch_samples",
            None,
        )
        if planned_sample_count is not None:
            values[2] = int(planned_sample_count)
            values[3] = int(planned_sample_count)
        if optimizer_batch_samples is not None and total_train_batch_size is None:
            # In Transformers 5.x index 4 is total_train_batch_size. In 4.57
            # index 4 is the epoch_based boolean; its startup batch-size value
            # remains an outer-loop nominal value and must not be overwritten.
            values[4] = int(optimizer_batch_samples)
        return tuple(values)

    def _inner_training_loop(self, *args: Any, **kwargs: Any) -> Any:
        resume_from_checkpoint = kwargs.get("resume_from_checkpoint")
        if resume_from_checkpoint is None and len(args) >= 3:
            resume_from_checkpoint = args[2]
        self._shaft_initial_global_step = self._checkpoint_global_step(
            resume_from_checkpoint
        )
        self._shaft_resume_final_metrics_corrected = False
        output = super()._inner_training_loop(*args, **kwargs)
        if (
            self._shaft_resume_final_metrics_corrected
            and hasattr(output, "_replace")
            and "train_loss" in output.metrics
        ):
            output = output._replace(training_loss=float(output.metrics["train_loss"]))
        return output

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self._is_dynamic_resume_final_metrics(logs):
            self._correct_dynamic_resume_final_metrics(logs)
            self._shaft_resume_final_metrics_corrected = True
        return super().log(logs, *args, **kwargs)

    def _is_dynamic_resume_final_metrics(self, logs: dict[str, float]) -> bool:
        return bool(
            not getattr(self, "_shaft_resume_final_metrics_corrected", False)
            and int(getattr(self, "_shaft_initial_global_step", 0)) > 0
            and getattr(
                self.train_batch_sampler,
                "planned_optimizer_step_sample_counts",
                None,
            )
            is not None
            and "train_runtime" in logs
            and "train_loss" in logs
        )

    def _correct_dynamic_resume_final_metrics(
        self,
        logs: dict[str, float],
    ) -> None:
        step_sample_counts = tuple(
            int(value)
            for value in self.train_batch_sampler.planned_optimizer_step_sample_counts
        )
        start_step = int(self._shaft_initial_global_step)
        stop_step = min(int(self.state.global_step), len(step_sample_counts))
        executed_steps = stop_step - start_step
        runtime = float(logs.get("train_runtime", 0.0))
        if executed_steps <= 0 or runtime <= 0:
            return
        executed_samples = sum(step_sample_counts[start_step:stop_step])
        logs["train_steps_per_second"] = round(executed_steps / runtime, 3)
        logs["train_samples_per_second"] = round(executed_samples / runtime, 3)
        logs["train_loss"] = float(self._total_loss_scalar) / executed_steps

    @staticmethod
    def _checkpoint_global_step(resume_from_checkpoint: Any) -> int:
        if not isinstance(resume_from_checkpoint, (str, Path)):
            return 0
        state_path = Path(resume_from_checkpoint) / "trainer_state.json"
        if not state_path.is_file():
            return 0
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            return max(int(payload.get("global_step", 0)), 0)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return 0
