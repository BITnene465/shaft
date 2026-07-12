from __future__ import annotations

from functools import partial
import json
from pathlib import Path
from typing import Any

import torch
from accelerate.data_loader import BatchSamplerShard, DataLoaderDispatcher
from torch.utils.data import DataLoader, IterableDataset
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

from shaft.data.sampler import ShaftBoundedBatchSampler, ShaftSampleSampler


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
        if isinstance(train_sampler, ShaftSampleSampler) and (
            int(train_sampler.rank) != 0 or int(train_sampler.world_size) != 1
        ):
            raise ValueError(
                "HF Trainer requires an unsharded ShaftSampleSampler "
                "(rank=0, world_size=1); Accelerate owns the single distributed "
                "batch-sharding step."
            )
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
            # Worker base seeds must not advance the model/dropout RNG when a
            # committed bounded state resumes and recreates persistent workers.
            "generator": torch.Generator().manual_seed(
                int(self.args.data_seed or self.args.seed)
            ),
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
            prepared = self.accelerator.prepare(dataloader)
        finally:
            # Dynamic cardinality is train-only. Distributed eval must keep equal
            # per-rank step counts or its prediction collectives can deadlock.
            self.accelerator.even_batches = previous_even_batches
        batch_sampler = getattr(prepared, "batch_sampler", None)
        if not isinstance(self.train_batch_sampler, ShaftBoundedBatchSampler):
            return prepared
        world_size = int(self.train_batch_sampler.spec.data_world_size)
        if isinstance(prepared, DataLoaderDispatcher):
            raise RuntimeError(
                "Bounded variable batches require dispatch_batches=False."
            )
        if world_size > 1 and not isinstance(batch_sampler, BatchSamplerShard):
            raise RuntimeError(
                "Distributed bounded batches require Accelerate BatchSamplerShard."
            )
        if isinstance(batch_sampler, BatchSamplerShard):
            if bool(batch_sampler.even_batches) or bool(batch_sampler.split_batches):
                raise RuntimeError(
                    "Bounded variable batches require Accelerate "
                    "even_batches=False and split_batches=False."
                )
            if int(batch_sampler.num_processes) != world_size:
                raise RuntimeError(
                    "Accelerate batch-shard world size differs from bounded spec."
                )
        expected_local_batches = len(self.train_batch_sampler) // world_size
        if len(prepared) != expected_local_batches:
            raise RuntimeError(
                "Prepared bounded DataLoader length differs from the remaining "
                "global-microstep count."
            )
        return prepared

    def _inner_training_loop(self, *args: Any, **kwargs: Any) -> Any:
        resume_from_checkpoint = kwargs.get("resume_from_checkpoint")
        if resume_from_checkpoint is None and len(args) >= 3:
            resume_from_checkpoint = args[2]
        self._shaft_initial_global_step = self._checkpoint_global_step(
            resume_from_checkpoint
        )
        self._shaft_final_metrics_corrected = False
        output = super()._inner_training_loop(*args, **kwargs)
        if (
            self._shaft_final_metrics_corrected
            and hasattr(output, "_replace")
            and "train_loss" in output.metrics
        ):
            output = output._replace(training_loss=float(output.metrics["train_loss"]))
        return output

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self._is_bounded_final_metrics(logs):
            self._correct_bounded_final_metrics(logs)
            self._shaft_final_metrics_corrected = True
        return super().log(logs, *args, **kwargs)

    def _is_bounded_final_metrics(self, logs: dict[str, float]) -> bool:
        return bool(
            not getattr(self, "_shaft_final_metrics_corrected", False)
            and isinstance(self.train_batch_sampler, ShaftBoundedBatchSampler)
            and "train_runtime" in logs
            and "train_loss" in logs
        )

    def _correct_bounded_final_metrics(
        self,
        logs: dict[str, float],
    ) -> None:
        start_step = int(self._shaft_initial_global_step)
        executed_steps = int(self.state.global_step) - start_step
        runtime = float(logs.get("train_runtime", 0.0))
        if executed_steps <= 0 or runtime <= 0:
            return
        executed_samples = int(self.train_batch_sampler.executed_sample_count)
        logs["train_steps_per_second"] = round(executed_steps / runtime, 3)
        logs["train_samples_per_second"] = round(executed_samples / runtime, 3)
        if start_step > 0:
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
