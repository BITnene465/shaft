from __future__ import annotations

from functools import partial
import json
import logging
from pathlib import Path
from typing import Any

import torch
from accelerate.data_loader import BatchSamplerShard, DataLoaderDispatcher
from torch.utils.data import DataLoader, IterableDataset
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

from shaft.data.sampler import ShaftPlannedBatchSampler, ShaftSampleSampler


if is_datasets_available():
    import datasets


logger = logging.getLogger(__name__)


class _BatchSummaryFilter(logging.Filter):
    _PREFIXES = (
        "Num examples =",
        "Num Epochs =",
        "Num update steps per epoch =",
        "Instantaneous batch size per device =",
        "Total train batch size (w. parallel, distributed & accumulation) =",
        "Resuming training from checkpoint with epoch",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().strip()
        return not any(message.startswith(prefix) for prefix in self._PREFIXES)


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
            # committed planning state resumes and recreates persistent workers.
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
            # Planned batches can carry host-side structural metadata that a
            # model-family sequence policy must consume before tensor transfer
            # (for example Qwen3VL segment-local M-RoPE construction).  Keep
            # device placement owned by Trainer._prepare_inputs while still
            # letting Accelerate shard the global microbatch stream.
            prepared = self.accelerator.prepare_data_loader(
                dataloader,
                device_placement=False,
            )
        finally:
            # Custom grouped batching is train-only. Distributed eval must keep
            # equal per-rank step counts or its prediction collectives can deadlock.
            self.accelerator.even_batches = previous_even_batches
        batch_sampler = getattr(prepared, "batch_sampler", None)
        if not isinstance(self.train_batch_sampler, ShaftPlannedBatchSampler):
            return prepared
        world_size = int(self.train_batch_sampler.spec.data_world_size)
        if isinstance(prepared, DataLoaderDispatcher):
            raise RuntimeError(
                "Planned batches require dispatch_batches=False."
            )
        if world_size > 1 and not isinstance(batch_sampler, BatchSamplerShard):
            raise RuntimeError(
                "Distributed planned batches require Accelerate BatchSamplerShard."
            )
        if isinstance(batch_sampler, BatchSamplerShard):
            if bool(batch_sampler.even_batches) or bool(batch_sampler.split_batches):
                raise RuntimeError(
                    "Planned batches require Accelerate "
                    "even_batches=False and split_batches=False."
                )
            if int(batch_sampler.num_processes) != world_size:
                raise RuntimeError(
                    "Accelerate batch-shard world size differs from planning spec."
                )
        expected_local_batches = len(self.train_batch_sampler) // world_size
        if len(prepared) != expected_local_batches:
            raise RuntimeError(
                "Prepared planned DataLoader length differs from the remaining "
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
        if self._shaft_initial_global_step > 0:
            logger.info(
                "[train-resume] global_step=%s/%s planning_cursor=restored",
                self._shaft_initial_global_step,
                int(self.args.max_steps),
            )
        self._shaft_final_metrics_corrected = False
        log_filter: _BatchSummaryFilter | None = None
        if isinstance(self.train_batch_sampler, ShaftPlannedBatchSampler):
            spec = self.train_batch_sampler.spec
            local_min, local_max = spec.local_pack_count_bounds
            world_size = int(spec.data_world_size)
            accumulation = int(self.args.gradient_accumulation_steps)
            logger.info(
                "[train-batch] local_packs=%s..%s global_packs=%s..%s "
                "optimizer_packs=%s..%s per_device_train_batch_size=%s",
                local_min,
                local_max,
                local_min * world_size,
                local_max * world_size,
                local_min * world_size * accumulation,
                local_max * world_size * accumulation,
                local_max,
            )
            log_filter = _BatchSummaryFilter()
            logging.getLogger("transformers.trainer").addFilter(log_filter)
        try:
            output = super()._inner_training_loop(*args, **kwargs)
        finally:
            if log_filter is not None:
                logging.getLogger("transformers.trainer").removeFilter(log_filter)
        if (
            self._shaft_final_metrics_corrected
            and hasattr(output, "_replace")
            and "train_loss" in output.metrics
        ):
            output = output._replace(training_loss=float(output.metrics["train_loss"]))
        return output

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        efficiency_monitor = getattr(self, "efficiency_monitor", None)
        if efficiency_monitor is not None:
            logs.update(
                efficiency_monitor.report_pending(device=self.args.device)
            )
        if self._is_planned_final_metrics(logs):
            self._correct_planned_final_metrics(logs)
            self._shaft_final_metrics_corrected = True
        return super().log(logs, *args, **kwargs)

    def _is_planned_final_metrics(self, logs: dict[str, float]) -> bool:
        return bool(
            not getattr(self, "_shaft_final_metrics_corrected", False)
            and isinstance(self.train_batch_sampler, ShaftPlannedBatchSampler)
            and "train_runtime" in logs
            and "train_loss" in logs
        )

    def _correct_planned_final_metrics(
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
