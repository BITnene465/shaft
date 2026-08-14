from __future__ import annotations

import math
import logging
from pathlib import Path
import time
from typing import Any
import warnings

import torch
import transformers.trainer as hf_trainer_module
from transformers.debug_utils import DebugOption
from transformers import Trainer
from transformers.trainer_callback import ExportableState, TrainerCallback
from transformers.utils import is_torch_xla_available
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel

from shaft.config.algorithm import normalize_auxiliary_loss_weights
from shaft.config.training import EvalConfig
from shaft.model.parameters import model_parameter_count
from shaft.model.finetune import load_peft_checkpoint
from shaft.model.generation import export_model_cache
from shaft.model.resolution import ResolvedAdapterInit
from shaft.model.types import (
    ShaftEvalAuxiliaryStatistic,
    validate_auxiliary_weight_names,
)
from shaft.utils.contract_schema import (
    json_int,
    json_list,
    json_number,
    require_exact_keys,
    require_json_mapping,
)
from shaft.utils.distributed import (
    all_gather_objects,
    barrier_if_distributed,
    get_rank,
    get_world_size,
)
from .checkpointing import ShaftCheckpointCommitMixin, reject_model_only_checkpoint_resume
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


logger = logging.getLogger(__name__)


_SFT_REPORTING_STATE_VERSION = "shaft-sft-reporting-state-v1"


class ShaftSFTReportingStateCallback(TrainerCallback, ExportableState):
    """Persist the not-yet-logged HF loss and model-owned auxiliary window."""

    def __init__(self) -> None:
        self.snapshot: dict[str, Any] | None = None
        self._trainer: ShaftSFTTrainer | None = None

    def bind(self, trainer: ShaftSFTTrainer) -> None:
        self._trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        _ = args, state, kwargs
        if self._trainer is None:
            raise RuntimeError("SFT reporting-state callback is not bound to its trainer.")
        self._trainer._restore_pending_reporting_state()
        return control

    def state(self) -> dict[str, Any]:
        return {
            "args": {},
            "attributes": {"snapshot": self.snapshot},
        }


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
        auxiliary_loss_weights: dict[str, float] | None = None,
        online_eval_runner: ShaftOnlineEvalRunner | None = None,
        eval_config: EvalConfig | None = None,
        eval_data_collator: Any | None = None,
        efficiency_monitor: ShaftTrainingEfficiencyMonitor | None = None,
        resume_peft_artifact: ResolvedAdapterInit | None = None,
        **kwargs: Any,
    ) -> None:
        raw_auxiliary_loss_weights = (
            {} if auxiliary_loss_weights is None else auxiliary_loss_weights
        )
        normalized_auxiliary_loss_weights = normalize_auxiliary_loss_weights(
            raw_auxiliary_loss_weights
        )
        auxiliary_loss_names = validate_auxiliary_weight_names(
            kwargs.get("model_adapter"),
            normalized_auxiliary_loss_weights,
        )
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
        # HF derives FLOPs from model.num_parameters() on every microbatch.
        # Sharded backends can make that live view depend on the current
        # gather/partition state. Capture one HF-compatible logical count;
        # model_parameter_count reads ZeRO placeholders through ds_numel/ds_shape.
        self._shaft_floating_point_ops_parameter_count = model_parameter_count(
            self.model,
            exclude_embeddings=True,
        )
        self._shaft_floating_point_ops_main_input_name = str(
            getattr(self.model, "main_input_name", "input_ids")
        )
        if self.args.restore_callback_states_from_checkpoint:
            raise ValueError(
                "ShaftSFTTrainer does not support "
                "restore_callback_states_from_checkpoint=True because Shaft owns "
                "the exact-resume state of its reporting callback."
            )
        self.loss_name = str(loss_name).strip().lower()
        self.loss_fn = build_loss(self.loss_name)
        self.ignore_index = int(ignore_index)
        self.auxiliary_loss_weights = normalized_auxiliary_loss_weights
        self._shaft_auxiliary_loss_names = frozenset(auxiliary_loss_names)
        self.online_eval_runner = online_eval_runner
        self.eval_config = eval_config
        self.efficiency_monitor = efficiency_monitor
        self._shaft_train_auxiliary_loss_accumulators: dict[
            str,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._shaft_eval_auxiliary_statistic_accumulators: dict[
            str,
            tuple[str, float, dict[str, torch.Tensor]],
        ] = {}
        self._shaft_eval_primary_loss_accumulator: tuple[
            torch.Tensor,
            torch.Tensor,
        ] | None = None
        # HF releases have represented the current loss window as either loop-
        # local state or a Trainer attribute. The logging/save boundary passes it
        # explicitly, so checkpointing reads that bound tensor instead of either
        # private storage shape.
        self._shaft_reporting_tr_loss: torch.Tensor | None = None
        self._shaft_pending_reporting_loss: float | None = None
        self._shaft_pending_total_provisional: float | None = None
        self._shaft_resume_loss_baseline = 0.0
        self._shaft_reporting_state_callback = ShaftSFTReportingStateCallback()
        self._shaft_reporting_state_callback.bind(self)
        self.callback_handler.callbacks.insert(
            0,
            self._shaft_reporting_state_callback,
        )
        self._shaft_fp16_grad_overflow_count = 0
        self._configure_eval_data_collator(eval_data_collator)
        # HF uses this flag to collect one optimizer batch before backward and pass
        # its global normalization denominator into compute_loss.
        self.model_accepts_loss_kwargs = True
        self._shaft_preloaded_fsdp_peft_checkpoint: Path | None = None
        self._shaft_resume_peft_artifact = resume_peft_artifact
        self._shaft_uses_peft = isinstance(self.model, PeftModel)

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
        trial: Any | None = None,
        ignore_keys_for_eval: list[str] | None = None,
    ):
        resolved_checkpoint: Path | None = None
        upstream_resume_from_checkpoint = resume_from_checkpoint
        if resume_from_checkpoint is True:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
            if last_checkpoint is not None:
                resolved_checkpoint = Path(last_checkpoint).resolve()
                upstream_resume_from_checkpoint = str(resolved_checkpoint)
        elif resume_from_checkpoint not in (None, False):
            resolved_checkpoint = Path(str(resume_from_checkpoint)).resolve()

        if resolved_checkpoint is not None:
            reject_model_only_checkpoint_resume(resolved_checkpoint)

        if (
            resolved_checkpoint is not None
            and self.is_fsdp_enabled
            and isinstance(self.model, PeftModel)
        ):
            if self.model_init is not None:
                raise ValueError(
                    "FSDP PEFT resume does not support Trainer model_init because "
                    "the model would be replaced after the complete adapter preload."
                )
            local_error: Exception | None = None
            loaded_summary: tuple[int, int] | None = None
            try:
                if (
                    self._shaft_resume_peft_artifact is not None
                    and Path(self._shaft_resume_peft_artifact.path).resolve()
                    != resolved_checkpoint
                ):
                    raise ValueError(
                        "Resolved PEFT resume artifact differs from Trainer checkpoint."
                    )
                loaded_summary = load_peft_checkpoint(
                    self.model,
                    resolved_checkpoint,
                    resolved_artifact=self._shaft_resume_peft_artifact,
                )
            except Exception as exc:  # noqa: BLE001 - converge before FSDP collectives
                local_error = exc
            self._raise_synchronized_checkpoint_error(
                "FSDP PEFT model preload",
                local_error,
            )
            assert loaded_summary is not None
            self._shaft_preloaded_fsdp_peft_checkpoint = resolved_checkpoint
            logger.info(
                "[resume] preloaded complete PEFT state before FSDP wrapping: "
                "checkpoint=%s tensors=%d parameters=%d",
                resolved_checkpoint,
                loaded_summary[0],
                loaded_summary[1],
            )

        return super().train(
            resume_from_checkpoint=upstream_resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

    def _load_from_checkpoint(
        self,
        resume_from_checkpoint: str,
        model: torch.nn.Module | None = None,
    ) -> None:
        preloaded = self._shaft_preloaded_fsdp_peft_checkpoint
        if preloaded is not None and Path(resume_from_checkpoint).resolve() == preloaded:
            logger.info(
                "[resume] skipped incomplete upstream FSDP adapter-only model state; "
                "the complete standard PEFT state was loaded before FSDP wrapping"
            )
            return
        super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _load_best_model(self) -> None:
        if self.is_fsdp_enabled and self._shaft_uses_peft:
            raise RuntimeError(
                "FSDP PEFT load_best_model_at_end is disabled because the current "
                "Transformers/Accelerate adapter-only loader restores an incomplete "
                "local DTensor shard. Set train.load_best_model_at_end=false."
            )
        super()._load_best_model()

    def _auxiliary_coefficient(self, key: str, default: float) -> float:
        return self.auxiliary_loss_weights.get(key, float(default))

    def _inner_training_loop(self, *args: Any, **kwargs: Any) -> Any:
        if self.args.gradient_checkpointing:
            active = any(
                bool(getattr(module, "is_gradient_checkpointing", False))
                for module in self.model.modules()
            )
            if not active:
                raise RuntimeError(
                    "train.gradient_checkpointing=true, but the prepared model "
                    "does not report active model-side gradient checkpointing."
                )
        return super()._inner_training_loop(*args, **kwargs)

    def floating_point_ops(self, inputs: dict[str, torch.Tensor | Any]) -> int:
        parameter_count = self._shaft_floating_point_ops_parameter_count
        main_input_name = self._shaft_floating_point_ops_main_input_name
        if parameter_count is not None and main_input_name in inputs:
            return 6 * inputs[main_input_name].numel() * parameter_count
        return super().floating_point_ops(inputs)

    @staticmethod
    def _normalize_reporting_rank_state(
        value: Any,
        *,
        expected_rank: int,
        expected_global_step: int,
    ) -> dict[str, Any]:
        role = f"SFT reporting state rank {expected_rank}"
        payload = require_json_mapping(value, role=role)
        require_exact_keys(
            payload,
            expected=frozenset(
                {
                    "rank",
                    "global_step",
                    "tr_loss",
                    "total_loss_scalar",
                    "globalstep_last_logged",
                    "auxiliary",
                }
            ),
            role=role,
        )
        rank = json_int(payload, "rank", role=role)
        global_step = json_int(payload, "global_step", role=role)
        globalstep_last_logged = json_int(
            payload,
            "globalstep_last_logged",
            role=role,
        )
        if rank != expected_rank:
            raise ValueError(
                f"{role}.rank differs from its rank-list position: {rank}."
            )
        if global_step != expected_global_step:
            raise ValueError(
                f"{role}.global_step differs from checkpoint trainer state."
            )
        if not 0 <= globalstep_last_logged <= global_step:
            raise ValueError(
                f"{role}.globalstep_last_logged is outside [0, global_step]."
            )
        auxiliary_payload = require_json_mapping(
            payload["auxiliary"],
            role=f"{role}.auxiliary",
        )
        auxiliary: dict[str, dict[str, float | int]] = {}
        for name, raw_term in sorted(auxiliary_payload.items()):
            if not name.strip() or name != name.strip():
                raise ValueError(
                    f"{role}.auxiliary names must be non-empty canonical strings."
                )
            term_role = f"{role}.auxiliary.{name}"
            term = require_json_mapping(raw_term, role=term_role)
            require_exact_keys(
                term,
                expected=frozenset({"raw_sum", "weighted_sum", "count"}),
                role=term_role,
            )
            count = json_int(term, "count", role=term_role)
            if count < 0:
                raise ValueError(f"{term_role}.count must be >= 0.")
            auxiliary[name] = {
                "raw_sum": json_number(term["raw_sum"], role=f"{term_role}.raw_sum"),
                "weighted_sum": json_number(
                    term["weighted_sum"],
                    role=f"{term_role}.weighted_sum",
                ),
                "count": count,
            }
        return {
            "rank": rank,
            "global_step": global_step,
            "tr_loss": json_number(payload["tr_loss"], role=f"{role}.tr_loss"),
            "total_loss_scalar": json_number(
                payload["total_loss_scalar"],
                role=f"{role}.total_loss_scalar",
            ),
            "globalstep_last_logged": globalstep_last_logged,
            "auxiliary": auxiliary,
        }

    @classmethod
    def _normalize_reporting_snapshot(
        cls,
        value: Any,
        *,
        expected_global_step: int,
        expected_world_size: int,
    ) -> dict[str, Any]:
        role = "SFT checkpoint reporting snapshot"
        payload = require_json_mapping(value, role=role)
        require_exact_keys(
            payload,
            expected=frozenset({"version", "global_step", "world_size", "ranks"}),
            role=role,
        )
        if payload["version"] != _SFT_REPORTING_STATE_VERSION:
            raise ValueError(f"Unsupported {role} version: {payload['version']!r}.")
        global_step = json_int(payload, "global_step", role=role)
        world_size = json_int(payload, "world_size", role=role)
        if global_step != expected_global_step:
            raise ValueError(f"{role}.global_step differs from trainer state.")
        if world_size != expected_world_size:
            raise ValueError(
                f"{role}.world_size differs from the active data-parallel world."
            )
        raw_ranks = json_list(payload, "ranks", role=role)
        if len(raw_ranks) != world_size:
            raise ValueError(f"{role}.ranks does not contain every rank.")
        ranks = [
            cls._normalize_reporting_rank_state(
                raw_rank,
                expected_rank=rank,
                expected_global_step=global_step,
            )
            for rank, raw_rank in enumerate(raw_ranks)
        ]
        auxiliary_name_sets = {
            tuple(rank_payload["auxiliary"])
            for rank_payload in ranks
        }
        if len(auxiliary_name_sets) > 1:
            raise ValueError(
                f"{role} auxiliary term names differ across ranks: "
                f"{sorted(auxiliary_name_sets)!r}."
            )
        return {
            "version": _SFT_REPORTING_STATE_VERSION,
            "global_step": global_step,
            "world_size": world_size,
            "ranks": ranks,
        }

    def _local_reporting_rank_state(self) -> dict[str, Any]:
        tr_loss = self._shaft_reporting_tr_loss
        if not torch.is_tensor(tr_loss) or tr_loss.numel() != 1:
            raise RuntimeError(
                "SFT checkpoint reporting state requires the scalar loss window "
                "passed through HF's logging/save boundary."
            )
        auxiliary: dict[str, dict[str, float | int]] = {}
        for name, values in sorted(
            self._shaft_train_auxiliary_loss_accumulators.items()
        ):
            raw_sum, weighted_sum, count = values
            if any(not torch.is_tensor(value) or value.numel() != 1 for value in values):
                raise RuntimeError(
                    f"SFT auxiliary accumulator {name!r} must contain scalar tensors."
                )
            raw_value = float(raw_sum.item())
            weighted_value = float(weighted_sum.item())
            count_value = float(count.item())
            if not all(
                math.isfinite(value)
                for value in (raw_value, weighted_value, count_value)
            ):
                raise ValueError(
                    f"SFT auxiliary accumulator {name!r} contains non-finite state."
                )
            if count_value < 0.0 or not count_value.is_integer():
                raise ValueError(
                    f"SFT auxiliary accumulator {name!r} count must be a non-negative integer."
                )
            auxiliary[name] = {
                "raw_sum": raw_value,
                "weighted_sum": weighted_value,
                "count": int(count_value),
            }
        payload = {
            "rank": get_rank(),
            "global_step": int(self.state.global_step),
            "tr_loss": float(tr_loss.item()),
            "total_loss_scalar": float(self._total_loss_scalar),
            "globalstep_last_logged": int(self._globalstep_last_logged),
            "auxiliary": auxiliary,
        }
        return self._normalize_reporting_rank_state(
            payload,
            expected_rank=get_rank(),
            expected_global_step=int(self.state.global_step),
        )

    def _build_reporting_snapshot(self) -> dict[str, Any]:
        local_error: Exception | None = None
        local_payload: dict[str, Any] | None = None
        try:
            local_payload = self._local_reporting_rank_state()
        except Exception as exc:  # noqa: BLE001 - peers must enter the same gather
            local_error = exc
        statuses = all_gather_objects(
            {
                "ok": local_error is None,
                "error_type": None if local_error is None else type(local_error).__name__,
                "error": None if local_error is None else str(local_error),
                "payload": local_payload,
            }
        )
        failures = [status for status in statuses if status.get("ok") is not True]
        if failures:
            raise RuntimeError(
                "Distributed SFT reporting-state snapshot failed: "
                f"{failures!r}."
            )
        snapshot = {
            "version": _SFT_REPORTING_STATE_VERSION,
            "global_step": int(self.state.global_step),
            "world_size": get_world_size(),
            "ranks": [status["payload"] for status in statuses],
        }
        return self._normalize_reporting_snapshot(
            snapshot,
            expected_global_step=int(self.state.global_step),
            expected_world_size=get_world_size(),
        )

    def _load_reporting_snapshot_from_trainer_state(self) -> dict[str, Any]:
        callback_name = ShaftSFTReportingStateCallback.__name__
        stored = self.state.stateful_callbacks.get(callback_name)
        if isinstance(stored, list):
            raise ValueError(
                "SFT checkpoint reporting-state callback must use HF's canonical "
                "single-callback object format, not a list."
            )
        role = f"trainer_state.stateful_callbacks.{callback_name}"
        callback_state = require_json_mapping(stored, role=role)
        require_exact_keys(
            callback_state,
            expected=frozenset({"args", "attributes"}),
            role=role,
        )
        args = require_json_mapping(callback_state["args"], role=f"{role}.args")
        if args:
            raise ValueError(f"{role}.args must be empty.")
        attributes = require_json_mapping(
            callback_state["attributes"],
            role=f"{role}.attributes",
        )
        require_exact_keys(
            attributes,
            expected=frozenset({"snapshot"}),
            role=f"{role}.attributes",
        )
        return self._normalize_reporting_snapshot(
            attributes["snapshot"],
            expected_global_step=int(self.state.global_step),
            expected_world_size=get_world_size(),
        )

    def _restore_pending_reporting_state(self) -> None:
        self._shaft_reporting_tr_loss = None
        self._shaft_pending_reporting_loss = None
        self._shaft_pending_total_provisional = None
        self._shaft_resume_loss_baseline = 0.0
        self._shaft_train_auxiliary_loss_accumulators.clear()
        if int(self.state.global_step) == 0:
            self._shaft_reporting_state_callback.snapshot = None
            return
        snapshot = self._load_reporting_snapshot_from_trainer_state()
        self._shaft_reporting_state_callback.snapshot = snapshot
        rank_payload = snapshot["ranks"][get_rank()]
        pending_loss = float(rank_payload["tr_loss"])
        total_loss_scalar = float(rank_payload["total_loss_scalar"])
        self._shaft_resume_loss_baseline = sum(
            float(item["total_loss_scalar"]) + float(item["tr_loss"])
            for item in snapshot["ranks"]
        ) / int(snapshot["world_size"])
        # Preserve cumulative/global-step final loss for a no-op resume. Once a
        # real step runs, training_step removes the provisional pending term as
        # it moves that value back into HF's device-side reporting window.
        self._total_loss_scalar = total_loss_scalar + pending_loss
        self._shaft_pending_total_provisional = pending_loss
        self._globalstep_last_logged = int(
            rank_payload["globalstep_last_logged"]
        )
        self._shaft_pending_reporting_loss = pending_loss
        for name, term in rank_payload["auxiliary"].items():
            self._shaft_train_auxiliary_loss_accumulators[name] = tuple(
                torch.tensor(
                    float(term[field]),
                    dtype=torch.float32,
                    device=self.args.device,
                )
                for field in ("raw_sum", "weighted_sum", "count")
            )

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        if not torch.is_tensor(tr_loss) or tr_loss.numel() != 1:
            raise RuntimeError("HF reporting loss window must be a scalar tensor.")
        pending_loss = self._shaft_pending_reporting_loss
        if pending_loss is not None:
            provisional = self._shaft_pending_total_provisional
            if provisional is not None:
                self._total_loss_scalar -= provisional
                self._shaft_pending_total_provisional = None
            tr_loss.add_(tr_loss.new_tensor(pending_loss))
            self._shaft_pending_reporting_loss = None
        self._shaft_reporting_tr_loss = tr_loss
        return super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate,
        )

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
        self._record_executed_local_samples(batch_samples)
        return batch_samples, num_items

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        efficiency_enabled = bool(
            self.efficiency_monitor is not None
            and self.efficiency_monitor.enabled
        )
        started_at = time.perf_counter() if efficiency_enabled else 0.0
        if efficiency_enabled and self.efficiency_monitor.device_timing:
            if self.args.device.type == "cuda":
                self.efficiency_monitor.start_device_frame()
        result = super().training_step(model, inputs, num_items_in_batch)
        if not torch.is_tensor(result) or result.numel() != 1:
            raise RuntimeError("HF training_step must return a scalar loss tensor.")
        pending_loss = self._shaft_pending_reporting_loss
        if pending_loss is not None:
            provisional = self._shaft_pending_total_provisional
            if provisional is None:
                raise RuntimeError("SFT pending reporting loss has no total-loss baseline.")
            self._total_loss_scalar -= provisional
            self._shaft_pending_total_provisional = None
            execution_loss = result.detach().to(dtype=torch.float32)
            if (
                self.args.logging_nan_inf_filter
                and not is_torch_xla_available()
                and not bool(torch.isfinite(execution_loss).item())
            ):
                denominator = (
                    1 + int(self.state.global_step) - int(self._globalstep_last_logged)
                )
                pending = execution_loss.new_tensor(pending_loss)
                result = pending + pending / denominator
                self._shaft_pending_reporting_loss = None
            else:
                # Backward has already consumed the real current-step loss.
                # Returning pending + current only restores HF's reporting tensor,
                # before later GA microsteps preserve the uninterrupted FP32 order.
                result = execution_loss.new_tensor(pending_loss) + execution_loss
                self._shaft_pending_reporting_loss = None
        if efficiency_enabled:
            assert self.efficiency_monitor is not None
            self.efficiency_monitor.record_training_step(
                time.perf_counter() - started_at,
            )
        return result

    def _shaft_resume_train_loss_numerator(self) -> float:
        local = torch.tensor(
            float(self._total_loss_scalar) - self._shaft_resume_loss_baseline,
            dtype=torch.float64,
            device=self.args.device,
        ).reshape(1)
        gathered = self.accelerator.gather(local)
        return float(gathered.mean().item())

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
        shaft_owns_causal_objective = self.loss_name in {"auto", "causal_lm"}
        if shaft_owns_causal_objective:
            # Shaft's built-in SFT losses own shifted CE and its GA/DP denominator.
            # Forwarding labels would make HF compute a second full-vocabulary CE.
            model_inputs.pop("labels", None)
        if self.model_adapter is not None:
            model_inputs = self.model_adapter.prepare_sft_forward_inputs(
                model=model,
                inputs=model_inputs,
            )
        outputs = model(**model_inputs)
        is_training = bool(model.training)
        primary_loss_components: dict[str, torch.Tensor] | None = (
            {}
            if not is_training and self.args.average_tokens_across_devices
            else None
        )
        primary_loss = self.loss_fn(
            outputs=outputs,
            labels=labels,
            ignore_index=self.ignore_index,
            loss_scale=loss_scale,
            model=model,
            inputs=model_inputs,
            normalization_denominator=num_items_in_batch,
            component_output=primary_loss_components,
        )
        if num_items_in_batch is not None and self.args.average_tokens_across_devices:
            data_parallel_scale = self.accelerator.num_processes
            parallelism_config = getattr(self.accelerator, "parallelism_config", None)
            if parallelism_config is not None:
                data_parallel_scale //= parallelism_config.tp_size
            primary_loss_scale = (
                data_parallel_scale if self.args.n_gpu <= 1 else self.args.n_gpu
            )
            primary_loss = primary_loss * primary_loss_scale
        if primary_loss_components:
            numerator, denominator = self.accelerator.gather_for_metrics(
                (
                    primary_loss_components["numerator"].detach().to(
                        dtype=torch.float32
                    ),
                    primary_loss_components["denominator"].detach().to(
                        dtype=torch.float32
                    ),
                )
            )
            numerator = numerator.sum()
            denominator = denominator.sum()
            previous_primary = self._shaft_eval_primary_loss_accumulator
            if previous_primary is None:
                self._shaft_eval_primary_loss_accumulator = (
                    numerator,
                    denominator,
                )
            else:
                self._shaft_eval_primary_loss_accumulator = (
                    previous_primary[0] + numerator,
                    previous_primary[1] + denominator,
                )
        auxiliary_loss = primary_loss.new_zeros(())
        if shaft_owns_causal_objective and self.model_adapter is not None:
            if not is_training:
                self._record_eval_auxiliary_statistics(
                    self.model_adapter.resolve_sft_eval_auxiliary_statistics(
                        model=model,
                        outputs=outputs,
                        inputs=model_inputs,
                    )
                )
            else:
                terms = self.model_adapter.resolve_sft_auxiliary_loss_terms(
                    model=model,
                    outputs=outputs,
                    inputs=model_inputs,
                )
                accumulation_steps = max(
                    1,
                    int(
                        getattr(
                            self,
                            "current_gradient_accumulation_steps",
                            self.args.gradient_accumulation_steps,
                        )
                    ),
                )
                seen_term_names: set[str] = set()
                for term in terms:
                    if term.name in seen_term_names:
                        raise ValueError(
                            f"Model policy emitted duplicate auxiliary loss term {term.name!r}."
                        )
                    seen_term_names.add(term.name)
                    if term.name not in self._shaft_auxiliary_loss_names:
                        raise ValueError(
                            "Model policy emitted undeclared auxiliary loss term "
                            f"{term.name!r}; declare it in auxiliary_loss_names()."
                        )
                    coefficient = self._auxiliary_coefficient(
                        term.name,
                        term.coefficient,
                    )
                    value = term.value.to(
                        device=primary_loss.device,
                        dtype=primary_loss.dtype,
                    )
                    auxiliary_loss = auxiliary_loss + (
                        value * coefficient / accumulation_steps
                    )
                    detached = value.detach().to(dtype=torch.float32)
                    weighted = detached * coefficient
                    count = detached.new_ones(())
                    previous = self._shaft_train_auxiliary_loss_accumulators.get(
                        term.name
                    )
                    if previous is None:
                        self._shaft_train_auxiliary_loss_accumulators[term.name] = (
                            detached,
                            weighted,
                            count,
                        )
                    else:
                        self._shaft_train_auxiliary_loss_accumulators[term.name] = (
                            previous[0] + detached,
                            previous[1] + weighted,
                            previous[2] + count,
                        )
        loss = primary_loss + auxiliary_loss
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        grad_norm = logs.get("grad_norm")
        if (
            bool(getattr(self.args, "fp16", False))
            and grad_norm is not None
            and not math.isfinite(float(grad_norm))
        ):
            logs.pop("grad_norm")
            logs["grad_norm_overflow"] = 1.0
            self._shaft_fp16_grad_overflow_count += 1
            if (
                self._shaft_fp16_grad_overflow_count == 1
                and self.is_world_process_zero()
            ):
                logger.warning(
                    "[fp16] non-finite gradient norm was skipped by GradScaler; "
                    "recording grad_norm_overflow=1 instead of persisting NaN."
                )
        is_eval_log = any(
            str(key).startswith(("eval_", "test_")) for key in logs
        )
        if not is_eval_log:
            auxiliary_metrics, _ = self._consume_auxiliary_loss_accumulators(
                self._shaft_train_auxiliary_loss_accumulators,
                metric_prefix="aux",
            )
            logs.update(auxiliary_metrics)
        return super().log(logs, *args, **kwargs)

    def _consume_auxiliary_loss_accumulators(
        self,
        accumulators: dict[
            str,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        *,
        metric_prefix: str,
    ) -> tuple[dict[str, float], float]:
        metrics: dict[str, float] = {}
        total_weighted_mean = 0.0
        local_names = tuple(sorted(accumulators))
        peer_names = all_gather_objects(local_names)
        if any(names != peer_names[0] for names in peer_names[1:]):
            raise RuntimeError(
                "SFT auxiliary loss term names differ across distributed ranks: "
                f"{peer_names!r}."
            )
        for name in local_names:
            raw_sum, weighted_sum, count = accumulators[name]
            payload = torch.stack((raw_sum, weighted_sum, count))
            gathered = self.accelerator.gather(payload).reshape(-1, 3).sum(dim=0)
            total_count = float(gathered[2].item())
            if total_count > 0.0:
                metrics[f"{metric_prefix}/{name}"] = (
                    float(gathered[0].item()) / total_count
                )
                weighted_mean = (
                    float(gathered[1].item()) / total_count
                )
                metrics[f"{metric_prefix}/{name}_weighted"] = weighted_mean
                total_weighted_mean += weighted_mean
        accumulators.clear()
        return metrics, total_weighted_mean

    def _begin_eval_objective_collection(self) -> None:
        self._shaft_eval_auxiliary_statistic_accumulators.clear()
        self._shaft_eval_primary_loss_accumulator = None

    def _record_eval_auxiliary_statistics(
        self,
        statistics: tuple[ShaftEvalAuxiliaryStatistic, ...],
    ) -> None:
        seen_statistic_names: set[str] = set()
        for statistic in statistics:
            if statistic.name in seen_statistic_names:
                raise ValueError(
                    f"Model policy emitted duplicate eval auxiliary statistic {statistic.name!r}."
                )
            seen_statistic_names.add(statistic.name)
            coefficient_key = str(statistic.coefficient_key)
            if coefficient_key not in self._shaft_auxiliary_loss_names:
                raise ValueError(
                    "Model policy emitted an eval auxiliary statistic with undeclared "
                    f"coefficient_key {coefficient_key!r}."
                )
            coefficient = float(statistic.coefficient)
            component_names = tuple(sorted(statistic.components))
            gathered_values = self.accelerator.gather_for_metrics(
                tuple(
                    statistic.components[name].detach().to(dtype=torch.float32)
                    for name in component_names
                )
            )
            component_sums = {
                name: value.sum(dim=0, keepdim=True)
                for name, value in zip(
                    component_names,
                    gathered_values,
                    strict=True,
                )
            }
            previous = self._shaft_eval_auxiliary_statistic_accumulators.get(
                statistic.name
            )
            if previous is None:
                self._shaft_eval_auxiliary_statistic_accumulators[statistic.name] = (
                    coefficient_key,
                    coefficient,
                    component_sums,
                )
                continue
            (
                previous_coefficient_key,
                previous_coefficient,
                previous_components,
            ) = previous
            if previous_coefficient_key != coefficient_key:
                raise ValueError(
                    f"Eval auxiliary statistic {statistic.name!r} changed "
                    "coefficient_key within one evaluation pass."
                )
            if previous_coefficient != coefficient:
                raise ValueError(
                    f"Eval auxiliary statistic {statistic.name!r} changed coefficient "
                    "within one evaluation pass."
                )
            if set(previous_components) != set(component_sums):
                raise ValueError(
                    f"Eval auxiliary statistic {statistic.name!r} changed components "
                    "within one evaluation pass."
                )
            self._shaft_eval_auxiliary_statistic_accumulators[statistic.name] = (
                previous_coefficient_key,
                previous_coefficient,
                {
                    name: previous_components[name] + component_sums[name]
                    for name in previous_components
                },
            )

    def _finish_eval_objective_collection(
        self,
        *,
        metric_key_prefix: str,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self._shaft_eval_auxiliary_statistic_accumulators:
            if self.model_adapter is None:
                raise ValueError(
                    "Eval auxiliary statistics require a resolved model adapter."
                )
            aggregated_statistics = tuple(
                ShaftEvalAuxiliaryStatistic(
                    name=name,
                    coefficient=coefficient,
                    coefficient_key=coefficient_key,
                    components=components,
                )
                for name, (coefficient_key, coefficient, components) in sorted(
                    self._shaft_eval_auxiliary_statistic_accumulators.items()
                )
            )
            finalized = self.model_adapter.finalize_sft_eval_auxiliary_statistics(
                aggregated_statistics
            )
            seen_metric_names: set[str] = set()
            for metric in finalized:
                if metric.name in seen_metric_names:
                    raise ValueError(
                        f"Model policy finalized duplicate eval auxiliary metric {metric.name!r}."
                    )
                seen_metric_names.add(metric.name)
                if metric.coefficient_key not in self._shaft_auxiliary_loss_names:
                    raise ValueError(
                        "Model policy finalized an eval auxiliary metric with undeclared "
                        f"coefficient_key {metric.coefficient_key!r}."
                    )
                value = float(metric.value.detach().to(dtype=torch.float32).item())
                metrics[f"{metric_key_prefix}_aux/{metric.name}"] = value
                effective_coefficient = self._auxiliary_coefficient(
                    metric.coefficient_key,
                    metric.coefficient,
                )
                metrics[f"{metric_key_prefix}_aux/{metric.name}_weighted"] = (
                    value * effective_coefficient
                )
        self._shaft_eval_auxiliary_statistic_accumulators.clear()
        primary = self._shaft_eval_primary_loss_accumulator
        self._shaft_eval_primary_loss_accumulator = None
        if primary is None:
            return metrics
        denominator = float(primary[1].item())
        if denominator > 0.0:
            metrics[f"{metric_key_prefix}_loss"] = float(primary[0].item()) / denominator
        return metrics

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
            report_metrics.update(
                {
                    key: float(value)
                    for key, value in merged_metrics.items()
                    if key.startswith(f"{metric_key_prefix}_aux/")
                }
            )
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
        self._begin_eval_objective_collection()
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        except Exception:
            self._begin_eval_objective_collection()
            raise
        output.metrics.update(
            self._finish_eval_objective_collection(
                metric_key_prefix=metric_key_prefix,
            )
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
            report_metrics.update(
                {
                    key: float(value)
                    for key, value in dataset_metrics.items()
                    if key.startswith(f"{metric_key_prefix}_{dataset_name}_aux/")
                }
            )
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

    def save_state(self) -> None:
        # The run-root state is an observability summary, not a resumable
        # checkpoint. Do not serialize the last checkpoint's stale pending
        # reporting window under a later terminal global_step.
        callback_name = ShaftSFTReportingStateCallback.__name__
        self._shaft_reporting_state_callback.snapshot = None
        self.state.stateful_callbacks.pop(callback_name, None)
        return super().save_state()

    def _prepare_shaft_checkpoint_save(self, checkpoint_path: Path) -> None:
        self._shaft_reporting_state_callback.snapshot = (
            self._build_reporting_snapshot()
        )
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
            with export_model_cache(self.model):
                super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
            local_error = exc
        self._raise_synchronized_checkpoint_error("model save", local_error)
