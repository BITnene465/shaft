from __future__ import annotations

from contextlib import nullcontext
from typing import Any
import time

import torch
import torch.distributed as dist
from transformers import Trainer

from shaft.model import ShaftRolloutScoringPlan
from shaft.training.checkpointing import ShaftCheckpointCommitMixin
from shaft.training.optimizer_mixin import ShaftOptimizerMixin
from shaft.training.train_sampler_mixin import ShaftTrainSamplerMixin

from .loss import OPDObjectivePlan
from .execution import OPDExecutionRuntime
from .rollout import OPDRolloutRequest
from .teacher import OPDTeacherScoreRequest
from .telemetry import OPDTelemetryMonitor


class ShaftOPDTrainer(
    ShaftCheckpointCommitMixin,
    ShaftOptimizerMixin,
    ShaftTrainSamplerMixin,
    Trainer,
):
    """Fully on-policy direct-loss distillation with an independent teacher."""

    def __init__(
        self,
        *args: Any,
        execution_runtime: OPDExecutionRuntime,
        objective_plan: OPDObjectivePlan,
        telemetry_monitor: OPDTelemetryMonitor | None = None,
        **kwargs: Any,
    ) -> None:
        student_model = kwargs.get("model")
        if not isinstance(student_model, torch.nn.Module):
            raise TypeError("OPD trainer requires model as a torch module.")
        self.execution_runtime = execution_runtime
        self.execution_runtime.teacher_provider.validate_student_model(student_model)
        self.objective_plan = objective_plan
        self.objective = objective_plan.build()
        self.telemetry_monitor = telemetry_monitor
        super().__init__(*args, **kwargs)
        if self.telemetry_monitor is not None:
            self.telemetry_monitor.bind_component(self.execution_runtime.rollout_backend)
            self.telemetry_monitor.bind_component(self.execution_runtime.teacher_provider)
        self.execution_runtime.rollout_backend.prepare(
            model=self.model,
            accelerator=self.accelerator,
            processing_class=self.processing_class,
        )
        self.execution_runtime.teacher_provider.prepare(self.accelerator)
        self._opd_training_step_active = False
        self._opd_window_numerator: torch.Tensor | None = None
        self._opd_window_denominator: torch.Tensor | None = None

    def _score_plan(
        self,
        model_inputs: dict[str, Any],
        *,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ShaftRolloutScoringPlan:
        if self.model_adapter is None:
            raise RuntimeError("OPD trainer requires a model adapter for scoring inputs.")
        return self.model_adapter.build_rollout_scoring_plan(
            prompt_inputs=model_inputs,
            sequences=sequences,
            attention_mask=attention_mask,
        )

    @staticmethod
    def _scale_accumulated_gradients(
        model: torch.nn.Module,
        scale: torch.Tensor,
    ) -> None:
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            return
        with torch.no_grad():
            torch._foreach_mul_(gradients, scale)

    def _reset_opd_window(self) -> None:
        self._opd_window_numerator = None
        self._opd_window_denominator = None

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        if num_items_in_batch is not None:
            raise RuntimeError(
                "OPD owns completion-token normalization and must not receive an "
                "HF item-count denominator."
            )
        if self._opd_training_step_active:
            raise RuntimeError("Nested OPD training_step calls are not supported.")
        self._opd_training_step_active = True
        telemetry_started_at = (
            time.perf_counter() if self.telemetry_monitor is not None else None
        )
        try:
            result = super().training_step(
                model,
                inputs,
                num_items_in_batch=None,
            )
        except BaseException:
            self._reset_opd_window()
            if self.telemetry_monitor is not None:
                self.telemetry_monitor.discard_window()
            raise
        finally:
            self._opd_training_step_active = False

        if self.telemetry_monitor is not None and telemetry_started_at is not None:
            self.telemetry_monitor.finish_training_step(
                time.perf_counter() - telemetry_started_at
            )

        if not bool(self.accelerator.sync_gradients):
            return torch.zeros_like(result)
        if self._opd_window_numerator is None or self._opd_window_denominator is None:
            self._reset_opd_window()
            raise RuntimeError("OPD optimizer step ended without token-normalization state.")
        reporting_loss = (
            self._opd_window_numerator / self._opd_window_denominator
        ).to(device=result.device, dtype=result.dtype)
        self._reset_opd_window()
        return reporting_loss

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        if self._opd_training_step_active and num_items_in_batch is not None:
            raise RuntimeError("OPD training received two competing normalization contracts.")
        model_inputs = dict(inputs)
        sample_ids = [str(value) for value in model_inputs.pop("_shaft_sample_ids", [])]
        request_ids = tuple(
            str(value) for value in model_inputs.pop("_shaft_rollout_request_ids", [])
        )
        prompt_token_ids = tuple(
            tuple(int(token_id) for token_id in row)
            for row in model_inputs.pop("_shaft_rollout_prompt_ids", [])
        )
        generation_prompt_token_ids = tuple(
            tuple(int(token_id) for token_id in row)
            for row in model_inputs.pop(
                "_shaft_rollout_generation_prompt_ids",
                [],
            )
        )
        raw_images = model_inputs.pop("_shaft_rollout_images", None)
        telemetry_stats = model_inputs.pop("_shaft_opd_batch_stats", None)
        batch_size = int(model_inputs["input_ids"].shape[0])
        if len(sample_ids) != batch_size:
            sample_ids = [str(index) for index in range(batch_size)]
        if not request_ids:
            request_ids = tuple(sample_ids)
        if not prompt_token_ids:
            attention_mask = model_inputs["attention_mask"].to(dtype=torch.bool)
            prompt_token_ids = tuple(
                tuple(
                    int(value)
                    for value in model_inputs["input_ids"][row_index][
                        attention_mask[row_index]
                    ].tolist()
                )
                for row_index in range(batch_size)
            )
        if not generation_prompt_token_ids:
            generation_prompt_token_ids = prompt_token_ids
        ordered_images = (
            tuple(None for _ in range(batch_size))
            if raw_images is None
            else tuple(tuple(row) for row in raw_images)
        )
        if self.telemetry_monitor is not None and self._opd_training_step_active:
            if not isinstance(telemetry_stats, dict):
                raise ValueError("Enabled OPD telemetry requires collator batch statistics.")
            self.telemetry_monitor.stage_microbatch(telemetry_stats)
        rollout = self.execution_runtime.rollout_backend.generate(
            OPDRolloutRequest(
                model=model,
                model_inputs=model_inputs,
                generation_prompt_token_ids=generation_prompt_token_ids,
                prompt_token_ids=prompt_token_ids,
                ordered_images=ordered_images,
                sample_ids=tuple(sample_ids),
                request_ids=request_ids,
                model_version=int(self.state.global_step),
                accelerator=self.accelerator,
                processing_class=self.processing_class,
            )
        )
        score_plan = self._score_plan(
            model_inputs,
            sequences=rollout.sequences,
            attention_mask=rollout.attention_mask,
        )
        student_phase = (
            self.telemetry_monitor.phase("student_score")
            if self.telemetry_monitor is not None and self._opd_training_step_active
            else nullcontext()
        )
        with student_phase:
            student_outputs = model(**score_plan.model_inputs)
        score_completion_mask = score_plan.align_completion_mask(
            rollout.completion_mask,
            logits=student_outputs.logits,
        )
        causal_position_mask = score_completion_mask[:, 1:].to(
            device=student_outputs.logits.device,
            dtype=torch.bool,
        )
        flattened_student_logits = student_outputs.logits[:, :-1, :][causal_position_mask]
        teacher_phase = (
            self.telemetry_monitor.phase("teacher_score")
            if self.telemetry_monitor is not None and self._opd_training_step_active
            else nullcontext()
        )
        with teacher_phase:
            teacher_distribution = self.execution_runtime.teacher_provider.score(
                OPDTeacherScoreRequest(
                    model_inputs=score_plan.model_inputs,
                    causal_position_mask=causal_position_mask,
                    request_ids=request_ids,
                    objective_plan=self.objective_plan,
                )
            )
        objective_phase = (
            self.telemetry_monitor.phase("objective")
            if self.telemetry_monitor is not None and self._opd_training_step_active
            else nullcontext()
        )
        with objective_phase:
            components = self.objective.compute(
                flattened_student_logits,
                teacher_distribution,
            )
        if self.telemetry_monitor is not None and self._opd_training_step_active:
            self.telemetry_monitor.record_completion_tokens(
                int(causal_position_mask.sum().item())
            )
            self.telemetry_monitor.record_teacher_distribution(teacher_distribution)
        global_stats = torch.stack(
            (
                components.numerator.detach().to(dtype=torch.float32),
                components.denominator.detach().to(dtype=torch.float32),
            )
        )
        data_parallel_world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
            data_parallel_world_size = dist.get_world_size()
        global_numerator, global_denominator = global_stats.unbind()
        if bool(global_denominator.le(0).item()):
            raise RuntimeError("OPD global completion-token denominator must be positive.")

        if self._opd_training_step_active:
            previous_denominator = self._opd_window_denominator
            if previous_denominator is None:
                accumulated_denominator = global_denominator
                accumulated_numerator = global_numerator
            else:
                accumulated_denominator = previous_denominator + global_denominator
                assert self._opd_window_numerator is not None
                accumulated_numerator = self._opd_window_numerator + global_numerator
                self._scale_accumulated_gradients(
                    model,
                    previous_denominator / accumulated_denominator,
                )
            self._opd_window_denominator = accumulated_denominator
            self._opd_window_numerator = accumulated_numerator
            # HF Trainer divides every returned training loss by the current GA
            # width. Cancel that mechanical scaling here; the recurrence above
            # gives every microbatch numerator the one shared optimizer-window
            # completion-token denominator before clipping and optimizer.step().
            loss = (
                components.numerator
                * float(data_parallel_world_size)
                / accumulated_denominator
                * int(self.current_gradient_accumulation_steps)
            )
        else:
            loss = (
                components.numerator
                * float(data_parallel_world_size)
                / global_denominator
            )
        if return_outputs:
            return loss, student_outputs
        return loss

    def finalize_opd_telemetry(self) -> dict[str, float]:
        if self.telemetry_monitor is None:
            return {}
        return self.telemetry_monitor.finalize(final_global_step=int(self.state.global_step))
