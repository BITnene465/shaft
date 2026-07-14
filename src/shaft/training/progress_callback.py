from __future__ import annotations

import math
from typing import Any

import torch
from transformers import TrainerCallback

from shaft.observability import ShaftProgressManager, ShaftProgressTask


class ShaftProgressCallback(TrainerCallback):
    """Adapt Hugging Face Trainer events into Shaft progress tasks."""

    def __init__(self, progress_manager: ShaftProgressManager) -> None:
        self.progress_manager = progress_manager
        self.training_task: ShaftProgressTask | None = None
        self.prediction_task: ShaftProgressTask | None = None

    @staticmethod
    def _format_learning_rate_range(values: tuple[float, ...]) -> float | str | None:
        if not values or any(
            not math.isfinite(value) or value < 0 for value in values
        ):
            return None
        finite = values
        lower = min(finite)
        upper = max(finite)
        if math.isclose(lower, upper, rel_tol=1e-12, abs_tol=0.0):
            return lower

        def _parts(value: float) -> tuple[str, int]:
            mantissa, exponent = f"{value:.3e}".split("e", maxsplit=1)
            return f"{float(mantissa):g}", int(exponent)

        lower_mantissa, lower_exponent = _parts(lower)
        upper_mantissa, upper_exponent = _parts(upper)
        if lower_exponent == upper_exponent:
            return (
                f"{lower_mantissa}–{upper_mantissa}e{lower_exponent}"
            )
        return (
            f"{lower_mantissa}e{lower_exponent}–"
            f"{upper_mantissa}e{upper_exponent}"
        )

    def _resolve_learning_rate(
        self,
        *,
        optimizer: Any = None,
        lr_scheduler: Any = None,
    ) -> float | str | None:
        if lr_scheduler is not None and not isinstance(
            lr_scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            try:
                learning_rates = tuple(
                    float(value.item() if isinstance(value, torch.Tensor) else value)
                    for value in lr_scheduler.get_last_lr()
                )
            except (
                AssertionError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                learning_rates = ()
            resolved = self._format_learning_rate_range(learning_rates)
            if resolved is not None:
                return resolved
        if optimizer is not None:
            try:
                learning_rates = tuple(
                    float(
                        group["lr"].item()
                        if isinstance(group["lr"], torch.Tensor)
                        else group["lr"]
                    )
                    for group in optimizer.param_groups
                )
            except (
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                learning_rates = ()
            return self._format_learning_rate_range(learning_rates)
        return None

    @staticmethod
    def _is_world_process_zero(state: Any) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = args, control
        if not self._is_world_process_zero(state):
            return
        initial_step = max(int(state.global_step), 0)
        total_steps = max(int(state.max_steps), initial_step)
        learning_rate = self._resolve_learning_rate(
            optimizer=kwargs.get("optimizer"),
            lr_scheduler=kwargs.get("lr_scheduler"),
        )
        metrics = {} if learning_rate is None else {"lr": learning_rate}
        self.training_task = self.progress_manager.start_task(
            "train",
            label="train",
            total=total_steps,
            initial=initial_step,
            unit="step",
            metrics=metrics,
            summary_on_complete=True,
            display_rate=True,
        )

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = args, control
        if self.training_task is None:
            return
        step = max(int(state.global_step), 0)
        learning_rate = self._resolve_learning_rate(
            optimizer=kwargs.get("optimizer"),
            lr_scheduler=kwargs.get("lr_scheduler"),
        )
        metrics = None if learning_rate is None else {"lr": learning_rate}
        self.training_task.update(current=step, metrics=metrics)

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        _ = args, state, control, kwargs
        if self.training_task is None or not isinstance(logs, dict):
            return
        metrics: dict[str, Any] = {}
        if "loss" in logs:
            metrics["loss"] = logs["loss"]
        if "efficiency/useful_tokens_per_second" in logs:
            metrics["tok/s"] = logs["efficiency/useful_tokens_per_second"]
        if metrics:
            self.training_task.update(metrics=metrics)

    def on_prediction_step(
        self,
        args,
        state,
        control,
        eval_dataloader=None,
        **kwargs,
    ):  # noqa: ANN001
        _ = control, kwargs
        if not self._is_world_process_zero(state) or eval_dataloader is None:
            return
        strategy = getattr(args, "eval_strategy", "no")
        strategy = str(getattr(strategy, "value", strategy)).lower()
        if strategy == "no":
            return
        if self.prediction_task is None:
            try:
                total = len(eval_dataloader)
            except TypeError:
                total = None
            parent_task_id = "train" if self.progress_manager.is_task_active("train") else None
            self.prediction_task = self.progress_manager.start_task(
                "eval.loss",
                label="eval",
                total=total,
                unit="batch",
                parent_task_id=parent_task_id,
                display_rate=True,
            )
        else:
            task_snapshot = self.progress_manager.snapshot.tasks["eval.loss"]
            if (
                task_snapshot.total is not None
                and task_snapshot.current >= task_snapshot.total
            ):
                previous_total = task_snapshot.total
                try:
                    next_total = len(eval_dataloader)
                except TypeError:
                    next_total = None
                self.prediction_task.set_total(
                    None if next_total is None else previous_total + next_total
                )
        task_snapshot = self.progress_manager.snapshot.tasks["eval.loss"]
        if task_snapshot.total is None or task_snapshot.current < task_snapshot.total:
            self.prediction_task.advance()

    def _finish_prediction(self, metrics: dict[str, Any] | None = None) -> None:
        if self.prediction_task is None:
            return
        progress_metrics = None
        if isinstance(metrics, dict) and "eval_loss" in metrics:
            progress_metrics = {"loss": metrics["eval_loss"]}
        self.prediction_task.complete(metrics=progress_metrics)
        self.prediction_task = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
        _ = args, state, control, kwargs
        self._finish_prediction(metrics)

    def on_predict(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
        _ = args, state, control, kwargs
        self._finish_prediction(metrics)

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = args, control, kwargs
        self._finish_prediction()
        if self.training_task is None:
            return
        step = max(int(state.global_step), 0)
        self.training_task.update(current=step)
        self.training_task.complete(message="training complete")
        self.training_task = None
