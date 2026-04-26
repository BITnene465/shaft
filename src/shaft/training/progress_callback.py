from __future__ import annotations

from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import TrainerCallback

from shaft.utils import create_progress_bar


class ShaftProgressCallback(TrainerCallback):
    def __init__(self, *, leave: bool = False, mininterval: float = 0.2) -> None:
        self.training_bar: tqdm | None = None
        self.prediction_bar: tqdm | None = None
        self.current_step: int = 0
        self.train_postfix: dict[str, Any] = {}
        self.leave = bool(leave)
        self.mininterval = float(mininterval)

    def _format_postfix_value(self, value: Any) -> Any:
        if isinstance(value, float):
            return f"{value:.4g}"
        return value

    def _resolve_learning_rate(self, *, optimizer: Any = None, lr_scheduler: Any = None) -> float | None:
        last_lr = None
        if lr_scheduler is not None and not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            try:
                last_lr = lr_scheduler.get_last_lr()[0]
            except (AssertionError, AttributeError, IndexError, KeyError, TypeError):
                last_lr = None
        if last_lr is None and optimizer is not None:
            try:
                last_lr = optimizer.param_groups[0]["lr"]
            except (AttributeError, IndexError, KeyError, TypeError):
                last_lr = None
        if isinstance(last_lr, torch.Tensor):
            last_lr = last_lr.item()
        if last_lr is None:
            return None
        try:
            return float(last_lr)
        except (TypeError, ValueError):
            return None

    def _set_train_postfix(
        self,
        *,
        logs: dict[str, Any] | None = None,
        learning_rate: float | None = None,
    ) -> None:
        if self.training_bar is None:
            return
        keys = ("loss", "learning_rate", "grad_norm", "eval_loss", "eval_final_loss", "eval_final_score")
        if logs is not None:
            for key in keys:
                if key not in logs:
                    continue
                self.train_postfix[key] = self._format_postfix_value(logs[key])
        if learning_rate is not None:
            self.train_postfix["learning_rate"] = self._format_postfix_value(learning_rate)
        if self.train_postfix:
            self.training_bar.set_postfix(dict(self.train_postfix), refresh=False)

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = control
        if not state.is_world_process_zero:
            return
        self.train_postfix = {}
        self.current_step = max(int(state.global_step), 0)
        total_steps = max(int(state.max_steps), self.current_step)
        self.training_bar = create_progress_bar(
            total=total_steps,
            initial=self.current_step,
            desc="train",
            unit="step",
            leave=self.leave,
            mininterval=self.mininterval,
            colour="green",
        )
        self._set_train_postfix(
            learning_rate=self._resolve_learning_rate(
                optimizer=kwargs.get("optimizer"),
                lr_scheduler=kwargs.get("lr_scheduler"),
            )
        )

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = args, control, kwargs
        if self.training_bar is None:
            return
        step = int(state.global_step)
        if step > self.current_step:
            self.training_bar.update(step - self.current_step)
            self.current_step = step
        self._set_train_postfix(
            learning_rate=self._resolve_learning_rate(
                optimizer=kwargs.get("optimizer"),
                lr_scheduler=kwargs.get("lr_scheduler"),
            )
        )

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        _ = args, state, control, kwargs
        if logs is None:
            return
        self._set_train_postfix(logs=dict(logs))

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):  # noqa: ANN001
        _ = state, control, kwargs
        if str(args.eval_strategy).lower() != "no" and eval_dataloader is not None:
            if self.prediction_bar is None:
                self.prediction_bar = create_progress_bar(
                    total=len(eval_dataloader),
                    desc="eval",
                    unit="batch",
                    leave=self.leave,
                    mininterval=self.mininterval,
                    colour="cyan",
                )
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = args, state, control, kwargs
        if self.prediction_bar is not None:
            self.prediction_bar.close()
        self.prediction_bar = None

    def on_predict(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = args, state, control, kwargs
        if self.prediction_bar is not None:
            self.prediction_bar.close()
        self.prediction_bar = None

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = args, state, control, kwargs
        if self.training_bar is not None:
            self.training_bar.close()
        self.training_bar = None
