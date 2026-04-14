from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm
from transformers import TrainerCallback

from shaft.utils import create_progress_bar


class ShaftProgressCallback(TrainerCallback):
    def __init__(self, *, leave: bool = False, mininterval: float = 0.2) -> None:
        self.training_bar: tqdm | None = None
        self.prediction_bar: tqdm | None = None
        self.current_step: int = 0
        self.leave = bool(leave)
        self.mininterval = float(mininterval)

    def _set_train_postfix(self, logs: dict[str, Any]) -> None:
        if self.training_bar is None:
            return
        keys = ("loss", "learning_rate", "grad_norm", "eval_loss")
        postfix: dict[str, Any] = {}
        for key in keys:
            if key not in logs:
                continue
            value = logs[key]
            if isinstance(value, float):
                postfix[key] = f"{value:.4g}"
            else:
                postfix[key] = value
        if postfix:
            self.training_bar.set_postfix(postfix, refresh=False)

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = control, kwargs
        if not state.is_world_process_zero:
            return
        self.current_step = int(state.global_step)
        self.training_bar = create_progress_bar(
            total=int(state.max_steps),
            desc="train",
            unit="step",
            leave=self.leave,
            mininterval=self.mininterval,
            colour="green",
        )

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = args, control, kwargs
        if self.training_bar is None:
            return
        step = int(state.global_step)
        if step > self.current_step:
            self.training_bar.update(step - self.current_step)
            self.current_step = step

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        _ = args, state, control, kwargs
        if logs is None:
            return
        self._set_train_postfix(dict(logs))

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
