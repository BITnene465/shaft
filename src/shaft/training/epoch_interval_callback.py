from __future__ import annotations

import math

from transformers import TrainerCallback
from transformers.trainer_utils import IntervalStrategy, SaveStrategy


class ShaftEpochIntervalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        eval_epoch_interval: int = 1,
        save_epoch_interval: int = 1,
    ) -> None:
        self.eval_epoch_interval = max(int(eval_epoch_interval), 1)
        self.save_epoch_interval = max(int(save_epoch_interval), 1)

    def on_epoch_end(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = kwargs
        epoch = getattr(state, "epoch", None)
        if epoch is None:
            return control
        completed_epoch = max(int(round(float(epoch))), 0)
        total_epochs = max(int(math.ceil(float(getattr(args, "num_train_epochs", 0.0)))), 0)
        is_final_epoch = total_epochs > 0 and completed_epoch >= total_epochs

        if (
            args.eval_strategy == IntervalStrategy.EPOCH
            and control.should_evaluate
            and self.eval_epoch_interval > 1
            and not is_final_epoch
            and completed_epoch % self.eval_epoch_interval != 0
        ):
            control.should_evaluate = False

        if (
            args.save_strategy == SaveStrategy.EPOCH
            and control.should_save
            and self.save_epoch_interval > 1
            and not is_final_epoch
            and completed_epoch % self.save_epoch_interval != 0
        ):
            control.should_save = False

        return control
