from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.config import TrainConfig
from shaft.training.sft_trainer import ShaftSFTTrainer

from .base import AlgorithmContext, ShaftTrainerSpec, trainer_spec_contract
from .registry import register_algorithm


@dataclass
@register_algorithm("sft")
class SFTAlgorithm:
    name: str = "sft"

    def prepare_trainer(
        self,
        *,
        context: AlgorithmContext,
        **kwargs: Any,
    ) -> ShaftTrainerSpec[ShaftSFTTrainer]:
        _ = context
        train_config: TrainConfig = kwargs.pop("train_config")
        trainer_kwargs = {
            "loss_name": train_config.loss_name,
            "optimizer_name": train_config.optimizer_name,
            "scheduler_name": train_config.scheduler_name,
            "scheduler_num_cycles": train_config.scheduler_num_cycles,
            "scheduler_power": train_config.scheduler_power,
            "adam_beta1": train_config.adam_beta1,
            "adam_beta2": train_config.adam_beta2,
            "adam_epsilon": train_config.adam_epsilon,
            "model_adapter": kwargs.pop("model_adapter"),
            "finetune_plan": kwargs.pop("finetune_plan"),
            "resolved_optimizer_plan": kwargs.pop("resolved_optimizer_plan"),
            "param_group_lrs": dict(train_config.param_group_lrs),
            "no_decay_name_patterns": list(train_config.no_decay_name_patterns),
            "ignore_index": -100,
            **kwargs,
        }
        return ShaftTrainerSpec(
            trainer_cls=ShaftSFTTrainer,
            kwargs=trainer_kwargs,
            contract=trainer_spec_contract(
                algorithm=self.name,
                args=trainer_kwargs["args"],
                train_config=train_config,
                extra={"loss_name": train_config.loss_name},
            ),
        )

    def build_trainer(self, *, context: AlgorithmContext, **kwargs: Any) -> ShaftSFTTrainer:
        return self.prepare_trainer(context=context, **kwargs).build()
