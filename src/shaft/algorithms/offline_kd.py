from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.config import TrainConfig
from shaft.offline_kd.trainer import ShaftOfflineKDTrainer
from shaft.training.distribution_loss import resolve_distribution_objective_plan

from .base import AlgorithmContext, ShaftTrainerSpec, trainer_spec_contract
from .registry import register_algorithm


@dataclass
@register_algorithm("offline_kd")
class OfflineKDAlgorithm:
    name: str = "offline_kd"

    def prepare_trainer(
        self, *, context: AlgorithmContext, **kwargs: Any
    ) -> ShaftTrainerSpec[ShaftOfflineKDTrainer]:
        train_config: TrainConfig = kwargs.pop("train_config")
        offline_kd_config = kwargs.pop("offline_kd_config")
        if context.params:
            raise ValueError("offline_kd does not consume algorithm.params.")
        objective = offline_kd_config.objective
        plan = resolve_distribution_objective_plan(objective)
        trainer_kwargs = {
            "shaft_max_shard_size": train_config.max_shard_size,
            "loss_name": "auto",
            "optimizer_name": train_config.optimizer_name,
            "scheduler_name": train_config.scheduler_name,
            "scheduler_num_cycles": train_config.scheduler_num_cycles,
            "scheduler_power": train_config.scheduler_power,
            "adam_beta1": train_config.adam_beta1,
            "adam_beta2": train_config.adam_beta2,
            "adam_epsilon": train_config.adam_epsilon,
            "auxiliary_loss_weights": {},
            "objective_plan": plan,
            "ce_weight": offline_kd_config.loss.ce_weight,
            "kd_weight": offline_kd_config.loss.kd_weight,
            "resolved_optimizer_plan": kwargs.pop("resolved_optimizer_plan"),
            "param_group_lrs": dict(train_config.param_group_lrs),
            "no_decay_name_patterns": list(train_config.no_decay_name_patterns),
            "ignore_index": -100,
            **kwargs,
        }
        return ShaftTrainerSpec(
            trainer_cls=ShaftOfflineKDTrainer,
            kwargs=trainer_kwargs,
            contract=trainer_spec_contract(
                algorithm=self.name,
                args=trainer_kwargs["args"],
                train_config=train_config,
                extra={
                    "objective_mode": objective.mode,
                    "divergence": objective.divergence,
                    "temperature": objective.temperature,
                    "top_k": objective.top_k,
                    "ce_weight": offline_kd_config.loss.ce_weight,
                    "kd_weight": offline_kd_config.loss.kd_weight,
                },
            ),
        )

    def build_trainer(self, *, context: AlgorithmContext, **kwargs: Any) -> ShaftOfflineKDTrainer:
        return self.prepare_trainer(context=context, **kwargs).build()
