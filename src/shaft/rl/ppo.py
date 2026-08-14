from __future__ import annotations

from typing import Any

from shaft.algorithms.base import AlgorithmContext
from shaft.algorithms.registry import ALGORITHM_REGISTRY
from shaft.algorithms import ppo as _ppo  # noqa: F401
from shaft.data import PPOCollator, PPODataset

from .base import RLResolvedArgs, RLRuntime, RLRuntimeContext, RLTrainerInputs
from .registry import register_rl_runtime


@register_rl_runtime("ppo")
class PPORuntime(RLRuntime):
    name = "ppo"
    dataset_type = PPODataset
    input_builder = PPOCollator
    supports_checkpoint = False
    input_mode = "generation"

    def validate_config(self, config) -> None:
        if config.train.resume_from_checkpoint is not None:
            raise ValueError(
                "TRL PPOTrainer does not support resume_from_checkpoint in current "
                "Shaft RL runtime."
            )
        if str(config.train.save_strategy) != "no":
            raise ValueError(
                "Shaft PPO does not publish resumable training checkpoints; "
                "set train.save_strategy='no'."
            )

    def resolve_args(self, config, training_args) -> RLResolvedArgs:
        _ = config, training_args
        return RLResolvedArgs()

    def _build_collator(self, context: RLRuntimeContext) -> PPOCollator:
        artifacts = context.artifacts
        config = context.config
        return PPOCollator(
            model_adapter=artifacts.model_adapter,
            template=artifacts.template,
            processor=artifacts.processor,
            tokenizer=artifacts.tokenizer,
            min_pixels=config.data.min_pixels,
            max_pixels=config.data.max_pixels,
            max_length=config.data.max_length,
            add_eos_token=config.data.add_eos_token,
        )

    def build_trainer_inputs(self, context: RLRuntimeContext) -> RLTrainerInputs:
        return RLTrainerInputs(
            train_dataset=context.dataset_bundle.train_dataset,
            eval_dataset=context.dataset_bundle.eval_dataset,
            data_collator=self._build_collator(context),
        )

    def prepare_trainer(
        self,
        context: RLRuntimeContext,
        *,
        trainer_inputs: RLTrainerInputs,
        callbacks: list[Any] | None,
        resolved_optimizer_plan: Any,
    ):
        algorithm = ALGORITHM_REGISTRY.get(self.name)()
        return algorithm.prepare_trainer(
            context=AlgorithmContext(params=dict(context.config.algorithm.params)),
            train_config=context.config.train,
            rlhf_config=context.config.rlhf.ppo,
            finetune_mode=context.config.model.finetune.mode,
            model=context.artifacts.model,
            model_meta=context.artifacts.model_meta,
            args=context.training_args,
            train_dataset=trainer_inputs.train_dataset,
            eval_dataset=(
                trainer_inputs.eval_dataset if context.config.eval.enabled else None
            ),
            processing_class=context.artifacts.processor,
            callbacks=callbacks,
            model_adapter=context.artifacts.model_adapter,
            resolved_optimizer_plan=resolved_optimizer_plan,
            data_collator=trainer_inputs.data_collator,
        )
