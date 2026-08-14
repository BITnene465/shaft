from __future__ import annotations

from typing import Any

from shaft.algorithms.base import AlgorithmContext
from shaft.algorithms.registry import ALGORITHM_REGISTRY
from shaft.algorithms.rlhf_utils import build_trl_dpo_config
from shaft.algorithms import dpo as _dpo  # noqa: F401
from shaft.data import DPODataset, DPOCollator, ShaftSampleSampler
from shaft.training.resume_contract import register_training_resume_policy
from shaft.training.trl_trainers import ShaftDPOTrainer

from .base import RLResolvedArgs, RLRuntime, RLRuntimeContext, RLTrainerInputs
from .common import (
    build_dpo_collator,
    resolve_named_eval_dataset,
    resolve_rl_eval_input_policy,
)
from .registry import register_rl_runtime
from .resume import build_trl_resume_implementation, dpo_resume_objective


def _resume_implementation(config: Any) -> dict[str, Any]:
    _ = config
    return build_trl_resume_implementation(
        algorithm="dpo",
        config_builder=build_trl_dpo_config,
        trainer_impl=ShaftDPOTrainer,
    )


register_training_resume_policy(
    "dpo",
    objective_builder=dpo_resume_objective,
    implementation_builder=_resume_implementation,
)


@register_rl_runtime("dpo")
class DPORuntime(RLRuntime):
    name = "dpo"
    dataset_type = DPODataset
    input_builder = DPOCollator

    def resolve_args(self, config, training_args) -> RLResolvedArgs:
        args = build_trl_dpo_config(
            train_args=training_args,
            rlhf_config=config.rlhf.dpo,
        )
        return RLResolvedArgs(
            trainer_args=args,
            resume_context={"resolved_trainer_args": args},
        )

    def validate_dataset_bundle(
        self,
        bundle,
        *,
        resolved_args,
        training_args,
        resume_checkpoint,
    ) -> None:
        _ = resolved_args, resume_checkpoint
        if isinstance(bundle.train_sampler, ShaftSampleSampler):
            bundle.train_sampler.validate_epoch_sharding(
                per_device_batch_size=int(training_args.per_device_train_batch_size),
                data_world_size=int(training_args.world_size),
                dataloader_drop_last=bool(training_args.dataloader_drop_last),
                require_equal_rank_batch_cardinality=True,
            )

    def build_trainer_inputs(self, context: RLRuntimeContext) -> RLTrainerInputs:
        eval_dataset, _ = resolve_named_eval_dataset(context)
        eval_policy = resolve_rl_eval_input_policy(context)
        default_budget = eval_policy.default_pixel_budget
        budgets_by_dataset = eval_policy.pixel_budgets_by_dataset()
        return RLTrainerInputs(
            train_dataset=context.dataset_bundle.train_dataset,
            eval_dataset=eval_dataset,
            data_collator=build_dpo_collator(
                context,
                min_pixels=context.config.data.min_pixels,
                max_pixels=context.config.data.max_pixels,
            ),
            eval_data_collator=build_dpo_collator(
                context,
                min_pixels=default_budget.min_pixels,
                max_pixels=default_budget.max_pixels,
                pixel_budgets_by_dataset=budgets_by_dataset,
            ),
            eval_config=context.config.eval,
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
            rlhf_config=context.config.rlhf.dpo,
            finetune_mode=context.config.model.finetune.mode,
            model=context.artifacts.model,
            args=context.training_args,
            train_dataset=trainer_inputs.train_dataset,
            eval_dataset=(trainer_inputs.eval_dataset if context.config.eval.enabled else None),
            processing_class=context.artifacts.processor,
            callbacks=callbacks,
            model_adapter=context.artifacts.model_adapter,
            resolved_optimizer_plan=resolved_optimizer_plan,
            shaft_checkpoint_protocol=context.checkpoint_protocol,
            train_sampler=context.dataset_bundle.train_sampler,
            resolved_dpo_args=context.resolved_args.trainer_args,
            data_collator=trainer_inputs.data_collator,
            eval_data_collator=trainer_inputs.eval_data_collator,
            eval_config=trainer_inputs.eval_config,
        )
