from __future__ import annotations

from functools import partial
from typing import Any

from shaft.algorithms.base import AlgorithmContext
from shaft.algorithms.registry import ALGORITHM_REGISTRY
from shaft.algorithms.rlhf_utils import (
    build_trl_grpo_config,
    resolve_grpo_grouped_sample_contract,
    validate_grpo_checkpoint_cadence,
    validate_grpo_rollout_checkpointability,
    validate_grpo_vllm_runtime_compatibility,
)
from shaft.algorithms import grpo as _grpo  # noqa: F401
from shaft.data import GRPODataset, SFTCollator, SFTDataset
from shaft.training.input_contract import callable_semantic_signature
from shaft.training.online_eval import ShaftOnlineEvalRunner
from shaft.training.resume_contract import register_training_resume_policy
from shaft.training.trl_trainers import ShaftGRPOTrainer

from .base import RLResolvedArgs, RLRuntime, RLRuntimeContext, RLTrainerInputs
from .common import resolve_named_eval_dataset, resolve_rl_eval_input_policy
from .registry import register_rl_runtime
from .resume import build_trl_resume_implementation, grpo_resume_objective


def _resume_implementation(config: Any) -> dict[str, Any]:
    _ = config
    from shaft.algorithms import rlhf_utils as rlhf_utils_module

    return build_trl_resume_implementation(
        algorithm="grpo",
        config_builder=build_trl_grpo_config,
        trainer_impl=ShaftGRPOTrainer,
        extra_config_policy=(
            callable_semantic_signature(
                rlhf_utils_module._precision_model_init_kwargs,
                role="trl_config:precision_model_init",
            ),
            callable_semantic_signature(
                rlhf_utils_module._set_default_model_init_kwargs,
                role="trl_config:set_default_model_init",
            ),
        ),
    )


register_training_resume_policy(
    "grpo",
    objective_builder=grpo_resume_objective,
    implementation_builder=_resume_implementation,
)


@register_rl_runtime("grpo")
class GRPORuntime(RLRuntime):
    name = "grpo"
    dataset_type = SFTDataset
    input_builder = GRPODataset

    def resolve_args(self, config, training_args) -> RLResolvedArgs:
        args = build_trl_grpo_config(
            train_args=training_args,
            rlhf_config=config.rlhf.grpo,
        )
        validate_grpo_vllm_runtime_compatibility(args)
        validate_grpo_rollout_checkpointability(
            args,
            resume_requested=(config.train.resume_from_checkpoint is not None),
        )
        return RLResolvedArgs(
            trainer_args=args,
            sample_contract=resolve_grpo_grouped_sample_contract(args),
            resume_context={"resolved_trainer_args": args},
        )

    def build_train_sample_budget(
        self,
        *,
        batch_contract,
        resolved_args,
        training_args,
    ) -> int | None:
        _ = batch_contract
        return resolved_args.sample_contract.finite_sample_plan_size(
            max_steps=training_args.max_steps,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        )

    def validate_dataset_bundle(
        self,
        bundle,
        *,
        resolved_args,
        training_args,
        resume_checkpoint,
    ) -> None:
        if bundle.train_sampler is None:
            raise RuntimeError("GRPO requires a Shaft sample plan from the data center.")
        contract = resolved_args.sample_contract
        args = resolved_args.trainer_args
        epoch_microsteps = contract.validate_epoch_sharding(
            sample_count=len(bundle.train_sampler.plan),
            per_device_generation_batch_size=(
                int(training_args.per_device_train_batch_size) * int(args.steps_per_generation)
            ),
            data_world_size=int(training_args.world_size),
            dataloader_drop_last=bool(training_args.dataloader_drop_last),
        )
        validate_grpo_checkpoint_cadence(
            args,
            epoch_microsteps=epoch_microsteps,
            resume_checkpoint=resume_checkpoint,
        )

    def bind_execution_fingerprint(
        self,
        fingerprint: str,
        *,
        resolved_args: RLResolvedArgs,
    ) -> str:
        return resolved_args.sample_contract.execution_fingerprint(fingerprint)

    def input_options(
        self,
        context: RLRuntimeContext,
        *,
        sequence_execution_contract: Any,
    ) -> dict[str, Any]:
        options = super().input_options(
            context,
            sequence_execution_contract=sequence_execution_contract,
        )
        contract = context.resolved_args.sample_contract
        options.update(
            {
                "grouped_sample_contract": {
                    "mini_repeat_count": contract.mini_repeat_count,
                    "batch_size": contract.batch_size,
                    "iteration_count": contract.iteration_count,
                    "steps_per_iteration": contract.steps_per_iteration,
                },
                "rollout": context.config.rlhf.grpo.rollout,
                "vllm": context.config.rlhf.grpo.vllm,
            }
        )
        return options

    @staticmethod
    def _wrap_dataset(dataset: Any, context: RLRuntimeContext, dataset_name: str | None = None):
        policy = resolve_rl_eval_input_policy(context)
        budget = (
            policy.default_pixel_budget
            if dataset_name is None
            else policy.pixel_budget_for(dataset_name)
        )
        return GRPODataset(
            dataset,
            template=context.artifacts.template,
            image_preprocessor=partial(
                context.artifacts.model_adapter.prepare_rollout_image,
                min_pixels=budget.min_pixels,
                max_pixels=budget.max_pixels,
            ),
        )

    def build_trainer_inputs(self, context: RLRuntimeContext) -> RLTrainerInputs:
        config = context.config
        artifacts = context.artifacts
        eval_dataset, use_named = resolve_named_eval_dataset(context)
        eval_policy = resolve_rl_eval_input_policy(context)
        default_budget = eval_policy.default_pixel_budget
        budgets_by_dataset = eval_policy.pixel_budgets_by_dataset()
        train_dataset = GRPODataset(
            context.dataset_bundle.train_dataset,
            template=artifacts.template,
            image_preprocessor=partial(
                artifacts.model_adapter.prepare_rollout_image,
                min_pixels=config.data.min_pixels,
                max_pixels=config.data.max_pixels,
            ),
        )
        if eval_dataset is not None and not (config.eval.online_metrics_enabled and use_named):
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    dataset_name: self._wrap_dataset(dataset, context, dataset_name)
                    for dataset_name, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._wrap_dataset(eval_dataset, context)
        online_eval_runner = None
        if config.eval.enabled and config.eval.online_metrics_enabled:
            online_eval_runner = ShaftOnlineEvalRunner(
                eval_config=config.eval,
                prompt_collator=SFTCollator(
                    model_adapter=artifacts.model_adapter,
                    template=artifacts.template,
                    processor=artifacts.processor,
                    tokenizer=artifacts.tokenizer,
                    min_pixels=default_budget.min_pixels,
                    max_pixels=default_budget.max_pixels,
                    max_length=config.data.max_length,
                    add_eos_token=config.data.add_eos_token,
                    include_targets_in_inputs=False,
                    include_metadata=True,
                    input_mode="generation",
                    pixel_budgets_by_dataset=budgets_by_dataset,
                ),
                progress_manager=context.progress_manager,
            )
        return RLTrainerInputs(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            online_eval_runner=online_eval_runner,
            eval_config=config.eval,
        )

    def prepare_trainer(
        self,
        context: RLRuntimeContext,
        *,
        trainer_inputs: RLTrainerInputs,
        callbacks: list[Any] | None,
        finetune_plan: Any,
        resolved_optimizer_plan: Any,
    ):
        algorithm = ALGORITHM_REGISTRY.get(self.name)()
        if context.dataset_bundle.train_sampler is None:
            raise RuntimeError("GRPO requires a Shaft sample plan from the data center.")
        return algorithm.prepare_trainer(
            context=AlgorithmContext(params=dict(context.config.algorithm.params)),
            train_config=context.config.train,
            rlhf_config=context.config.rlhf.grpo,
            finetune_mode=context.config.model.finetune.mode,
            model=context.artifacts.model,
            args=context.training_args,
            train_dataset=trainer_inputs.train_dataset,
            eval_dataset=(trainer_inputs.eval_dataset if context.config.eval.enabled else None),
            processing_class=context.artifacts.processor,
            callbacks=callbacks,
            model_adapter=context.artifacts.model_adapter,
            finetune_plan=finetune_plan,
            resolved_optimizer_plan=resolved_optimizer_plan,
            shaft_checkpoint_protocol=context.checkpoint_protocol,
            sample_plan=context.dataset_bundle.train_sampler.plan,
            grouped_sample_contract=context.resolved_args.sample_contract,
            resolved_grpo_args=context.resolved_args.trainer_args,
            online_eval_runner=trainer_inputs.online_eval_runner,
            eval_config=trainer_inputs.eval_config,
        )
