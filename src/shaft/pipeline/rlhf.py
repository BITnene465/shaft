from __future__ import annotations

from contextlib import nullcontext
from functools import partial
import logging
from typing import Any

from transformers import TrainingArguments

from shaft.algorithms.base import AlgorithmContext
from shaft.algorithms.registry import ALGORITHM_REGISTRY
from shaft.algorithms.rlhf_utils import (
    build_trl_dpo_config,
    build_trl_grpo_config,
    resolve_grpo_grouped_sample_contract,
    validate_grpo_checkpoint_cadence,
    validate_grpo_rollout_checkpointability,
    validate_grpo_vllm_runtime_compatibility,
)
from shaft.algorithms import dpo as _dpo  # noqa: F401
from shaft.algorithms import grpo as _grpo  # noqa: F401
from shaft.algorithms import ppo as _ppo  # noqa: F401
from shaft.config import RuntimeConfig, resolve_eval_input_policy
from shaft.data import (
    DPOCollator,
    DPODataset,
    GRPODataset,
    PPOCollator,
    PPODataset,
    SFTCollator,
    SFTDataset,
    ShaftDataCenter,
    ShaftSampleSampler,
    validate_sample_schedule_world_size,
)
from shaft.model import (
    build_model_tokenizer_processor,
    resolve_model_plan,
    validate_model_artifact_checkpointability,
)
from shaft.model import summarize_resolved_finetune_plan, write_resolved_finetune_summary
from shaft.observability import build_progress_manager
from shaft.plugins import (
    ExecutionProxy,
    TrainerHookCallback,
    build_hook_manager,
    build_interceptor_manager,
)
from shaft.training.online_eval import ShaftOnlineEvalRunner
from shaft.training.batch_planning import (
    build_batch_contract,
    ShaftBatchingMetadataCallback,
    build_batching_run_metadata,
    load_checkpoint_batching_metadata,
    publish_batching_run_metadata,
    validate_batching_resume_contract,
)
from shaft.training.progress_callback import ShaftProgressCallback
from shaft.training.reproducibility import initialize_training_randomness
from shaft.training.checkpointing import (
    ensure_hf_export_layout,
    prune_root_output_layout,
    resolve_best_export_dir,
    resolve_checkpoint_protocol,
    resolve_resume_checkpoint_generation,
    resume_checkpoint_consensus_fingerprints,
    validate_resume_checkpoint,
    validate_resolved_resume_checkpoint_guard,
    validate_training_state_policy,
)
from shaft.training.distributed import initialize_process_group_if_needed
from shaft.training.distributed import is_rank_zero
from shaft.training.topology import validate_training_topology
from shaft.training.efficiency import invalidate_training_efficiency_summary
from shaft.training.eval_policy import log_eval_input_policy
from shaft.training.input_contract import (
    build_train_input_contract,
    validate_train_data_identity_checkpointability,
    validate_train_input_checkpointability,
)
from shaft.training.optimizer_plan import build_resolved_optimizer_plan
from shaft.training.resume_contract import (
    build_training_resume_contract,
    build_training_resume_preflight_contract,
    distributed_training_contract_stage,
    training_contract_section_fingerprint,
)

from .registry import PIPELINE_REGISTRY, register_pipeline
from .execution import finalize_training_outputs, prepare_pipeline_call
from .training_args import build_hf_training_args

logger = logging.getLogger(__name__)


@register_pipeline("shaft_rlhf")
class ShaftRLHFPipeline:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.interceptor_manager = None
        self.hook_manager = None
        self.progress_manager = build_progress_manager(config)
        self._bootstrap_training_args: TrainingArguments | None = None

    def initialize_runtime(self) -> None:
        if self.interceptor_manager is not None or self.hook_manager is not None:
            return
        initialize_training_randomness(
            seed=self.config.experiment.seed,
            full_determinism=self.config.train.full_determinism,
        )
        self.interceptor_manager = build_interceptor_manager(self.config.plugins.interceptors)
        self.hook_manager = build_hook_manager(self.config.plugins.hooks)

    def close(self) -> None:
        self.progress_manager.close()

    def build_training_args(self, *, resolved_model_plan=None) -> TrainingArguments:
        return build_hf_training_args(
            self.config,
            resolved_model_plan=resolved_model_plan,
        )

    def _progress_phase(self, task_id: str, *, label: str, message: str):
        if not self.progress_manager.enabled:
            return nullcontext()
        return self.progress_manager.start_task(
            task_id,
            label=label,
            unit="phase",
            message=message,
        )

    def _build_collator(
        self,
        algorithm_name: str,
        *,
        artifacts: Any,
        min_pixels: int | None,
        max_pixels: int | None,
        pixel_budgets_by_dataset: dict[
            str,
            tuple[int | None, int | None],
        ]
        | None = None,
    ):
        common_kwargs = {
            "model_adapter": artifacts.model_adapter,
            "template": artifacts.template,
            "processor": artifacts.processor,
            "tokenizer": artifacts.tokenizer,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "max_length": self.config.data.max_length,
            "add_eos_token": self.config.data.add_eos_token,
        }
        if algorithm_name == "dpo":
            return DPOCollator(
                **common_kwargs,
                pixel_budgets_by_dataset=pixel_budgets_by_dataset,
            )
        if algorithm_name == "ppo":
            return PPOCollator(**common_kwargs)
        raise ValueError(f"Unsupported RLHF algorithm: {algorithm_name!r}.")

    def run(self) -> dict[str, Any]:
        config = self.config
        initialize_process_group_if_needed(use_cpu=config.train.use_cpu)
        algorithm_name = str(config.algorithm.name).strip().lower()
        with distributed_training_contract_stage(
            stage="config-preflight",
            fingerprints=lambda: {
                "algorithm": algorithm_name,
                "hooks": "\x1f".join(config.plugins.hooks) or "none",
                "interceptors": ("\x1f".join(config.plugins.interceptors) or "none"),
            },
        ):
            self.initialize_runtime()
            if self.interceptor_manager is None or self.hook_manager is None:
                raise RuntimeError("RLHF runtime plugin managers were not initialized.")
            if algorithm_name not in {"dpo", "ppo", "grpo"}:
                raise ValueError(
                    "ShaftRLHFPipeline only supports dpo/ppo/grpo, got "
                    f"algorithm={algorithm_name!r}."
                )
            if algorithm_name == "ppo":
                if config.train.resume_from_checkpoint is not None:
                    raise ValueError(
                        "TRL PPOTrainer does not support resume_from_checkpoint "
                        "in current Shaft pipeline."
                    )
                if str(config.train.save_strategy) != "no":
                    raise ValueError(
                        "Shaft PPO does not publish resumable training checkpoints; "
                        "set train.save_strategy='no'."
                    )
            validate_training_state_policy(config)
            validate_training_topology(config)
        # This helper owns a distributed I/O convergence collective. It must run
        # only after config-preflight has reached rank consensus; otherwise a
        # rank-local plugin/config failure can strand peers in the nested
        # collective before the outer status envelope is exchanged.
        invalidate_training_efficiency_summary(config.experiment.output_dir)
        preflight_training_contract = None
        resolved_resume_checkpoint = None
        with distributed_training_contract_stage(
            stage="cheap-preflight",
            fingerprints=lambda: {
                "batch": batch_contract.fingerprint,
                **resume_checkpoint_consensus_fingerprints(
                    resolved_resume_checkpoint,
                    protocol=checkpoint_protocol,
                ),
                **(
                    {}
                    if preflight_training_contract is None
                    else {"training": preflight_training_contract.fingerprint}
                ),
            },
        ):
            training_args = (
                self._bootstrap_training_args
                if self._bootstrap_training_args is not None
                else self.build_training_args()
            )
            resolved_dpo_args = (
                build_trl_dpo_config(
                    train_args=training_args,
                    rlhf_config=config.rlhf.dpo,
                )
                if algorithm_name == "dpo"
                else None
            )
            resolved_grpo_args = (
                build_trl_grpo_config(
                    train_args=training_args,
                    rlhf_config=config.rlhf.grpo,
                )
                if algorithm_name == "grpo"
                else None
            )
            if resolved_grpo_args is not None:
                validate_grpo_vllm_runtime_compatibility(resolved_grpo_args)
                validate_grpo_rollout_checkpointability(
                    resolved_grpo_args,
                    resume_requested=(config.train.resume_from_checkpoint is not None),
                )
            batch_contract = build_batch_contract(
                config=config,
                training_args=training_args,
            )
            checkpoint_protocol = resolve_checkpoint_protocol(config.train.distributed.strategy)
            resolved_resume_checkpoint = resolve_resume_checkpoint_generation(
                config.train.resume_from_checkpoint,
                protocol=checkpoint_protocol,
            )
            resume_checkpoint = (
                None
                if resolved_resume_checkpoint is None
                else str(resolved_resume_checkpoint.path)
            )
            if resolved_resume_checkpoint is not None:
                validate_resume_checkpoint(
                    resolved_resume_checkpoint,
                    finetune_mode=config.model.finetune.mode,
                    protocol=checkpoint_protocol,
                )
                checkpoint_metadata = load_checkpoint_batching_metadata(resume_checkpoint)
                checkpoint_training_contract = checkpoint_metadata.training_resume_contract
                if checkpoint_training_contract is None:
                    raise ValueError(
                        "Checkpoint predates the unified training resume contract and "
                        "cannot be used for exact resume."
                    )
                preflight_training_contract = build_training_resume_preflight_contract(
                    checkpoint_contract=checkpoint_training_contract,
                    config=config,
                    training_args=training_args,
                    batch_contract_fingerprint=batch_contract.fingerprint,
                    resolved_dpo_args=resolved_dpo_args,
                    resolved_grpo_args=resolved_grpo_args,
                    hook_instances=self.hook_manager.hooks,
                    interceptor_instances=self.interceptor_manager.interceptors,
                )
                validate_batching_resume_contract(
                    resume_checkpoint,
                    expected_contract=batch_contract,
                    expected_training_resume_contract=preflight_training_contract,
                    require_train_input_contract_payload=True,
                    require_training_resume_contract_payload=True,
                )

        with distributed_training_contract_stage(
            stage="pre-model",
            fingerprints=lambda: {
                "batch": batch_contract.fingerprint,
                "model_plan": model_plan.fingerprint,
                **(
                    {}
                    if pre_model_training_contract is None
                    else {
                        "plugins": training_contract_section_fingerprint(
                            pre_model_training_contract,
                            section="implementation",
                            key="plugins",
                        ),
                        "training": pre_model_training_contract.fingerprint,
                    }
                ),
            },
        ):
            checkpointing_requested = algorithm_name != "ppo" and (
                str(config.train.save_strategy).strip().lower() != "no"
                or resume_checkpoint is not None
            )
            model_plan = resolve_model_plan(
                config,
                init_from_checkpoint=config.train.init_from_checkpoint,
                require_immutable_artifact=checkpointing_requested,
            )
            validate_model_artifact_checkpointability(
                model_plan,
                save_strategy=config.train.save_strategy,
                resume_requested=resume_checkpoint is not None,
            )
            resolved_training_args = self.build_training_args(resolved_model_plan=model_plan)
            resolved_batch_contract = build_batch_contract(
                config=config,
                training_args=resolved_training_args,
            )
            if resolved_batch_contract.fingerprint != batch_contract.fingerprint:
                raise ValueError(
                    "Model-owned distributed defaults changed the batch contract after "
                    "cheap resume preflight."
                )
            training_args = resolved_training_args
            batch_contract = resolved_batch_contract
            resolved_dpo_args = (
                build_trl_dpo_config(
                    train_args=training_args,
                    rlhf_config=config.rlhf.dpo,
                )
                if algorithm_name == "dpo"
                else None
            )
            resolved_grpo_args = (
                build_trl_grpo_config(
                    train_args=training_args,
                    rlhf_config=config.rlhf.grpo,
                )
                if algorithm_name == "grpo"
                else None
            )
            grouped_sample_contract = (
                resolve_grpo_grouped_sample_contract(resolved_grpo_args)
                if resolved_grpo_args is not None
                else None
            )
            sequence_execution_contract = model_plan.build_sequence_execution_contract(
                layout="padded",
                device_type="cpu" if bool(config.train.use_cpu) else "cuda",
                attention_implementation=config.model.attn_implementation,
                torch_dtype=config.model.torch_dtype,
                distributed_strategy=config.train.distributed.strategy,
                torch_compile=bool(getattr(training_args, "torch_compile", False)),
            )
            pre_model_training_contract = (
                None
                if algorithm_name == "ppo"
                else build_training_resume_contract(
                    config=config,
                    training_args=training_args,
                    batch_contract_fingerprint=batch_contract.fingerprint,
                    train_input_contract_fingerprint="pending-train-input",
                    data_execution_fingerprint="pending-data-execution",
                    model_plan_fingerprint=model_plan.fingerprint,
                    resolved_finetune_plan_fingerprint="pending-model-load",
                    resolved_optimizer_plan_fingerprint="pending-model-load",
                    sequence_execution_contract_fingerprint=(
                        sequence_execution_contract.fingerprint
                    ),
                    sequence_execution_capabilities=(
                        sequence_execution_contract.capability_signature
                    ),
                    resolved_dpo_args=resolved_dpo_args,
                    resolved_grpo_args=resolved_grpo_args,
                    hook_instances=self.hook_manager.hooks,
                    interceptor_instances=self.interceptor_manager.interceptors,
                )
            )
        training_resume_contract = None
        with distributed_training_contract_stage(
            stage="pre-model-data",
            fingerprints=lambda: {
                "train_execution": train_execution_fingerprint,
                "train_stream": train_stream_fingerprint,
            },
        ):
            validate_sample_schedule_world_size(
                strategy=config.data.schedule.mixing,
                shuffle=config.data.schedule.shuffle,
                world_size=int(training_args.world_size),
            )
            if algorithm_name == "dpo":
                dataset_cls = DPODataset
            elif algorithm_name == "ppo":
                dataset_cls = PPODataset
            else:
                dataset_cls = SFTDataset
            train_sample_budget = (
                grouped_sample_contract.finite_sample_plan_size(
                    max_steps=training_args.max_steps,
                    gradient_accumulation_steps=(training_args.gradient_accumulation_steps),
                )
                if grouped_sample_contract is not None
                else batch_contract.finite_sample_plan_size(
                    max_steps=training_args.max_steps,
                )
            )
            with self._progress_phase(
                "startup.data",
                label="data",
                message="loading",
            ):
                data_center = ShaftDataCenter(
                    config.data,
                    seed=config.experiment.seed,
                    train_sample_budget=train_sample_budget,
                )
                dataset_bundle = data_center.build_dataset_bundle(dataset_cls)
            if grouped_sample_contract is not None:
                assert resolved_grpo_args is not None
                if dataset_bundle.train_sampler is None:
                    raise RuntimeError("GRPO requires a Shaft sample plan from the data center.")
                sample_plan = dataset_bundle.train_sampler.plan
                epoch_microsteps = grouped_sample_contract.validate_epoch_sharding(
                    sample_count=len(sample_plan),
                    per_device_generation_batch_size=(
                        int(training_args.per_device_train_batch_size)
                        * int(resolved_grpo_args.steps_per_generation)
                    ),
                    data_world_size=int(training_args.world_size),
                    dataloader_drop_last=bool(training_args.dataloader_drop_last),
                )
                validate_grpo_checkpoint_cadence(
                    resolved_grpo_args,
                    epoch_microsteps=epoch_microsteps,
                    resume_checkpoint=resume_checkpoint,
                )
            elif algorithm_name == "dpo" and isinstance(
                dataset_bundle.train_sampler,
                ShaftSampleSampler,
            ):
                dataset_bundle.train_sampler.validate_epoch_sharding(
                    per_device_batch_size=int(training_args.per_device_train_batch_size),
                    data_world_size=int(training_args.world_size),
                    dataloader_drop_last=bool(training_args.dataloader_drop_last),
                    require_equal_rank_batch_cardinality=True,
                )
            train_execution_fingerprint = str(
                dataset_bundle.train_execution_fingerprint or ""
            ).strip()
            if not train_execution_fingerprint:
                raise RuntimeError("ShaftDataCenter did not publish a train execution fingerprint.")
            if grouped_sample_contract is not None:
                train_execution_fingerprint = grouped_sample_contract.execution_fingerprint(
                    train_execution_fingerprint
                )
            train_stream_fingerprint = str(dataset_bundle.train_stream_fingerprint or "").strip()
            if not train_stream_fingerprint:
                raise RuntimeError("ShaftDataCenter did not publish a train stream fingerprint.")
            validate_train_data_identity_checkpointability(
                data_execution_contract_complete=(dataset_bundle.train_execution_contract_complete),
                incomplete_reasons=(dataset_bundle.train_execution_incomplete_reasons),
                train_dataset_type=type(dataset_bundle.train_dataset),
                save_strategy=config.train.save_strategy,
                resume_requested=resume_checkpoint is not None,
            )
            if resume_checkpoint is not None:
                validate_batching_resume_contract(
                    resume_checkpoint,
                    expected_contract=batch_contract,
                    expected_sample_execution_fingerprint=(train_execution_fingerprint),
                    expected_training_resume_contract=training_resume_contract,
                )
        # HF/DeepSpeed model loading may own collectives. The preceding data
        # stage is its data readiness consensus. Model construction additionally
        # converges its pure-local prepare/finalize phases, while the raw loader
        # invocation between them remains outside every status envelope.
        def run_local_model_build_phase(phase: str, operation):
            result = None
            with distributed_training_contract_stage(
                stage=f"model-{phase}",
                fingerprints=lambda: {"model_plan": model_plan.fingerprint},
            ):
                result = operation()
            return result

        with self._progress_phase(
            "startup.model",
            label="model",
            message="loading",
        ):
            artifacts = build_model_tokenizer_processor(
                config,
                init_from_checkpoint=config.train.init_from_checkpoint,
                sequence_execution_contract=sequence_execution_contract,
                resolved_model_plan=model_plan,
                local_phase_runner=run_local_model_build_phase,
            )
        with distributed_training_contract_stage(
            stage="post-model",
            fingerprints=lambda: post_model_fingerprints,
        ):
            artifacts.model_adapter.configure_sequence_execution(
                model=artifacts.model,
                contract=sequence_execution_contract,
            )
            artifacts.model_adapter.validate_sequence_execution(
                model=artifacts.model,
                contract=sequence_execution_contract,
            )
            finetune_plan = getattr(artifacts, "finetune_plan", None)
            if finetune_plan is None:
                raise RuntimeError("RLHF model loader must publish a resolved finetune plan.")
            resolved_optimizer_plan = build_resolved_optimizer_plan(
                model=artifacts.model,
                args=training_args,
                finetune_plan=finetune_plan,
                model_adapter=artifacts.model_adapter,
                param_group_lrs=dict(config.train.param_group_lrs),
                no_decay_name_patterns=list(config.train.no_decay_name_patterns),
            )
            input_builder = {
                "dpo": DPOCollator,
                "ppo": PPOCollator,
                "grpo": GRPODataset,
            }[algorithm_name]
            input_options: dict[str, Any] = {
                "min_pixels": config.data.min_pixels,
                "max_pixels": config.data.max_pixels,
                "max_length": config.data.max_length,
                "add_eos_token": config.data.add_eos_token,
                "input_mode": ("generation" if algorithm_name == "ppo" else "training"),
                "ignore_index": -100,
                "sequence_execution_contract_fingerprint": (
                    sequence_execution_contract.fingerprint
                ),
            }
            if algorithm_name == "grpo":
                input_options.update(
                    {
                        "grouped_sample_contract": (
                            None
                            if grouped_sample_contract is None
                            else {
                                "mini_repeat_count": (grouped_sample_contract.mini_repeat_count),
                                "batch_size": grouped_sample_contract.batch_size,
                                "iteration_count": (grouped_sample_contract.iteration_count),
                                "steps_per_iteration": (
                                    grouped_sample_contract.steps_per_iteration
                                ),
                            }
                        ),
                        "rollout": config.rlhf.grpo.rollout,
                        "vllm": config.rlhf.grpo.vllm,
                    }
                )
            train_input_contract = build_train_input_contract(
                algorithm=algorithm_name,
                data_execution_fingerprint=train_execution_fingerprint,
                data_execution_contract_complete=bool(
                    dataset_bundle.train_execution_contract_complete
                ),
                data_execution_incomplete_reasons=(
                    dataset_bundle.train_execution_incomplete_reasons
                ),
                train_dataset_type=type(dataset_bundle.train_dataset),
                model_plan_fingerprint=model_plan.fingerprint,
                model_adapter=artifacts.model_adapter,
                processor=artifacts.processor,
                tokenizer=artifacts.tokenizer,
                template=artifacts.template,
                input_builder=input_builder,
                input_options=input_options,
            )
            validate_train_input_checkpointability(
                train_input_contract,
                save_strategy=config.train.save_strategy,
            )
            if not train_input_contract.exact_resume_safe:
                logger.warning(
                    "[train-input-contract] checkpointing=off exact_resume_safe=false reasons=%s",
                    list(train_input_contract.incomplete_reasons),
                )
            if algorithm_name != "ppo":
                training_resume_contract = build_training_resume_contract(
                    config=config,
                    training_args=training_args,
                    batch_contract_fingerprint=batch_contract.fingerprint,
                    train_input_contract_fingerprint=train_input_contract.fingerprint,
                    data_execution_fingerprint=(
                        train_input_contract.data_execution_fingerprint
                    ),
                    model_plan_fingerprint=model_plan.fingerprint,
                    resolved_finetune_plan_fingerprint=finetune_plan.fingerprint,
                    resolved_optimizer_plan_fingerprint=(resolved_optimizer_plan.fingerprint),
                    sequence_execution_contract_fingerprint=(
                        sequence_execution_contract.fingerprint
                    ),
                    sequence_execution_capabilities=(
                        sequence_execution_contract.capability_signature
                    ),
                    resolved_dpo_args=resolved_dpo_args,
                    resolved_grpo_args=resolved_grpo_args,
                    hook_instances=self.hook_manager.hooks,
                    interceptor_instances=(self.interceptor_manager.interceptors),
                )
            post_model_fingerprints = {
                "finetune_plan": finetune_plan.fingerprint,
                "optimizer_plan": resolved_optimizer_plan.fingerprint,
                "train_input": train_input_contract.fingerprint,
            }
            if training_resume_contract is not None:
                post_model_fingerprints["training"] = training_resume_contract.fingerprint
            if resume_checkpoint is not None:
                validate_batching_resume_contract(
                    resume_checkpoint,
                    expected_contract=batch_contract,
                    expected_sample_execution_fingerprint=(train_execution_fingerprint),
                    expected_train_input_contract=train_input_contract,
                    expected_training_resume_contract=training_resume_contract,
                )
        with distributed_training_contract_stage(
            stage="batching-metadata-build",
            fingerprints=lambda: {
                "batch": batch_contract.fingerprint,
                "train_input": train_input_contract.fingerprint,
                "training": (
                    "checkpointing-off"
                    if training_resume_contract is None
                    else training_resume_contract.fingerprint
                ),
            },
        ):
            batching_metadata = build_batching_run_metadata(
                config=config,
                training_args=training_args,
                batch_contract=batch_contract,
                sample_execution_fingerprint=train_execution_fingerprint,
                train_input_contract=train_input_contract,
                training_resume_contract=training_resume_contract,
            )
            logger.info("[batching-metadata] %s", batching_metadata.to_dict())
        publish_batching_run_metadata(config.experiment.output_dir, batching_metadata)
        with distributed_training_contract_stage(
            stage="trainer-input-build",
            fingerprints=lambda: {
                "train_input": train_input_contract.fingerprint,
                "training": (
                    "checkpointing-off"
                    if training_resume_contract is None
                    else training_resume_contract.fingerprint
                ),
                "algorithm": (f"{type(algorithm).__module__}.{type(algorithm).__qualname__}"),
                "callbacks": str(len(callbacks)),
            },
        ):
            freeze_summary = summarize_resolved_finetune_plan(
                artifacts.model,
                finetune=config.model.finetune,
                plan=finetune_plan,
                model_adapter=artifacts.model_adapter,
            )
            if is_rank_zero():
                write_resolved_finetune_summary(config.experiment.output_dir, freeze_summary)
                logger.info("[startup] resolved freeze summary: %s", freeze_summary.to_log_dict())
            train_dataset = dataset_bundle.train_dataset
            eval_dataset: Any = dataset_bundle.eval_dataset
            use_named_eval_datasets = bool(
                algorithm_name in {"dpo", "grpo"}
                and config.eval.enabled
                and dataset_bundle.eval_datasets_by_name
                and config.eval.datasets
                and (
                    config.eval.loss_metrics_enabled
                    or config.eval.online_metrics_enabled
                    or config.eval.metric_for_best_model in {"eval_final_loss", "eval_final_score"}
                )
            )
            if use_named_eval_datasets:
                eval_dataset = dataset_bundle.eval_datasets_by_name
            eval_input_policy = None
            eval_default_pixel_budget = None
            eval_pixel_budgets_by_dataset: dict[
                str,
                tuple[int | None, int | None],
            ] = {}
            if algorithm_name in {"dpo", "grpo"}:
                eval_input_policy = resolve_eval_input_policy(
                    config.eval,
                    train_min_pixels=config.data.min_pixels,
                    train_max_pixels=config.data.max_pixels,
                )
                if config.eval.enabled:
                    log_eval_input_policy(
                        policy=eval_input_policy,
                        model_adapter=artifacts.model_adapter,
                    )
                eval_default_pixel_budget = eval_input_policy.default_pixel_budget
                eval_pixel_budgets_by_dataset = eval_input_policy.pixel_budgets_by_dataset()
            if algorithm_name == "grpo":
                assert eval_input_policy is not None
                assert eval_default_pixel_budget is not None
                train_grpo_dataset_kwargs = {
                    "template": artifacts.template,
                    "image_preprocessor": partial(
                        artifacts.model_adapter.prepare_rollout_image,
                        min_pixels=config.data.min_pixels,
                        max_pixels=config.data.max_pixels,
                    ),
                }
                train_dataset = GRPODataset(train_dataset, **train_grpo_dataset_kwargs)
                if eval_dataset is not None and not (
                    config.eval.online_metrics_enabled and use_named_eval_datasets
                ):

                    def _build_grpo_eval_dataset(dataset: Any, dataset_name: str | None = None):
                        pixel_budget = eval_input_policy.pixel_budget_for(dataset_name)
                        return GRPODataset(
                            dataset,
                            template=artifacts.template,
                            image_preprocessor=partial(
                                artifacts.model_adapter.prepare_rollout_image,
                                min_pixels=pixel_budget.min_pixels,
                                max_pixels=pixel_budget.max_pixels,
                            ),
                        )

                    if isinstance(eval_dataset, dict):
                        eval_dataset = {
                            dataset_name: _build_grpo_eval_dataset(dataset, dataset_name)
                            for dataset_name, dataset in eval_dataset.items()
                        }
                    else:
                        eval_dataset = _build_grpo_eval_dataset(eval_dataset)
            callbacks = [ShaftBatchingMetadataCallback(batching_metadata)]
            # Keep callback topology identical on every rank. Non-zero ranks receive
            # a manager without sinks, and ShaftProgressCallback is a no-op there.
            callbacks.append(ShaftProgressCallback(self.progress_manager))
            if self.hook_manager.hooks:
                callbacks.append(TrainerHookCallback(self.hook_manager))
            callbacks_or_none = callbacks or None
            online_eval_runner = None
            if (
                algorithm_name == "grpo"
                and config.eval.enabled
                and config.eval.online_metrics_enabled
            ):
                online_eval_runner = ShaftOnlineEvalRunner(
                    eval_config=config.eval,
                    prompt_collator=SFTCollator(
                        model_adapter=artifacts.model_adapter,
                        template=artifacts.template,
                        processor=artifacts.processor,
                        tokenizer=artifacts.tokenizer,
                        min_pixels=eval_default_pixel_budget.min_pixels,
                        max_pixels=eval_default_pixel_budget.max_pixels,
                        max_length=config.data.max_length,
                        add_eos_token=config.data.add_eos_token,
                        include_targets_in_inputs=False,
                        include_metadata=True,
                        input_mode="generation",
                        pixel_budgets_by_dataset=eval_pixel_budgets_by_dataset,
                    ),
                    progress_manager=self.progress_manager,
                )

            algorithm_cls = ALGORITHM_REGISTRY.get(algorithm_name)
            algorithm = algorithm_cls()
            processing_class = artifacts.processor
            algorithm_extra_kwargs: dict[str, Any] = {}
            if algorithm_name == "ppo":
                algorithm_extra_kwargs["model_meta"] = artifacts.model_meta
            else:
                algorithm_extra_kwargs["shaft_checkpoint_protocol"] = checkpoint_protocol
            trainer_kwargs: dict[str, Any] = {
                "context": AlgorithmContext(params=dict(config.algorithm.params)),
                "train_config": config.train,
                "rlhf_config": getattr(config.rlhf, algorithm_name),
                "finetune_mode": config.model.finetune.mode,
                "model": artifacts.model,
                "args": training_args,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset if config.eval.enabled else None,
                "processing_class": processing_class,
                "callbacks": callbacks_or_none,
                "model_adapter": artifacts.model_adapter,
                "finetune_plan": finetune_plan,
                "resolved_optimizer_plan": resolved_optimizer_plan,
                **algorithm_extra_kwargs,
            }
            if algorithm_name == "dpo":
                trainer_kwargs["train_sampler"] = dataset_bundle.train_sampler
                trainer_kwargs["resolved_dpo_args"] = resolved_dpo_args
            if algorithm_name == "grpo":
                assert dataset_bundle.train_sampler is not None
                trainer_kwargs["sample_plan"] = dataset_bundle.train_sampler.plan
                trainer_kwargs["grouped_sample_contract"] = grouped_sample_contract
                trainer_kwargs["resolved_grpo_args"] = resolved_grpo_args
                trainer_kwargs["online_eval_runner"] = online_eval_runner
                trainer_kwargs["eval_config"] = config.eval
            if algorithm_name != "grpo":
                trainer_kwargs["data_collator"] = self._build_collator(
                    algorithm_name,
                    artifacts=artifacts,
                    min_pixels=config.data.min_pixels,
                    max_pixels=config.data.max_pixels,
                )
            if algorithm_name == "dpo":
                assert eval_default_pixel_budget is not None
                trainer_kwargs["eval_data_collator"] = self._build_collator(
                    algorithm_name,
                    artifacts=artifacts,
                    min_pixels=eval_default_pixel_budget.min_pixels,
                    max_pixels=eval_default_pixel_budget.max_pixels,
                    pixel_budgets_by_dataset=eval_pixel_budgets_by_dataset,
                )
                trainer_kwargs["eval_config"] = config.eval
        trainer_spec = None
        with distributed_training_contract_stage(
            stage="trainer-prepare",
            fingerprints=lambda: {
                "algorithm": f"{type(algorithm).__module__}.{type(algorithm).__qualname__}",
                "trainer": trainer_spec.fingerprint if trainer_spec is not None else "missing",
            },
        ):
            # Reference/value/reward model copies, reward construction, runtime
            # validation, and TRL config resolution are pure-local preparation.
            # Converge failures here before any peer enters Trainer/Accelerator.
            trainer_spec = algorithm.prepare_trainer(**trainer_kwargs)
        assert trainer_spec is not None
        # Trainer/Accelerator construction may initialize backend collectives.
        # Invoke the single constructor boundary only after every rank has a
        # matching prepared spec, outside every status-envelope collective.
        trainer = trainer_spec.build()

        with distributed_training_contract_stage(
            stage="resume-load-guard",
            fingerprints=lambda: resume_checkpoint_consensus_fingerprints(
                resolved_resume_checkpoint,
                protocol=checkpoint_protocol,
            ),
        ):
            if resolved_resume_checkpoint is not None:
                validate_resolved_resume_checkpoint_guard(resolved_resume_checkpoint)
        if algorithm_name == "ppo":
            train_result = trainer.train()
        else:
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        best_export_dir = (
            resolve_best_export_dir(config.experiment.output_dir)
            if config.train.save_final_model
            else None
        )
        finalize_training_outputs(
            trainer=trainer,
            best_export_dir=best_export_dir,
            save_final_state=config.train.save_final_state,
            validate_export=(
                None
                if best_export_dir is None
                else lambda normalized_export_dir: ensure_hf_export_layout(
                    normalized_export_dir,
                    finetune_mode=config.model.finetune.mode,
                    model_meta=artifacts.model_adapter,
                )
            ),
            prune_output=lambda: prune_root_output_layout(config.experiment.output_dir),
        )
        if train_result is not None and hasattr(train_result, "metrics"):
            return dict(train_result.metrics or {})
        log_history = getattr(getattr(trainer, "state", None), "log_history", None)
        if isinstance(log_history, list):
            for entry in reversed(log_history):
                if isinstance(entry, dict):
                    return dict(entry)
        return {}


def run_rlhf(config: RuntimeConfig) -> dict[str, Any]:
    initialize_process_group_if_needed(use_cpu=config.train.use_cpu)
    pipeline = None
    with distributed_training_contract_stage(
        stage="runtime-init",
        fingerprints=lambda: {
            "algorithm": str(config.algorithm.name).strip().lower(),
            "hooks": "\x1f".join(config.plugins.hooks) or "none",
            "interceptors": ("\x1f".join(config.plugins.interceptors) or "none"),
        },
    ):
        pipeline_cls = PIPELINE_REGISTRY.get("shaft_rlhf")
        pipeline = pipeline_cls(config)
        pipeline._bootstrap_training_args = pipeline.build_training_args()
        pipeline.initialize_runtime()
    assert pipeline is not None
    assert pipeline.interceptor_manager is not None
    runner = ExecutionProxy(
        point="pipeline.rlhf.run",
        target=pipeline.run,
        interceptor_manager=pipeline.interceptor_manager,
    )
    try:
        # A rank-local before interceptor can fail before pipeline.run reaches
        # its first collective. Converge that readiness phase independently,
        # then invoke the collective-owning pipeline body outside the envelope.
        invocation = prepare_pipeline_call(
            runner,
            stage="rlhf-before-interceptors",
        )
        return runner.invoke(invocation)
    except BaseException as exc:
        pipeline.progress_manager.record_failure(str(exc) or type(exc).__name__)
        raise
    finally:
        pipeline.close()
