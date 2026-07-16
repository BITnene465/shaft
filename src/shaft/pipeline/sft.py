from __future__ import annotations

from contextlib import nullcontext
import logging
from typing import Any

from transformers import TrainingArguments

from shaft.algorithms.base import AlgorithmContext
from shaft.algorithms.registry import ALGORITHM_REGISTRY
from shaft.algorithms import sft as _sft  # noqa: F401
from shaft.config import RuntimeConfig, resolve_eval_input_policy
from shaft.data import (
    SFTCollator,
    SFTDataset,
    ShaftPlannedBatchSampler,
    ShaftBatchPlanner,
    ShaftBatchPlanningSpec,
    ShaftBatchPlanningState,
    ShaftBatchMicrobatchPlan,
    ShaftDataCenter,
    ShaftSFTSampleCostProvider,
    ShaftSampleSampler,
    sft_runtime_source_identity,
    validate_sample_schedule_world_size,
    validate_sft_cost_model_adapter,
    validate_sft_cost_dataset,
)
from shaft.model import (
    build_model_tokenizer_processor,
    resolve_model_plan,
    summarize_resolved_finetune_plan,
    validate_model_artifact_checkpointability,
    write_resolved_finetune_summary,
)
from shaft.model.generation import align_model_generation_config
from shaft.observability import (
    ShaftTrainingEfficiencyContract,
    build_progress_manager,
    training_hardware_fingerprint,
    training_software_fingerprint,
)
from shaft.plugins import (
    ExecutionProxy,
    TrainerHookCallback,
    build_hook_manager,
    build_interceptor_manager,
)
from shaft.training.batch_planning import (
    ShaftBatchContract,
    ShaftBatchingMetadataCallback,
    ShaftBatchPlanningCallback,
    build_batch_contract,
    build_batch_planning_resume_contract_fingerprint,
    build_batching_run_metadata,
    load_checkpoint_batching_metadata,
    load_batch_planning_state,
    publish_batching_run_metadata,
    validate_batching_resume_contract,
    validate_batch_planning_resume_contract,
)
from shaft.training.epoch_interval_callback import ShaftEpochIntervalCallback
from shaft.training.eval_policy import log_eval_input_policy
from shaft.training.efficiency import (
    ShaftTrainingEfficiencyCallback,
    ShaftTrainingEfficiencyMonitor,
    ShaftTrainingEfficiencySnapshotInvalidationCallback,
    invalidate_training_efficiency_summary,
)
from shaft.training.online_eval import ShaftOnlineEvalRunner
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
from shaft.training.distributed import initialize_process_group_if_needed, is_rank_zero
from shaft.training.topology import validate_training_topology

from .registry import PIPELINE_REGISTRY, register_pipeline
from .execution import finalize_training_outputs, prepare_pipeline_call
from .training_args import build_hf_training_args


logger = logging.getLogger(__name__)


def _build_planned_batch_sampler(
    *,
    config: RuntimeConfig,
    training_args: TrainingArguments,
    train_dataset: SFTDataset,
    train_schedule: Any,
    artifacts: Any,
    resume_checkpoint: str | None,
    resume_global_step: int,
    resume_commit_fingerprint: str | None,
    resume_contract_fingerprint: str | None,
    batch_contract: ShaftBatchContract,
) -> tuple[ShaftPlannedBatchSampler, ShaftBatchPlanningSpec]:
    provider: ShaftSFTSampleCostProvider | None = None
    spec: ShaftBatchPlanningSpec | None = None
    initial_state: ShaftBatchPlanningState | None = None
    preflight_fingerprint: str | None = None
    preflight_plan: ShaftBatchMicrobatchPlan | None = None
    with distributed_training_contract_stage(
        stage="batch-planning-startup",
        fingerprints=lambda: {
            "contract": spec.fingerprint,
            **({} if int(training_args.world_size) <= 1 else {"preflight": preflight_fingerprint}),
        },
    ):
        if train_schedule is None or not hasattr(train_schedule, "ref_at"):
            raise TypeError(
                "Planned grouping requires Shaft's horizon-independent sample schedule."
            )
        validate_sft_cost_dataset(train_dataset)
        validate_sft_cost_model_adapter(artifacts.model_adapter)
        provider = ShaftSFTSampleCostProvider(
            dataset=train_dataset,
            model_adapter=artifacts.model_adapter,
            template=artifacts.template,
            processor=artifacts.processor,
            tokenizer=artifacts.tokenizer,
            min_pixels=config.data.min_pixels,
            max_pixels=config.data.max_pixels,
            max_length=config.data.max_length,
            add_eos_token=config.data.add_eos_token,
            loss_scale_name=config.train.loss_scale,
            cache_size=int(config.data.batching.cost_cache_size),
        )
        max_tokens = batch_contract.local_token_capacity
        if max_tokens is None:
            raise ValueError("Planned grouping requires a resolved local token capacity.")
        spec = ShaftBatchPlanningSpec(
            grouping=batch_contract.grouping,
            packing=batch_contract.packing,
            layout=batch_contract.layout,
            max_sequence_length=batch_contract.max_sequence_length,
            data_world_size=batch_contract.data_world_size,
            buffer_size=int(batch_contract.buffer_size or 0),
            cardinality=batch_contract.cardinality,
            per_device_microbatch_size=batch_contract.per_device_microbatch_size,
            max_tokens_per_microbatch=int(max_tokens),
            resource_budgets=batch_contract.resource_budgets,
            seed=int(config.experiment.seed),
            sample_schedule_fingerprint=str(train_schedule.fingerprint),
            cost_fingerprint=str(provider.fingerprint),
        )
        if resume_checkpoint is not None:
            if not resume_contract_fingerprint:
                raise ValueError(
                    "Planned resume requires a resolved training contract fingerprint."
                )
            initial_state = load_batch_planning_state(
                resume_checkpoint,
                expected_spec=spec,
                expected_global_step=resume_global_step,
                gradient_accumulation_steps=int(training_args.gradient_accumulation_steps),
                expected_resume_contract_fingerprint=resume_contract_fingerprint,
                expected_commit_fingerprint=resume_commit_fingerprint,
            )
            for buffered in initial_state.buffer:
                resolved_cost = provider(buffered.sample_ref)
                if resolved_cost != buffered.cost:
                    raise ValueError(
                        "Batch-planning resume detected changed cost for buffered "
                        f"draw_id={buffered.sample_ref.context.draw_id}; source media, "
                        "prompt transforms, or cost semantics changed in place."
                    )
        if int(training_args.world_size) > 1:
            preflight_plan = ShaftBatchPlanner(
                schedule=train_schedule,
                cost_provider=provider,
                spec=spec,
                state=initial_state,
            ).next_global_microbatch()
            preflight_fingerprint = preflight_plan.fingerprint
        assert provider is not None and spec is not None
        if resume_checkpoint is not None:
            # The custom sampler already starts at the committed optimizer boundary.
            # HF must not replay its own local-batch skip on top of that state.
            training_args.ignore_data_skip = True

        sampler = ShaftPlannedBatchSampler(
            train_schedule,
            cost_provider=provider,
            spec=spec,
            global_microstep_count=(
                int(training_args.max_steps) * int(training_args.gradient_accumulation_steps)
            ),
            planning_frame_size=int(training_args.gradient_accumulation_steps),
            initial_state=initial_state,
            preflight_plan=preflight_plan,
        )
    return sampler, spec


@register_pipeline("shaft_sft")
class ShaftSFTPipeline:
    """HF-first SFT pipeline with optional lazy global batch planning."""

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
                raise RuntimeError("SFT runtime plugin managers were not initialized.")
            if algorithm_name != "sft":
                raise ValueError(
                    f"ShaftSFTPipeline only supports sft, got {algorithm_name!r}. "
                    "Use ShaftRLHFPipeline for DPO/PPO."
                )
            validate_training_state_policy(config)
            validate_training_topology(config)
        # This helper owns a distributed I/O convergence collective. It must run
        # only after config-preflight has reached rank consensus; otherwise a
        # rank-local plugin/config failure can strand peers in the nested
        # collective before the outer status envelope is exchanged.
        invalidate_training_efficiency_summary(config.experiment.output_dir)
        # Build the cheap, model-independent projection first. On resume this
        # rejects objective/optimizer/GA/plugin drift before immutable local
        # model weights are hashed.
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
            batch_contract = build_batch_contract(
                config=config,
                training_args=training_args,
            )
            planned = batch_contract.is_planned
            checkpoint_protocol = resolve_checkpoint_protocol(config.train.distributed.strategy)
            resolved_resume_checkpoint = resolve_resume_checkpoint_generation(
                config.train.resume_from_checkpoint,
                protocol=checkpoint_protocol,
                require_planning_state=planned,
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
                "plugins": training_contract_section_fingerprint(
                    pre_model_training_contract,
                    section="implementation",
                    key="plugins",
                ),
                "training": pre_model_training_contract.fingerprint,
            },
        ):
            checkpointing_requested = (
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
            validate_sample_schedule_world_size(
                strategy=config.data.schedule.mixing,
                shuffle=config.data.schedule.shuffle,
                world_size=int(training_args.world_size),
            )
            sequence_execution_contract = model_plan.build_sequence_execution_contract(
                layout=batch_contract.layout,
                device_type="cpu" if bool(config.train.use_cpu) else "cuda",
                attention_implementation=config.model.attn_implementation,
                torch_dtype=config.model.torch_dtype,
                distributed_strategy=config.train.distributed.strategy,
                torch_compile=bool(getattr(training_args, "torch_compile", False)),
            )
            logger.info(
                "[sequence-contract] model=%s layout=%s device=%s attention=%s "
                "dtype=%s distributed=%s compile=%s fingerprint=%s",
                config.model.model_type,
                sequence_execution_contract.layout,
                sequence_execution_contract.device_type,
                sequence_execution_contract.attention_implementation,
                sequence_execution_contract.torch_dtype,
                sequence_execution_contract.distributed_strategy,
                sequence_execution_contract.torch_compile,
                sequence_execution_contract.fingerprint,
            )
            pre_model_training_contract = build_training_resume_contract(
                config=config,
                training_args=training_args,
                batch_contract_fingerprint=batch_contract.fingerprint,
                train_input_contract_fingerprint="pending-train-input",
                data_execution_fingerprint="pending-data-execution",
                model_plan_fingerprint=model_plan.fingerprint,
                resolved_finetune_plan_fingerprint="pending-model-load",
                resolved_optimizer_plan_fingerprint="pending-model-load",
                sequence_execution_contract_fingerprint=(sequence_execution_contract.fingerprint),
                sequence_execution_capabilities=(sequence_execution_contract.capability_signature),
                hook_instances=self.hook_manager.hooks,
                interceptor_instances=self.interceptor_manager.interceptors,
            )
        training_resume_contract = None
        planning_resume_contract = None
        logger.info(
            "[batch-contract] grouping=%s cardinality=%s packing=%s layout=%s "
            "local_pack_range=%s global_pack_range=%s "
            "gradient_accumulation=%s optimizer_pack_range=%s "
            "buffer_size=%s cost_cache_size=%s max_tokens=%s "
            "resource_budgets=%s "
            "min_pixels=%s max_pixels=%s",
            batch_contract.grouping,
            batch_contract.cardinality,
            batch_contract.packing,
            batch_contract.layout,
            batch_contract.local_pack_count_bounds,
            batch_contract.global_pack_count_bounds,
            batch_contract.gradient_accumulation_steps,
            batch_contract.optimizer_pack_count_bounds,
            batch_contract.buffer_size,
            config.data.batching.cost_cache_size if planned else None,
            batch_contract.local_token_capacity,
            dict(batch_contract.resource_budgets),
            config.data.min_pixels,
            config.data.max_pixels,
        )

        with distributed_training_contract_stage(
            stage="pre-model-data",
            fingerprints=lambda: {
                "train_execution": train_execution_fingerprint,
                "train_stream": train_stream_fingerprint,
            },
        ):
            with self._progress_phase(
                "startup.data",
                label="data",
                message="loading",
            ):
                data_center = ShaftDataCenter(
                    config.data,
                    seed=config.experiment.seed,
                    train_sample_budget=batch_contract.finite_sample_plan_size(
                        max_steps=training_args.max_steps,
                    ),
                )
                dataset_bundle = data_center.build_dataset_bundle(SFTDataset)
            train_dataset = dataset_bundle.train_dataset
            train_sampler = dataset_bundle.train_sampler
            train_schedule = dataset_bundle.train_schedule
            if isinstance(train_sampler, ShaftSampleSampler):
                train_sampler.validate_epoch_sharding(
                    per_device_batch_size=int(training_args.per_device_train_batch_size),
                    data_world_size=int(training_args.world_size),
                    dataloader_drop_last=bool(training_args.dataloader_drop_last),
                )
            train_execution_fingerprint = str(
                dataset_bundle.train_execution_fingerprint or ""
            ).strip()
            train_stream_fingerprint = str(dataset_bundle.train_stream_fingerprint or "").strip()
            if not train_execution_fingerprint:
                raise RuntimeError("ShaftDataCenter did not publish a train execution fingerprint.")
            if not train_stream_fingerprint:
                raise RuntimeError("ShaftDataCenter did not publish a train stream fingerprint.")
            validate_train_data_identity_checkpointability(
                data_execution_contract_complete=(dataset_bundle.train_execution_contract_complete),
                incomplete_reasons=(dataset_bundle.train_execution_incomplete_reasons),
                train_dataset_type=type(train_dataset),
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
            fingerprints=lambda: {
                "finetune_plan": finetune_plan.fingerprint,
                "optimizer_plan": resolved_optimizer_plan.fingerprint,
                "train_input": train_input_contract.fingerprint,
                "training": training_resume_contract.fingerprint,
            },
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
                raise RuntimeError(
                    "Checkpointable SFT requires a resolved finetune plan from the model loader."
                )
            resolved_optimizer_plan = build_resolved_optimizer_plan(
                model=artifacts.model,
                args=training_args,
                finetune_plan=finetune_plan,
                model_adapter=artifacts.model_adapter,
                param_group_lrs=dict(config.train.param_group_lrs),
                no_decay_name_patterns=list(config.train.no_decay_name_patterns),
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
                train_dataset_type=type(train_dataset),
                model_plan_fingerprint=model_plan.fingerprint,
                model_adapter=artifacts.model_adapter,
                processor=artifacts.processor,
                tokenizer=artifacts.tokenizer,
                template=artifacts.template,
                input_builder=SFTCollator,
                input_options={
                    "min_pixels": config.data.min_pixels,
                    "max_pixels": config.data.max_pixels,
                    "max_length": config.data.max_length,
                    "add_eos_token": config.data.add_eos_token,
                    "loss_scale_name": config.train.loss_scale,
                    "layout": batch_contract.layout,
                    "packing_mode": batch_contract.packing,
                    "include_targets_in_inputs": True,
                    "include_metadata": False,
                    "input_mode": "training",
                    "ignore_index": -100,
                    "sequence_execution_contract_fingerprint": (
                        sequence_execution_contract.fingerprint
                    ),
                },
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
            training_resume_contract = build_training_resume_contract(
                config=config,
                training_args=training_args,
                batch_contract_fingerprint=batch_contract.fingerprint,
                train_input_contract_fingerprint=train_input_contract.fingerprint,
                data_execution_fingerprint=(train_input_contract.data_execution_fingerprint),
                model_plan_fingerprint=model_plan.fingerprint,
                resolved_finetune_plan_fingerprint=finetune_plan.fingerprint,
                resolved_optimizer_plan_fingerprint=(resolved_optimizer_plan.fingerprint),
                sequence_execution_contract_fingerprint=(sequence_execution_contract.fingerprint),
                sequence_execution_capabilities=(sequence_execution_contract.capability_signature),
                hook_instances=self.hook_manager.hooks,
                interceptor_instances=self.interceptor_manager.interceptors,
            )
            planning_resume_contract = (
                build_batch_planning_resume_contract_fingerprint(
                    training_resume_contract=training_resume_contract,
                    sequence_execution_contract_fingerprint=(
                        sequence_execution_contract.fingerprint
                    ),
                )
                if planned
                else None
            )
            if resume_checkpoint is not None:
                validate_batching_resume_contract(
                    resume_checkpoint,
                    expected_contract=batch_contract,
                    expected_sample_execution_fingerprint=(train_execution_fingerprint),
                    expected_train_input_contract=train_input_contract,
                    expected_training_resume_contract=training_resume_contract,
                )
                if planned:
                    assert planning_resume_contract is not None
                    validate_batch_planning_resume_contract(
                        resume_checkpoint,
                        expected_resume_contract_fingerprint=(planning_resume_contract),
                        expected_commit_fingerprint=(
                            None
                            if resolved_resume_checkpoint is None
                            else resolved_resume_checkpoint.commit_fingerprint
                        ),
                    )
        with distributed_training_contract_stage(
            stage="post-model-finalization",
            fingerprints=lambda: {
                "model_plan": model_plan.fingerprint,
                "finetune_plan": finetune_plan.fingerprint,
                "training": training_resume_contract.fingerprint,
            },
        ):
            align_model_generation_config(
                artifacts.model,
                tokenizer=artifacts.tokenizer,
                max_new_tokens=config.eval.max_new_tokens,
                do_sample=config.eval.do_sample,
                temperature=config.eval.temperature,
                repetition_penalty=1.0,
            )
            freeze_summary = summarize_resolved_finetune_plan(
                artifacts.model,
                finetune=config.model.finetune,
                plan=finetune_plan,
                model_adapter=artifacts.model_adapter,
            )
            if is_rank_zero():
                write_resolved_finetune_summary(
                    config.experiment.output_dir,
                    freeze_summary,
                )
                logger.info("[startup] resolved freeze summary: %s", freeze_summary.to_log_dict())

        train_batch_sampler: ShaftPlannedBatchSampler | None = None
        planning_spec: ShaftBatchPlanningSpec | None = None
        if planned:
            train_batch_sampler, planning_spec = _build_planned_batch_sampler(
                config=config,
                training_args=training_args,
                train_dataset=train_dataset,
                train_schedule=train_schedule,
                artifacts=artifacts,
                resume_checkpoint=resume_checkpoint,
                resume_global_step=(
                    0
                    if resolved_resume_checkpoint is None
                    else resolved_resume_checkpoint.global_step
                ),
                resume_commit_fingerprint=(
                    None
                    if resolved_resume_checkpoint is None
                    else resolved_resume_checkpoint.commit_fingerprint
                ),
                resume_contract_fingerprint=planning_resume_contract,
                batch_contract=batch_contract,
            )
            train_sampler = None

        with distributed_training_contract_stage(
            stage="batching-metadata-build",
            fingerprints=lambda: {
                "batch": batch_contract.fingerprint,
                "train_input": train_input_contract.fingerprint,
                "training": training_resume_contract.fingerprint,
                "planner": ("none" if planning_spec is None else planning_spec.fingerprint),
            },
        ):
            batching_metadata = build_batching_run_metadata(
                config=config,
                training_args=training_args,
                planning_spec=planning_spec,
                batch_contract=batch_contract,
                sample_execution_fingerprint=train_execution_fingerprint,
                train_input_contract=train_input_contract,
                training_resume_contract=training_resume_contract,
            )
            logger.info("[batching-metadata] %s", batching_metadata.to_dict())
        publish_batching_run_metadata(config.experiment.output_dir, batching_metadata)

        with distributed_training_contract_stage(
            stage="trainer-input-preflight",
            fingerprints=lambda: {
                "training": training_resume_contract.fingerprint,
                "efficiency": (
                    "disabled"
                    if efficiency_contract is None
                    else efficiency_contract.source_fingerprint
                ),
            },
        ):
            eval_dataset: Any = dataset_bundle.eval_dataset
            use_named_eval_datasets = bool(
                config.eval.enabled
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

            resume_global_step = (
                0
                if resolved_resume_checkpoint is None
                else resolved_resume_checkpoint.global_step
            )
            efficiency_enabled = bool(config.train.efficiency.enabled)
            efficiency_contract: ShaftTrainingEfficiencyContract | None = None
            if efficiency_enabled:
                source_identity = sft_runtime_source_identity(train_dataset)
                efficiency_contract = ShaftTrainingEfficiencyContract(
                    algorithm=algorithm_name,
                    model_type=config.model.model_type,
                    model_name_or_path=config.model.model_name_or_path,
                    model_plan_fingerprint=model_plan.fingerprint,
                    finetune_mode=config.model.finetune.mode,
                    torch_dtype=config.model.torch_dtype,
                    attention_implementation=config.model.attn_implementation,
                    seed=config.experiment.seed,
                    max_steps=training_args.max_steps,
                    num_train_epochs=training_args.num_train_epochs,
                    data_world_size=batch_contract.data_world_size,
                    gradient_accumulation_steps=(training_args.gradient_accumulation_steps),
                    max_length=config.data.max_length,
                    min_pixels=config.data.min_pixels,
                    max_pixels=config.data.max_pixels,
                    optimizer_name=config.train.optimizer_name,
                    scheduler_name=config.train.scheduler_name,
                    learning_rate=config.train.learning_rate,
                    source_fingerprint=source_identity.fingerprint,
                    source_contract_complete=source_identity.complete,
                    sample_execution_fingerprint=train_execution_fingerprint,
                    sample_stream_fingerprint=train_stream_fingerprint,
                    software_fingerprint=training_software_fingerprint(),
                    hardware_fingerprint=training_hardware_fingerprint(training_args.device),
                    measurement_protocol="shaft-efficiency-optimizer-frame-v3",
                    timing_mode=(
                        "cuda_optimizer_frame"
                        if str(config.train.efficiency.device_timing).strip().lower() != "off"
                        and training_args.device.type == "cuda"
                        else "host_optimizer_frame"
                    ),
                    batch_contract_fingerprint=batch_contract.fingerprint,
                    sequence_contract_fingerprint=(sequence_execution_contract.fingerprint),
                )
        efficiency_monitor: ShaftTrainingEfficiencyMonitor | None = None
        if efficiency_contract is not None:
            efficiency_monitor = ShaftTrainingEfficiencyMonitor.from_checkpoint(
                output_dir=config.experiment.output_dir,
                checkpoint_dir=resume_checkpoint,
                checkpoint_global_step=resume_global_step,
                device_timing=(str(config.train.efficiency.device_timing).strip().lower() != "off"),
                persist=bool(config.train.efficiency.persist),
                contract=efficiency_contract,
            )

        with distributed_training_contract_stage(
            stage="trainer-input-build",
            fingerprints=lambda: {
                "training": training_resume_contract.fingerprint,
                "algorithm": (f"{type(algorithm).__module__}.{type(algorithm).__qualname__}"),
                "trainer": trainer_spec.fingerprint,
                "callbacks": str(len(callbacks)),
            },
        ):
            callbacks: list[Any] = [ShaftBatchingMetadataCallback(batching_metadata)]
            if train_batch_sampler is not None and planning_spec is not None:
                if planning_resume_contract is None:
                    raise RuntimeError(
                        "Batch-planning training contract fingerprint was not resolved."
                    )
                callbacks.append(
                    ShaftBatchPlanningCallback(
                        train_batch_sampler,
                        planning_spec,
                        gradient_accumulation_steps=int(training_args.gradient_accumulation_steps),
                        resume_contract_fingerprint=planning_resume_contract,
                    )
                )
            if efficiency_monitor is not None:
                callbacks.append(ShaftTrainingEfficiencyCallback(efficiency_monitor))
            else:
                callbacks.append(ShaftTrainingEfficiencySnapshotInvalidationCallback())
            # Keep callback topology identical on every rank. Non-zero ranks receive
            # a manager without sinks, and ShaftProgressCallback is a no-op there.
            callbacks.append(ShaftProgressCallback(self.progress_manager))
            if (
                config.eval.enabled
                and config.eval.eval_strategy == "epoch"
                and int(config.eval.epoch_interval) > 1
            ) or (
                config.train.save_strategy == "epoch" and int(config.train.save_epoch_interval) > 1
            ):
                callbacks.append(
                    ShaftEpochIntervalCallback(
                        eval_epoch_interval=int(config.eval.epoch_interval),
                        save_epoch_interval=int(config.train.save_epoch_interval),
                    )
                )
            if self.hook_manager.hooks:
                callbacks.append(TrainerHookCallback(self.hook_manager))

            collator = SFTCollator(
                model_adapter=artifacts.model_adapter,
                template=artifacts.template,
                processor=artifacts.processor,
                tokenizer=artifacts.tokenizer,
                min_pixels=config.data.min_pixels,
                max_pixels=config.data.max_pixels,
                max_length=config.data.max_length,
                add_eos_token=config.data.add_eos_token,
                include_targets_in_inputs=True,
                include_metadata=False,
                loss_scale_name=config.train.loss_scale,
                layout=batch_contract.layout,
                packing_mode=batch_contract.packing,
                collect_stats=efficiency_monitor is not None,
            )
            eval_collator = SFTCollator(
                model_adapter=artifacts.model_adapter,
                template=artifacts.template,
                processor=artifacts.processor,
                tokenizer=artifacts.tokenizer,
                min_pixels=eval_default_pixel_budget.min_pixels,
                max_pixels=eval_default_pixel_budget.max_pixels,
                max_length=config.data.max_length,
                add_eos_token=config.data.add_eos_token,
                include_targets_in_inputs=True,
                include_metadata=False,
                loss_scale_name=config.train.loss_scale,
                layout="padded",
                packing_mode="none",
                collect_stats=False,
                pixel_budgets_by_dataset=eval_pixel_budgets_by_dataset,
            )
            online_eval_runner = None
            if config.eval.enabled and config.eval.online_metrics_enabled:
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
                        collect_stats=False,
                        pixel_budgets_by_dataset=eval_pixel_budgets_by_dataset,
                    ),
                    progress_manager=self.progress_manager,
                )

            algorithm = ALGORITHM_REGISTRY.get(algorithm_name)()
            trainer_kwargs = {
                "context": AlgorithmContext(params=dict(config.algorithm.params)),
                "train_config": config.train,
                "model": artifacts.model,
                "args": training_args,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset if config.eval.enabled else None,
                "train_sampler": train_sampler,
                "train_batch_sampler": train_batch_sampler,
                "processing_class": artifacts.processor,
                "data_collator": collator,
                "eval_data_collator": eval_collator,
                "callbacks": callbacks,
                "online_eval_runner": online_eval_runner,
                "eval_config": config.eval,
                "model_adapter": artifacts.model_adapter,
                "finetune_plan": finetune_plan,
                "resolved_optimizer_plan": resolved_optimizer_plan,
                "efficiency_monitor": efficiency_monitor,
                "shaft_checkpoint_protocol": checkpoint_protocol,
            }
            trainer_spec = algorithm.prepare_trainer(**trainer_kwargs)
        # Trainer/Accelerator construction may initialize backend collectives.
        # trainer-input-build prepares the pure-local spec; invoke its single
        # constructor boundary outside the status-envelope collective.
        trainer = trainer_spec.build()

        with distributed_training_contract_stage(
            stage="trainer-finalization",
            fingerprints=lambda: {
                "training": training_resume_contract.fingerprint,
                "trainer": (f"{type(trainer).__module__}.{type(trainer).__qualname__}"),
            },
        ):
            if efficiency_monitor is not None:
                efficiency_monitor.bind_update_applied_provider(
                    lambda: not bool(trainer.accelerator.optimizer_step_was_skipped)
                )

        with distributed_training_contract_stage(
            stage="resume-load-guard",
            fingerprints=lambda: resume_checkpoint_consensus_fingerprints(
                resolved_resume_checkpoint,
                protocol=checkpoint_protocol,
            ),
        ):
            if resolved_resume_checkpoint is not None:
                validate_resolved_resume_checkpoint_guard(resolved_resume_checkpoint)
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        finalize_efficiency = getattr(trainer, "finalize_training_efficiency", None)
        if callable(finalize_efficiency):
            train_result.metrics.update(finalize_efficiency())
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
        return dict(train_result.metrics or {})


def run_sft(config: RuntimeConfig) -> dict[str, Any]:
    initialize_process_group_if_needed(use_cpu=config.train.use_cpu)
    pipeline = None
    with distributed_training_contract_stage(
        stage="runtime-init",
        fingerprints=lambda: {
            "algorithm": "sft",
            "hooks": "\x1f".join(config.plugins.hooks) or "none",
            "interceptors": ("\x1f".join(config.plugins.interceptors) or "none"),
        },
    ):
        pipeline_cls = PIPELINE_REGISTRY.get("shaft_sft")
        pipeline = pipeline_cls(config)
        pipeline._bootstrap_training_args = pipeline.build_training_args()
        pipeline.initialize_runtime()
    assert pipeline is not None
    assert pipeline.interceptor_manager is not None
    runner = ExecutionProxy(
        point="pipeline.sft.run",
        target=pipeline.run,
        interceptor_manager=pipeline.interceptor_manager,
    )
    try:
        # A rank-local before interceptor can fail before pipeline.run reaches
        # its first collective. Converge that readiness phase independently,
        # then invoke the collective-owning pipeline body outside the envelope.
        invocation = prepare_pipeline_call(
            runner,
            stage="sft-before-interceptors",
        )
        return runner.invoke(invocation)
    except BaseException as exc:
        pipeline.progress_manager.record_failure(str(exc) or type(exc).__name__)
        raise
    finally:
        pipeline.close()
