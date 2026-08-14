from __future__ import annotations

import logging
from typing import Any

from transformers import TrainingArguments

from shaft.config import RuntimeConfig
from shaft.data import (
    ShaftDataCenter,
    ShaftSampleSampler,
    validate_sample_schedule_world_size,
)
from shaft.model import (
    build_model_tokenizer_processor,
    materialize_resolved_model_artifact_identity,
    resolve_model_plan,
    summarize_resolved_finetune_plan,
    validate_model_artifact_checkpointability,
    validate_resolved_model_descriptor,
    write_resolved_finetune_summary,
)
from shaft.observability import build_progress_manager
from shaft.opd import (
    OPDCollator,
    OPDDataset,
    ShaftOPDTrainer,
    build_opd_execution_runtime,
    resolve_opd_execution_plan,
)
from shaft.opd.loss import resolve_opd_objective_plan
from shaft.opd.telemetry import (
    OPDTelemetryCallback,
    OPDTelemetryContract,
    OPDTelemetryMonitor,
)
from shaft.plugins import (
    ExecutionProxy,
    TrainerHookCallback,
    build_hook_manager,
    build_interceptor_manager,
)
from shaft.training.batch_planning import (
    ShaftBatchingMetadataCallback,
    build_batch_contract,
    build_batching_run_metadata,
    load_checkpoint_batching_metadata,
    publish_batching_run_metadata,
    validate_batching_resume_contract,
)
from shaft.training.checkpointing import (
    ensure_hf_export_layout,
    prune_root_output_layout,
    resolve_best_export_dir,
    resolve_checkpoint_protocol,
    resolve_resume_checkpoint_generation,
    validate_resume_checkpoint,
    validate_resolved_resume_checkpoint_guard,
    validate_training_state_policy,
)
from shaft.training.distributed import initialize_process_group_if_needed, is_rank_zero
from shaft.training.efficiency import invalidate_training_efficiency_summary
from shaft.training.input_contract import (
    build_train_input_contract,
    validate_train_data_identity_checkpointability,
    validate_train_input_checkpointability,
)
from shaft.training.optimizer_plan import build_resolved_optimizer_plan
from shaft.training.progress_callback import ShaftProgressCallback
from shaft.training.reproducibility import initialize_training_randomness
from shaft.training.resume_contract import (
    build_training_resume_contract,
    build_training_resume_preflight_contract,
    distributed_training_contract_stage,
)
from shaft.training.topology import validate_training_topology

from .execution import finalize_training_outputs, prepare_pipeline_call
from .registry import PIPELINE_REGISTRY, register_pipeline
from .training_args import (
    build_hf_training_args,
    resolve_training_compute_dtype,
    validate_trainable_parameter_precision,
)

logger = logging.getLogger(__name__)


def _stored_opd_input_fingerprint(training_contract: Any) -> str:
    objective = dict(training_contract.objective)
    value = objective.get("teacher_student_input_fingerprint")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "OPD checkpoint training contract has no teacher/student input fingerprint."
        )
    return value


@register_pipeline("shaft_opd")
class ShaftOPDPipeline:
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

    def build_training_args(self, *, resolved_model_plan=None) -> TrainingArguments:
        return build_hf_training_args(
            self.config,
            resolved_model_plan=resolved_model_plan,
        )

    def close(self) -> None:
        self.progress_manager.close()

    def run(self) -> dict[str, Any]:
        config = self.config
        initialize_process_group_if_needed(use_cpu=config.train.use_cpu)
        train_execution_fingerprint = ""
        with distributed_training_contract_stage(
            stage="opd-config-preflight",
            fingerprints=lambda: {
                "algorithm": str(config.algorithm.name),
                "hooks": "\x1f".join(config.plugins.hooks) or "none",
                "interceptors": "\x1f".join(config.plugins.interceptors) or "none",
            },
        ):
            self.initialize_runtime()
            if self.interceptor_manager is None or self.hook_manager is None:
                raise RuntimeError("OPD runtime plugin managers were not initialized.")
            validate_training_state_policy(config)
            validate_training_topology(config)

        invalidate_training_efficiency_summary(config.experiment.output_dir)
        training_args = (
            self._bootstrap_training_args
            if self._bootstrap_training_args is not None
            else self.build_training_args()
        )
        batch_contract = build_batch_contract(config=config, training_args=training_args)
        checkpoint_protocol = resolve_checkpoint_protocol(config.train.distributed.strategy)
        resolved_resume_checkpoint = resolve_resume_checkpoint_generation(
            config.train.resume_from_checkpoint,
            protocol=checkpoint_protocol,
            finetune_mode=config.model.finetune.mode,
        )
        resume_checkpoint = (
            None if resolved_resume_checkpoint is None else str(resolved_resume_checkpoint.path)
        )
        if resolved_resume_checkpoint is not None:
            validate_resume_checkpoint(
                resolved_resume_checkpoint,
                finetune_mode=config.model.finetune.mode,
                protocol=checkpoint_protocol,
            )
        checkpointing_requested = bool(
            config.train.save_strategy != "no" or resume_checkpoint is not None
        )

        execution_plan = None
        objective_plan = resolve_opd_objective_plan(config.opd.objective)
        with distributed_training_contract_stage(
            stage="opd-execution-plan",
            fingerprints=lambda: {
                "rollout_backend": execution_plan.rollout_backend_name,
                "teacher_provider": execution_plan.teacher_provider_name,
            },
        ):
            execution_plan = resolve_opd_execution_plan(config.opd)
            execution_plan.validate_checkpointing(
                checkpointing_requested=checkpointing_requested
            )
        assert execution_plan is not None

        teacher_artifact_plan = execution_plan.teacher_provider_type.resolve_artifact_plan(
            config,
            checkpointing_requested=checkpointing_requested,
        )
        student_plan = resolve_model_plan(
            config,
            init_from_checkpoint=config.train.init_from_checkpoint,
            require_immutable_artifact=False,
        )
        if checkpointing_requested:
            student_plan = materialize_resolved_model_artifact_identity(student_plan)

        post_model_fingerprints: dict[str, str] = {}
        telemetry_monitor: OPDTelemetryMonitor | None = None
        with distributed_training_contract_stage(
            stage="opd-model-plan",
            fingerprints=lambda: {
                "student": student_plan.fingerprint,
                "teacher": teacher_artifact_plan.fingerprint,
            },
        ):
            validate_resolved_model_descriptor(student_plan)
            student_plan.model_adapter.validate_training_finetune_config(config.model.finetune)
            student_plan.model_adapter.validate_distributed_config(
                config.train,
                finetune=config.model.finetune,
            )
            validate_model_artifact_checkpointability(
                student_plan,
                save_strategy=config.train.save_strategy,
                resume_requested=resume_checkpoint is not None,
            )
            teacher_artifact_plan.validate(
                save_strategy=config.train.save_strategy,
                resume_requested=resume_checkpoint is not None,
            )
            resolved_training_args = self.build_training_args(resolved_model_plan=student_plan)
            resolved_batch_contract = build_batch_contract(
                config=config,
                training_args=resolved_training_args,
            )
            if resolved_batch_contract.fingerprint != batch_contract.fingerprint:
                raise ValueError(
                    "Model-owned defaults changed the OPD batch contract after preflight."
                )
            training_args = resolved_training_args
            batch_contract = resolved_batch_contract
            student_dtype = resolve_training_compute_dtype(
                training_args,
                model_torch_dtype=config.model.torch_dtype,
            )
            teacher_dtype = resolve_training_compute_dtype(
                training_args,
                model_torch_dtype=config.opd.teacher.torch_dtype,
            )
            student_sequence_contract = student_plan.build_sequence_execution_contract(
                layout="padded",
                device_type="cpu" if config.train.use_cpu else "cuda",
                attention_implementation=config.model.attn_implementation,
                torch_dtype=student_dtype,
                distributed_strategy=config.train.distributed.strategy,
                torch_compile=bool(getattr(training_args, "torch_compile", False)),
            )
            teacher_sequence_contract = teacher_artifact_plan.build_sequence_contract(
                training_args=training_args,
                device_type="cpu" if config.train.use_cpu else "cuda",
                distributed_strategy=config.train.distributed.strategy,
                torch_dtype=teacher_dtype,
            )

        preflight_training_contract = None
        if resolved_resume_checkpoint is not None:
            checkpoint_metadata = load_checkpoint_batching_metadata(resume_checkpoint)
            checkpoint_training_contract = checkpoint_metadata.training_resume_contract
            if checkpoint_training_contract is None:
                raise ValueError("OPD checkpoint predates the unified training resume contract.")
            preflight_training_contract = build_training_resume_preflight_contract(
                checkpoint_contract=checkpoint_training_contract,
                config=config,
                training_args=training_args,
                batch_contract_fingerprint=batch_contract.fingerprint,
                algorithm_resume_context={
                    "teacher_model_plan_fingerprint": teacher_artifact_plan.fingerprint,
                    "teacher_student_input_fingerprint": (
                        _stored_opd_input_fingerprint(checkpoint_training_contract)
                    ),
                },
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
            stage="opd-data",
            fingerprints=lambda: {
                "batch": batch_contract.fingerprint,
                "train_execution": train_execution_fingerprint,
            },
        ):
            validate_sample_schedule_world_size(
                strategy=config.data.schedule.mixing,
                shuffle=config.data.schedule.shuffle,
                world_size=int(training_args.world_size),
            )
            data_center = ShaftDataCenter(
                config.data,
                seed=config.experiment.seed,
                train_sample_budget=batch_contract.finite_sample_plan_size(
                    max_steps=training_args.max_steps
                ),
            )
            dataset_bundle = data_center.build_dataset_bundle(OPDDataset)
            if isinstance(dataset_bundle.train_sampler, ShaftSampleSampler):
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
                raise RuntimeError("OPD data center did not publish an execution fingerprint.")
            validate_train_data_identity_checkpointability(
                data_execution_contract_complete=(dataset_bundle.train_execution_contract_complete),
                incomplete_reasons=dataset_bundle.train_execution_incomplete_reasons,
                train_dataset_type=type(dataset_bundle.train_dataset),
                save_strategy=config.train.save_strategy,
                resume_requested=resume_checkpoint is not None,
            )

        def build_artifacts(
            runtime_config: RuntimeConfig,
            plan: Any,
            sequence_contract: Any,
            *,
            role: str,
            init_from_checkpoint: str | None,
        ):
            def run_local_model_build_phase(phase: str, operation):
                result = None
                with distributed_training_contract_stage(
                    stage=f"opd-{role}-model-{phase}",
                    fingerprints=lambda: {"model_plan": plan.fingerprint},
                ):
                    result = operation()
                return result

            return build_model_tokenizer_processor(
                runtime_config,
                init_from_checkpoint=init_from_checkpoint,
                sequence_execution_contract=sequence_contract,
                resolved_model_plan=plan,
                local_phase_runner=run_local_model_build_phase,
            )

        student = build_artifacts(
            config,
            student_plan,
            student_sequence_contract,
            role="student",
            init_from_checkpoint=config.train.init_from_checkpoint,
        )
        teacher = teacher_artifact_plan.build_artifacts(build_artifacts)
        execution_runtime = build_opd_execution_runtime(
            execution_plan,
            rollout_config=config.opd.rollout,
            teacher_config=config.opd.teacher,
            teacher_model=teacher_artifact_plan.provider_model(teacher),
            seed=config.experiment.seed,
        )

        with distributed_training_contract_stage(
            stage="opd-post-model",
            fingerprints=lambda: post_model_fingerprints,
        ):
            student.model_adapter.configure_sequence_execution(
                model=student.model,
                contract=student_sequence_contract,
            )
            student.model_adapter.validate_sequence_execution(
                model=student.model,
                contract=student_sequence_contract,
            )
            input_compatibility = teacher_artifact_plan.configure_and_validate(
                student=student,
                teacher=teacher,
                provider=execution_runtime.teacher_provider,
                sequence_contract=teacher_sequence_contract,
            )
            finetune_plan = student.finetune_plan
            if finetune_plan is None:
                raise RuntimeError("OPD student loader must publish a finetune plan.")
            validate_trainable_parameter_precision(student.model, training_args)
            resolved_optimizer_plan = build_resolved_optimizer_plan(
                model=student.model,
                args=training_args,
                model_adapter=student.model_adapter,
                param_group_lrs=dict(config.train.param_group_lrs),
                no_decay_name_patterns=list(config.train.no_decay_name_patterns),
            )
            max_prompt_length = int(config.data.max_length) - int(config.opd.rollout.max_new_tokens)
            collator = OPDCollator(
                model_adapter=student.model_adapter,
                template=student.template,
                processor=student.processor,
                tokenizer=student.tokenizer,
                min_pixels=config.data.min_pixels,
                max_pixels=config.data.max_pixels,
                max_length=config.data.max_length,
                add_eos_token=False,
                max_prompt_length=max_prompt_length,
                retain_rollout_media=execution_plan.rollout_backend_type.requires_raw_media,
                collect_telemetry=bool(config.train.efficiency.enabled),
            )
            train_input_contract = build_train_input_contract(
                algorithm="opd",
                data_execution_fingerprint=train_execution_fingerprint,
                data_execution_contract_complete=bool(
                    dataset_bundle.train_execution_contract_complete
                ),
                data_execution_incomplete_reasons=(
                    dataset_bundle.train_execution_incomplete_reasons
                ),
                train_dataset_type=type(dataset_bundle.train_dataset),
                model_plan_fingerprint=student_plan.fingerprint,
                model_adapter=student.model_adapter,
                processor=student.processor,
                tokenizer=student.tokenizer,
                template=student.template,
                input_builder=OPDCollator,
                input_options={
                    "min_pixels": config.data.min_pixels,
                    "max_pixels": config.data.max_pixels,
                    "max_length": config.data.max_length,
                    "max_prompt_length": max_prompt_length,
                    "input_mode": "generation",
                    "retain_rollout_media": (
                        execution_plan.rollout_backend_type.requires_raw_media
                    ),
                    "collect_telemetry": bool(config.train.efficiency.enabled),
                    "teacher_student_input_fingerprint": input_compatibility,
                },
            )
            validate_train_input_checkpointability(
                train_input_contract,
                save_strategy=config.train.save_strategy,
            )
            training_resume_contract = build_training_resume_contract(
                config=config,
                training_args=training_args,
                batch_contract_fingerprint=batch_contract.fingerprint,
                train_input_contract_fingerprint=train_input_contract.fingerprint,
                data_execution_fingerprint=train_execution_fingerprint,
                model_plan_fingerprint=student_plan.fingerprint,
                resolved_finetune_plan_fingerprint=finetune_plan.fingerprint,
                resolved_optimizer_plan_fingerprint=resolved_optimizer_plan.fingerprint,
                sequence_execution_contract_fingerprint=(student_sequence_contract.fingerprint),
                sequence_execution_capabilities=(student_sequence_contract.capability_signature),
                resolved_experts_implementation=(
                    student_plan.model_adapter.resolve_experts_implementation(
                        config.model.experts_implementation
                    )
                ),
                algorithm_resume_context={
                    "teacher_model_plan_fingerprint": teacher_artifact_plan.fingerprint,
                    "teacher_student_input_fingerprint": input_compatibility,
                },
                hook_instances=self.hook_manager.hooks,
                interceptor_instances=self.interceptor_manager.interceptors,
            )
            if config.train.efficiency.enabled:
                opd_device_timing = bool(
                    str(config.train.efficiency.device_timing).strip().lower() != "off"
                    and training_args.device.type == "cuda"
                )
                telemetry_contract = OPDTelemetryContract(
                    training_resume_fingerprint=training_resume_contract.fingerprint,
                    rollout_backend=execution_plan.rollout_backend_name,
                    teacher_provider=execution_plan.teacher_provider_name,
                    teacher_artifact_fingerprint=teacher_artifact_plan.fingerprint,
                    objective_mode=objective_plan.mode,
                    timing_mode=("wall+cuda_events" if opd_device_timing else "wall"),
                )
                telemetry_monitor = OPDTelemetryMonitor.from_checkpoint(
                    output_dir=config.experiment.output_dir,
                    checkpoint_dir=resume_checkpoint,
                    checkpoint_global_step=(
                        0
                        if resolved_resume_checkpoint is None
                        else resolved_resume_checkpoint.global_step
                    ),
                    contract=telemetry_contract,
                    persist=bool(config.train.efficiency.persist),
                    device_timing=opd_device_timing,
                )
            if resume_checkpoint is not None:
                validate_batching_resume_contract(
                    resume_checkpoint,
                    expected_contract=batch_contract,
                    expected_sample_execution_fingerprint=train_execution_fingerprint,
                    expected_train_input_contract=train_input_contract,
                    expected_training_resume_contract=training_resume_contract,
                )
            post_model_fingerprints = {
                "student_finetune": finetune_plan.fingerprint,
                "student_optimizer": resolved_optimizer_plan.fingerprint,
                "teacher": teacher_artifact_plan.fingerprint,
                "rollout_backend": execution_plan.rollout_backend_name,
                "teacher_provider": execution_plan.teacher_provider_name,
                "input_compatibility": input_compatibility,
                "training": training_resume_contract.fingerprint,
            }

        batching_metadata = build_batching_run_metadata(
            config=config,
            training_args=training_args,
            batch_contract=batch_contract,
            sample_execution_fingerprint=train_execution_fingerprint,
            train_input_contract=train_input_contract,
            training_resume_contract=training_resume_contract,
        )
        publish_batching_run_metadata(config.experiment.output_dir, batching_metadata)
        callbacks: list[Any] = [
            ShaftBatchingMetadataCallback(batching_metadata),
            ShaftProgressCallback(self.progress_manager),
        ]
        if telemetry_monitor is not None:
            callbacks.append(OPDTelemetryCallback(telemetry_monitor))
        if self.hook_manager.hooks:
            callbacks.append(TrainerHookCallback(self.hook_manager))

        freeze_summary = summarize_resolved_finetune_plan(
            student.model,
            finetune=config.model.finetune,
            plan=finetune_plan,
            model_adapter=student.model_adapter,
        )
        if is_rank_zero():
            write_resolved_finetune_summary(config.experiment.output_dir, freeze_summary)
            logger.info("[startup] OPD student freeze summary: %s", freeze_summary.to_log_dict())

        trainer = ShaftOPDTrainer(
            model=student.model,
            execution_runtime=execution_runtime,
            objective_plan=objective_plan,
            telemetry_monitor=telemetry_monitor,
            args=training_args,
            train_dataset=dataset_bundle.train_dataset,
            eval_dataset=None,
            data_collator=collator,
            processing_class=student.processor,
            callbacks=callbacks,
            train_sampler=dataset_bundle.train_sampler,
            shaft_checkpoint_protocol=checkpoint_protocol,
            optimizer_name=config.train.optimizer_name,
            scheduler_name=config.train.scheduler_name,
            scheduler_num_cycles=config.train.scheduler_num_cycles,
            scheduler_power=config.train.scheduler_power,
            adam_beta1=config.train.adam_beta1,
            adam_beta2=config.train.adam_beta2,
            adam_epsilon=config.train.adam_epsilon,
            model_adapter=student.model_adapter,
            resolved_optimizer_plan=resolved_optimizer_plan,
            param_group_lrs=dict(config.train.param_group_lrs),
            no_decay_name_patterns=list(config.train.no_decay_name_patterns),
        )
        if telemetry_monitor is not None:
            telemetry_monitor.bind_update_applied_provider(
                lambda: not bool(trainer.accelerator.optimizer_step_was_skipped)
            )
        if resolved_resume_checkpoint is not None:
            validate_resolved_resume_checkpoint_guard(resolved_resume_checkpoint)
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        train_result.metrics.update(trainer.finalize_opd_telemetry())

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
                    model_meta=student.model_adapter,
                )
            ),
            prune_output=lambda: prune_root_output_layout(config.experiment.output_dir),
        )
        return dict(getattr(train_result, "metrics", {}) or {})


def run_opd(config: RuntimeConfig) -> dict[str, Any]:
    initialize_process_group_if_needed(use_cpu=config.train.use_cpu)
    pipeline = None
    with distributed_training_contract_stage(
        stage="opd-runtime-init",
        fingerprints=lambda: {
            "algorithm": str(config.algorithm.name).strip().lower(),
            "hooks": "\x1f".join(config.plugins.hooks) or "none",
            "interceptors": "\x1f".join(config.plugins.interceptors) or "none",
        },
    ):
        pipeline_cls = PIPELINE_REGISTRY.get("shaft_opd")
        pipeline = pipeline_cls(config)
        pipeline._bootstrap_training_args = pipeline.build_training_args()
        pipeline.initialize_runtime()
    assert pipeline is not None
    assert pipeline.interceptor_manager is not None
    runner = ExecutionProxy(
        point="pipeline.opd.run",
        target=pipeline.run,
        interceptor_manager=pipeline.interceptor_manager,
    )
    try:
        invocation = prepare_pipeline_call(runner, stage="opd-before-interceptors")
        return runner.invoke(invocation)
    except BaseException as exc:
        pipeline.progress_manager.record_failure(str(exc) or type(exc).__name__)
        raise
    finally:
        pipeline.close()
