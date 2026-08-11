from __future__ import annotations

from contextlib import nullcontext
import logging
from typing import Any

from transformers import TrainingArguments

from shaft.config import RuntimeConfig
from shaft.data import (
    ShaftDataCenter,
    validate_sample_schedule_world_size,
)
from shaft.model import (
    build_model_tokenizer_processor,
    materialize_resolved_model_artifact_identity,
    resolve_model_plan,
    validate_model_artifact_checkpointability,
    validate_resolved_model_descriptor,
)
from shaft.model import summarize_resolved_finetune_plan, write_resolved_finetune_summary
from shaft.observability import build_progress_manager
from shaft.plugins import (
    ExecutionProxy,
    TrainerHookCallback,
    build_hook_manager,
    build_interceptor_manager,
)
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
from shaft.rl import RLRuntimeContext, build_rl_runtime

from .registry import PIPELINE_REGISTRY, register_pipeline
from .execution import finalize_training_outputs, prepare_pipeline_call
from .training_args import (
    build_hf_training_args,
    resolve_training_compute_dtype,
    validate_trainable_parameter_precision,
)

logger = logging.getLogger(__name__)


@register_pipeline("shaft_rl")
class ShaftRLPipeline:
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
        runtime = build_rl_runtime(algorithm_name)
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
                raise RuntimeError("RL runtime plugin managers were not initialized.")
            runtime.validate_config(config)
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
            resolved_runtime_args = runtime.resolve_args(config, training_args)
            batch_contract = build_batch_contract(
                config=config,
                training_args=training_args,
            )
            checkpoint_protocol = resolve_checkpoint_protocol(config.train.distributed.strategy)
            resolved_resume_checkpoint = resolve_resume_checkpoint_generation(
                config.train.resume_from_checkpoint,
                protocol=checkpoint_protocol,
                finetune_mode=config.model.finetune.mode,
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
                    **resolved_runtime_args.resume_contract_kwargs(),
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

        checkpointing_requested = runtime.checkpointing_requested(
            config,
            resume_checkpoint=resume_checkpoint,
        )
        with distributed_training_contract_stage(
            stage="model-plan-local",
            fingerprints=lambda: {
                "checkpointing": (
                    "required" if checkpointing_requested else "disabled"
                ),
                "model_plan": model_plan.fingerprint,
            },
        ):
            model_plan = resolve_model_plan(
                config,
                init_from_checkpoint=config.train.init_from_checkpoint,
                require_immutable_artifact=False,
            )
        if checkpointing_requested:
            # This helper owns a long-timeout Gloo consensus and must not be
            # nested inside the generic rank-status envelope above.
            model_plan = materialize_resolved_model_artifact_identity(model_plan)

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
            validate_resolved_model_descriptor(model_plan)
            model_plan.model_adapter.validate_training_finetune_config(
                config.model.finetune
            )
            model_plan.model_adapter.validate_distributed_config(
                config.train,
                finetune=config.model.finetune,
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
            resolved_runtime_args = runtime.resolve_args(config, training_args)
            training_compute_dtype = resolve_training_compute_dtype(
                training_args,
                model_torch_dtype=config.model.torch_dtype,
            )
            sequence_execution_contract = model_plan.build_sequence_execution_contract(
                layout="padded",
                device_type="cpu" if bool(config.train.use_cpu) else "cuda",
                attention_implementation=config.model.attn_implementation,
                torch_dtype=training_compute_dtype,
                distributed_strategy=config.train.distributed.strategy,
                torch_compile=bool(getattr(training_args, "torch_compile", False)),
            )
            pre_model_training_contract = (
                build_training_resume_contract(
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
                    resolved_experts_implementation=(
                        model_plan.model_adapter.resolve_experts_implementation(
                            config.model.experts_implementation
                        )
                    ),
                    **resolved_runtime_args.resume_contract_kwargs(),
                    hook_instances=self.hook_manager.hooks,
                    interceptor_instances=self.interceptor_manager.interceptors,
                )
                if runtime.supports_checkpoint
                else None
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
            train_sample_budget = runtime.build_train_sample_budget(
                batch_contract=batch_contract,
                resolved_args=resolved_runtime_args,
                training_args=training_args,
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
                dataset_bundle = data_center.build_dataset_bundle(runtime.dataset_type)
            runtime.validate_dataset_bundle(
                dataset_bundle,
                resolved_args=resolved_runtime_args,
                training_args=training_args,
                resume_checkpoint=resume_checkpoint,
            )
            train_execution_fingerprint = str(
                dataset_bundle.train_execution_fingerprint or ""
            ).strip()
            if not train_execution_fingerprint:
                raise RuntimeError("ShaftDataCenter did not publish a train execution fingerprint.")
            train_execution_fingerprint = runtime.bind_execution_fingerprint(
                train_execution_fingerprint,
                resolved_args=resolved_runtime_args,
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
        # converges prepare/finalize failures; checkpointable local-HF finalize
        # owns its own long-timeout artifact-identity consensus. The raw loader
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
                raise RuntimeError("RL model loader must publish a resolved finetune plan.")
            validate_trainable_parameter_precision(artifacts.model, training_args)
            resolved_optimizer_plan = build_resolved_optimizer_plan(
                model=artifacts.model,
                args=training_args,
                finetune_plan=finetune_plan,
                model_adapter=artifacts.model_adapter,
                param_group_lrs=dict(config.train.param_group_lrs),
                no_decay_name_patterns=list(config.train.no_decay_name_patterns),
            )
            runtime_context = RLRuntimeContext(
                config=config,
                training_args=training_args,
                artifacts=artifacts,
                dataset_bundle=dataset_bundle,
                resolved_args=resolved_runtime_args,
                progress_manager=self.progress_manager,
                checkpoint_protocol=checkpoint_protocol,
            )
            input_options = runtime.input_options(
                runtime_context,
                sequence_execution_contract=sequence_execution_contract,
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
                input_builder=runtime.input_builder,
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
            if runtime.supports_checkpoint:
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
                    resolved_experts_implementation=(
                        model_plan.model_adapter.resolve_experts_implementation(
                            config.model.experts_implementation
                        )
                    ),
                    **resolved_runtime_args.resume_contract_kwargs(),
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
                "runtime": f"{type(runtime).__module__}.{type(runtime).__qualname__}",
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
            callbacks = [ShaftBatchingMetadataCallback(batching_metadata)]
            # Keep callback topology identical on every rank. Non-zero ranks receive
            # a manager without sinks, and ShaftProgressCallback is a no-op there.
            callbacks.append(ShaftProgressCallback(self.progress_manager))
            if self.hook_manager.hooks:
                callbacks.append(TrainerHookCallback(self.hook_manager))
            callbacks_or_none = callbacks or None
            trainer_inputs = runtime.build_trainer_inputs(runtime_context)
        trainer_spec = None
        with distributed_training_contract_stage(
            stage="trainer-prepare",
            fingerprints=lambda: {
                "runtime": f"{type(runtime).__module__}.{type(runtime).__qualname__}",
                "trainer": trainer_spec.fingerprint if trainer_spec is not None else "missing",
            },
        ):
            # Reference/value/reward model copies, reward construction, runtime
            # validation, and TRL config resolution are pure-local preparation.
            # Converge failures here before any peer enters Trainer/Accelerator.
            trainer_spec = runtime.prepare_trainer(
                runtime_context,
                trainer_inputs=trainer_inputs,
                callbacks=callbacks_or_none,
                finetune_plan=finetune_plan,
                resolved_optimizer_plan=resolved_optimizer_plan,
            )
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
        train_result = runtime.train(trainer, resume_checkpoint=resume_checkpoint)
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


def run_rl(config: RuntimeConfig) -> dict[str, Any]:
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
        pipeline_cls = PIPELINE_REGISTRY.get("shaft_rl")
        pipeline = pipeline_cls(config)
        pipeline._bootstrap_training_args = pipeline.build_training_args()
        pipeline.initialize_runtime()
    assert pipeline is not None
    assert pipeline.interceptor_manager is not None
    runner = ExecutionProxy(
        point="pipeline.rl.run",
        target=pipeline.run,
        interceptor_manager=pipeline.interceptor_manager,
    )
    try:
        # A rank-local before interceptor can fail before pipeline.run reaches
        # its first collective. Converge that readiness phase independently,
        # then invoke the collective-owning pipeline body outside the envelope.
        invocation = prepare_pipeline_call(
            runner,
            stage="rl-before-interceptors",
        )
        return runner.invoke(invocation)
    except BaseException as exc:
        pipeline.progress_manager.record_failure(str(exc) or type(exc).__name__)
        raise
    finally:
        pipeline.close()
