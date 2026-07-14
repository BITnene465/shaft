from __future__ import annotations

from contextlib import nullcontext
import json
import logging
from pathlib import Path
from typing import Any

from transformers import TrainingArguments

from shaft.algorithms.base import AlgorithmContext
from shaft.algorithms.registry import ALGORITHM_REGISTRY
from shaft.algorithms import sft as _sft  # noqa: F401
from shaft.config import RuntimeConfig
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
    sft_runtime_source_identity,
    validate_sft_cost_model_adapter,
    validate_sft_cost_dataset,
)
from shaft.model import (
    build_model_tokenizer_processor,
    resolve_model_plan,
    summarize_resolved_finetune_plan,
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
    load_batch_planning_state,
    publish_batching_run_metadata,
    validate_batching_resume_contract,
    validate_batch_planning_resume_contract,
)
from shaft.training.epoch_interval_callback import ShaftEpochIntervalCallback
from shaft.training.efficiency import (
    ShaftTrainingEfficiencyCallback,
    ShaftTrainingEfficiencyMonitor,
    ShaftTrainingEfficiencySnapshotInvalidationCallback,
    invalidate_training_efficiency_summary,
)
from shaft.training.online_eval import ShaftOnlineEvalRunner
from shaft.training.progress_callback import ShaftProgressCallback
from shaft.training.reproducibility import initialize_training_randomness
from shaft.training.checkpointing import (
    ensure_hf_export_layout,
    prune_root_output_layout,
    resolve_best_export_dir,
    resolve_resume_checkpoint,
    validate_resume_checkpoint,
    validate_training_state_policy,
)
from shaft.training.distributed import (
    all_gather_objects,
    barrier_if_distributed,
    is_rank_zero,
)
from shaft.training.topology import validate_training_topology

from .registry import PIPELINE_REGISTRY, register_pipeline
from .training_args import build_hf_training_args


logger = logging.getLogger(__name__)


def _checkpoint_global_step(checkpoint: str | Path) -> int:
    state_path = Path(checkpoint) / "trainer_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Trainer state must be a JSON object: {state_path}")
    global_step = int(payload.get("global_step", 0))
    if global_step < 0:
        raise ValueError(f"Trainer global_step must be >= 0: {state_path}")
    return global_step


def _build_planned_batch_sampler(
    *,
    config: RuntimeConfig,
    training_args: TrainingArguments,
    train_dataset: SFTDataset,
    train_schedule: Any,
    artifacts: Any,
    resume_checkpoint: str | None,
    resume_contract_fingerprint: str | None,
    batch_contract: ShaftBatchContract,
) -> tuple[ShaftPlannedBatchSampler, ShaftBatchPlanningSpec]:
    local_error: Exception | None = None
    provider: ShaftSFTSampleCostProvider | None = None
    spec: ShaftBatchPlanningSpec | None = None
    initial_state: ShaftBatchPlanningState | None = None
    preflight_fingerprint: str | None = None
    preflight_plan: ShaftBatchMicrobatchPlan | None = None
    try:
        if train_schedule is None or not hasattr(train_schedule, "ref_at"):
            raise TypeError(
                "Planned grouping requires Shaft's horizon-independent "
                "sample schedule."
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
            raise ValueError(
                "Planned grouping requires a resolved local token capacity."
            )
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
                expected_global_step=_checkpoint_global_step(resume_checkpoint),
                gradient_accumulation_steps=int(
                    training_args.gradient_accumulation_steps
                ),
                expected_resume_contract_fingerprint=resume_contract_fingerprint,
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
    except Exception as exc:  # noqa: BLE001 - every rank must reach startup gather
        local_error = exc

    status = {
        "ok": local_error is None,
        "error_type": None if local_error is None else type(local_error).__name__,
        "error": None if local_error is None else str(local_error),
        "contract_fingerprint": None if spec is None else spec.fingerprint,
        "preflight_fingerprint": preflight_fingerprint,
    }
    statuses = all_gather_objects(status)
    failures = [item for item in statuses if not bool(item.get("ok"))]
    if failures:
        if local_error is not None:
            raise local_error
        raise RuntimeError(
            f"Batch planning startup failed on a peer rank: {failures!r}."
        )
    contracts = {str(item["contract_fingerprint"]) for item in statuses}
    if len(contracts) != 1:
        raise ValueError(
            f"Batch-planning contract differs across data ranks: {statuses!r}."
        )
    plans = {str(item["preflight_fingerprint"]) for item in statuses}
    if int(training_args.world_size) > 1 and len(plans) != 1:
        raise ValueError(
            "First-buffer costs or plans differ across data ranks; "
            "verify immutable media mounts and data snapshots."
        )
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
            int(training_args.max_steps)
            * int(training_args.gradient_accumulation_steps)
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
        self.interceptor_manager = build_interceptor_manager(config.plugins.interceptors)
        self.progress_manager = build_progress_manager(config)

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
        algorithm_name = str(config.algorithm.name).strip().lower()
        if algorithm_name != "sft":
            raise ValueError(
                f"ShaftSFTPipeline only supports sft, got {algorithm_name!r}. "
                "Use ShaftRLHFPipeline for DPO/PPO."
            )
        invalidate_training_efficiency_summary(config.experiment.output_dir)
        validate_training_state_policy(config)
        validate_training_topology(config)
        initialize_training_randomness(
            seed=config.experiment.seed,
            full_determinism=config.train.full_determinism,
        )
        model_plan = resolve_model_plan(
            config,
            init_from_checkpoint=config.train.init_from_checkpoint,
        )
        training_args = self.build_training_args(resolved_model_plan=model_plan)
        batch_contract = build_batch_contract(
            config=config,
            training_args=training_args,
        )
        sequence_execution_contract = (
            model_plan.build_sequence_execution_contract(
                layout=batch_contract.layout,
                device_type="cpu" if bool(config.train.use_cpu) else "cuda",
                attention_implementation=config.model.attn_implementation,
                torch_dtype=config.model.torch_dtype,
                distributed_strategy=config.train.distributed.strategy,
                torch_compile=bool(getattr(training_args, "torch_compile", False)),
            )
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
        planned = batch_contract.is_planned
        planning_resume_contract = (
            build_batch_planning_resume_contract_fingerprint(
                config=config,
                training_args=training_args,
                batch_contract=batch_contract,
                sequence_execution_contract_fingerprint=(
                    sequence_execution_contract.fingerprint
                ),
            )
            if planned
            else None
        )
        resume_checkpoint = resolve_resume_checkpoint(
            config.train.resume_from_checkpoint,
            require_planning_state=planned,
        )
        if resume_checkpoint is not None:
            validate_resume_checkpoint(
                resume_checkpoint,
                finetune_mode=config.model.finetune.mode,
            )
            validate_batching_resume_contract(
                resume_checkpoint,
                expected_contract=batch_contract,
            )
            if planned:
                if planning_resume_contract is None:
                    raise RuntimeError(
                        "Batch-planning training contract fingerprint was not resolved."
                    )
                validate_batch_planning_resume_contract(
                    resume_checkpoint,
                    expected_resume_contract_fingerprint=planning_resume_contract,
                )
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

        with self._progress_phase("startup.data", label="data", message="loading"):
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
        train_execution_fingerprint = str(
            dataset_bundle.train_execution_fingerprint or ""
        ).strip()
        if not train_execution_fingerprint:
            raise RuntimeError(
                "ShaftDataCenter did not publish a train execution fingerprint."
            )
        if resume_checkpoint is not None:
            validate_batching_resume_contract(
                resume_checkpoint,
                expected_contract=batch_contract,
                expected_sample_execution_fingerprint=train_execution_fingerprint,
            )

        with self._progress_phase("startup.model", label="model", message="loading"):
            artifacts = build_model_tokenizer_processor(
                config,
                init_from_checkpoint=config.train.init_from_checkpoint,
                sequence_execution_contract=sequence_execution_contract,
                resolved_model_plan=model_plan,
            )
        artifacts.model_adapter.configure_sequence_execution(
            model=artifacts.model,
            contract=sequence_execution_contract,
        )
        artifacts.model_adapter.validate_sequence_execution(
            model=artifacts.model,
            contract=sequence_execution_contract,
        )
        align_model_generation_config(
            artifacts.model,
            tokenizer=artifacts.tokenizer,
            max_new_tokens=config.eval.max_new_tokens,
            do_sample=config.eval.do_sample,
            temperature=config.eval.temperature,
            repetition_penalty=1.0,
        )
        finetune_plan = getattr(artifacts, "finetune_plan", None)
        if finetune_plan is not None:
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
                resume_contract_fingerprint=planning_resume_contract,
                batch_contract=batch_contract,
            )
            train_sampler = None

        batching_metadata = build_batching_run_metadata(
            config=config,
            training_args=training_args,
            planning_spec=planning_spec,
            batch_contract=batch_contract,
            sample_execution_fingerprint=train_execution_fingerprint,
        )
        publish_batching_run_metadata(config.experiment.output_dir, batching_metadata)
        logger.info("[batching-metadata] %s", batching_metadata.to_dict())

        eval_dataset: Any = dataset_bundle.eval_dataset
        use_named_eval_datasets = bool(
            config.eval.enabled
            and dataset_bundle.eval_datasets_by_name
            and config.eval.datasets
            and (
                config.eval.loss_metrics_enabled
                or config.eval.online_metrics_enabled
                or config.eval.metric_for_best_model
                in {"eval_final_loss", "eval_final_score"}
            )
        )
        if use_named_eval_datasets:
            eval_dataset = dataset_bundle.eval_datasets_by_name

        resume_global_step = (
            0 if resume_checkpoint is None else _checkpoint_global_step(resume_checkpoint)
        )
        efficiency_enabled = bool(config.train.efficiency.enabled)
        efficiency_monitor: ShaftTrainingEfficiencyMonitor | None = None
        if efficiency_enabled:
            source_identity = sft_runtime_source_identity(train_dataset)
            efficiency_monitor = ShaftTrainingEfficiencyMonitor.from_checkpoint(
                output_dir=config.experiment.output_dir,
                checkpoint_dir=resume_checkpoint,
                checkpoint_global_step=resume_global_step,
                device_timing=(
                    str(config.train.efficiency.device_timing).strip().lower() != "off"
                ),
                persist=bool(config.train.efficiency.persist),
                contract=ShaftTrainingEfficiencyContract(
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
                    gradient_accumulation_steps=(
                        training_args.gradient_accumulation_steps
                    ),
                    max_length=config.data.max_length,
                    min_pixels=config.data.min_pixels,
                    max_pixels=config.data.max_pixels,
                    optimizer_name=config.train.optimizer_name,
                    scheduler_name=config.train.scheduler_name,
                    learning_rate=config.train.learning_rate,
                    source_fingerprint=source_identity.fingerprint,
                    source_contract_complete=source_identity.complete,
                    sample_execution_fingerprint=train_execution_fingerprint,
                    software_fingerprint=training_software_fingerprint(),
                    hardware_fingerprint=training_hardware_fingerprint(
                        training_args.device
                    ),
                    measurement_protocol="shaft-efficiency-optimizer-frame-v2",
                    timing_mode=(
                        "cuda_optimizer_frame"
                        if str(config.train.efficiency.device_timing).strip().lower()
                        != "off"
                        and training_args.device.type == "cuda"
                        else "host_optimizer_frame"
                    ),
                    batch_contract_fingerprint=batch_contract.fingerprint,
                    sequence_contract_fingerprint=(
                        sequence_execution_contract.fingerprint
                    ),
                ),
            )
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
                    gradient_accumulation_steps=int(
                        training_args.gradient_accumulation_steps
                    ),
                    resume_contract_fingerprint=planning_resume_contract,
                )
            )
        if efficiency_monitor is not None:
            callbacks.append(ShaftTrainingEfficiencyCallback(efficiency_monitor))
        else:
            callbacks.append(ShaftTrainingEfficiencySnapshotInvalidationCallback())
        if self.progress_manager.enabled:
            callbacks.append(ShaftProgressCallback(self.progress_manager))
        if (
            (
                config.eval.enabled
                and config.eval.eval_strategy == "epoch"
                and int(config.eval.epoch_interval) > 1
            )
            or (
                config.train.save_strategy == "epoch"
                and int(config.train.save_epoch_interval) > 1
            )
        ):
            callbacks.append(
                ShaftEpochIntervalCallback(
                    eval_epoch_interval=int(config.eval.epoch_interval),
                    save_epoch_interval=int(config.train.save_epoch_interval),
                )
            )
        hook_manager = build_hook_manager(config.plugins.hooks)
        if hook_manager.hooks:
            callbacks.append(TrainerHookCallback(hook_manager))

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
            min_pixels=config.data.min_pixels,
            max_pixels=config.data.max_pixels,
            max_length=config.data.max_length,
            add_eos_token=config.data.add_eos_token,
            include_targets_in_inputs=True,
            include_metadata=False,
            loss_scale_name=config.train.loss_scale,
            layout="padded",
            packing_mode="none",
            collect_stats=False,
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
                    min_pixels=config.data.min_pixels,
                    max_pixels=config.data.max_pixels,
                    max_length=config.data.max_length,
                    add_eos_token=config.data.add_eos_token,
                    include_targets_in_inputs=False,
                    include_metadata=True,
                    padding_side="left",
                    collect_stats=False,
                ),
                progress_manager=self.progress_manager,
            )

        algorithm = ALGORITHM_REGISTRY.get(algorithm_name)()
        trainer = algorithm.build_trainer(
            context=AlgorithmContext(params=dict(config.algorithm.params)),
            train_config=config.train,
            model=artifacts.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if config.eval.enabled else None,
            train_sampler=train_sampler,
            train_batch_sampler=train_batch_sampler,
            processing_class=artifacts.processor,
            data_collator=collator,
            eval_data_collator=eval_collator,
            callbacks=callbacks,
            online_eval_runner=online_eval_runner,
            eval_config=config.eval,
            model_adapter=artifacts.model_adapter,
            finetune_plan=finetune_plan,
            efficiency_monitor=efficiency_monitor,
        )
        if efficiency_monitor is not None:
            efficiency_monitor.bind_update_applied_provider(
                lambda: not bool(trainer.accelerator.optimizer_step_was_skipped)
            )

        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        finalize_efficiency = getattr(trainer, "finalize_training_efficiency", None)
        if callable(finalize_efficiency):
            train_result.metrics.update(finalize_efficiency())
        barrier_if_distributed()
        if config.train.save_final_model:
            best_export_dir = resolve_best_export_dir(config.experiment.output_dir)
            trainer.save_model(output_dir=str(best_export_dir))
            if is_rank_zero():
                ensure_hf_export_layout(
                    best_export_dir,
                    finetune_mode=config.model.finetune.mode,
                    model_meta=artifacts.model_adapter,
                )
        if config.train.save_final_state:
            trainer.save_state()
        if is_rank_zero():
            prune_root_output_layout(config.experiment.output_dir)
        barrier_if_distributed()
        return dict(train_result.metrics or {})


def run_sft(config: RuntimeConfig) -> dict[str, Any]:
    pipeline_cls = PIPELINE_REGISTRY.get("shaft_sft")
    pipeline = pipeline_cls(config)
    runner = ExecutionProxy(
        point="pipeline.sft.run",
        target=pipeline.run,
        interceptor_manager=pipeline.interceptor_manager,
    )
    try:
        return runner()
    except BaseException as exc:
        pipeline.progress_manager.record_failure(str(exc) or type(exc).__name__)
        raise
    finally:
        pipeline.close()
