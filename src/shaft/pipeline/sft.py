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
    ShaftBoundedBatchSampler,
    ShaftBoundedBatchPlanner,
    ShaftBoundedBatchingSpec,
    ShaftBoundedBatchingState,
    ShaftDataCenter,
    ShaftSFTSampleCostProvider,
    validate_sft_cost_model_adapter,
    validate_sft_cost_dataset,
)
from shaft.model import (
    build_model_tokenizer_processor,
    summarize_resolved_finetune_plan,
    write_resolved_finetune_summary,
)
from shaft.model.generation import align_model_generation_config
from shaft.observability import build_progress_manager
from shaft.plugins import (
    ExecutionProxy,
    TrainerHookCallback,
    build_hook_manager,
    build_interceptor_manager,
)
from shaft.training.batch_planning import (
    ShaftBatchingMetadataCallback,
    ShaftBoundedBatchingCallback,
    build_bounded_resume_contract_fingerprint,
    build_batching_run_metadata,
    load_bounded_batching_state,
    publish_batching_run_metadata,
)
from shaft.training.epoch_interval_callback import ShaftEpochIntervalCallback
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
from .training_args import build_hf_training_args, resolve_step_sample_budget


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


def _build_bounded_batch_sampler(
    *,
    config: RuntimeConfig,
    training_args: TrainingArguments,
    train_dataset: SFTDataset,
    train_schedule: Any,
    artifacts: Any,
    resume_checkpoint: str | None,
) -> tuple[ShaftBoundedBatchSampler, ShaftBoundedBatchingSpec]:
    local_error: Exception | None = None
    provider: ShaftSFTSampleCostProvider | None = None
    spec: ShaftBoundedBatchingSpec | None = None
    initial_state: ShaftBoundedBatchingState | None = None
    preflight_fingerprint: str | None = None
    try:
        if train_schedule is None or not hasattr(train_schedule, "ref_at"):
            raise TypeError(
                "bounded_cost_aware SFT requires Shaft's horizon-independent "
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
            cache_size=config.data.batching.cost_cache_size,
        )
        max_padded_tokens = config.data.batching.max_padded_tokens
        max_samples = config.data.batching.max_samples_per_microbatch
        if max_padded_tokens is None or max_samples is None:
            raise ValueError(
                "bounded_cost_aware requires normalized max_padded_tokens and "
                "max_samples_per_microbatch."
            )
        spec = ShaftBoundedBatchingSpec(
            data_world_size=max(int(training_args.world_size), 1),
            buffer_size=int(config.data.batching.buffer_size),
            max_samples_per_microbatch=int(max_samples),
            max_padded_tokens=int(max_padded_tokens),
            max_vision_patches=config.data.batching.max_vision_patches,
            seed=int(config.experiment.seed),
            sample_schedule_fingerprint=str(train_schedule.fingerprint),
            cost_fingerprint=str(provider.fingerprint),
        )
        if resume_checkpoint is not None:
            resume_contract_fingerprint = build_bounded_resume_contract_fingerprint(
                config=config,
                training_args=training_args,
            )
            initial_state = load_bounded_batching_state(
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
                        "Bounded batching resume detected changed cost for buffered "
                        f"draw_id={buffered.sample_ref.context.draw_id}; source media, "
                        "prompt transforms, or cost semantics changed in place."
                    )
        if int(training_args.world_size) > 1:
            preflight = ShaftBoundedBatchPlanner(
                schedule=train_schedule,
                cost_provider=provider,
                spec=spec,
                state=initial_state,
            ).next_global_microbatch()
            preflight_fingerprint = preflight.fingerprint
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
            f"Bounded batching startup failed on a peer rank: {failures!r}."
        )
    contracts = {str(item["contract_fingerprint"]) for item in statuses}
    if len(contracts) != 1:
        raise ValueError(
            f"Bounded batching contract differs across data ranks: {statuses!r}."
        )
    plans = {str(item["preflight_fingerprint"]) for item in statuses}
    if int(training_args.world_size) > 1 and len(plans) != 1:
        raise ValueError(
            "Bounded batching first-buffer costs or plans differ across data ranks; "
            "verify immutable media mounts and data snapshots."
        )
    assert provider is not None and spec is not None
    if resume_checkpoint is not None:
        # The custom sampler already starts at the committed optimizer boundary.
        # HF must not replay its own local-batch skip on top of that state.
        training_args.ignore_data_skip = True

    sampler = ShaftBoundedBatchSampler(
        train_schedule,
        cost_provider=provider,
        spec=spec,
        global_microstep_count=(
            int(training_args.max_steps)
            * int(training_args.gradient_accumulation_steps)
        ),
        planning_frame_size=int(training_args.gradient_accumulation_steps),
        initial_state=initial_state,
    )
    return sampler, spec


@register_pipeline("shaft_sft")
class ShaftSFTPipeline:
    """HF-first SFT pipeline with a bounded, lazy cost-aware data path."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.interceptor_manager = build_interceptor_manager(config.plugins.interceptors)
        self.progress_manager = build_progress_manager(config)

    def close(self) -> None:
        self.progress_manager.close()

    def build_training_args(self) -> TrainingArguments:
        return build_hf_training_args(self.config)

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
        validate_training_state_policy(config)
        validate_training_topology(config)
        initialize_training_randomness(
            seed=config.experiment.seed,
            full_determinism=config.train.full_determinism,
        )
        training_args = self.build_training_args()
        bounded = config.data.batching.strategy == "bounded_cost_aware"
        logger.info(
            "[batching] strategy=%s buffer_size=%s cost_cache_size=%s "
            "per_device_batch=%s data_world_size=%s gradient_accumulation=%s "
            "max_samples=%s max_padded_tokens=%s max_vision_patches=%s "
            "min_pixels=%s max_pixels=%s",
            config.data.batching.strategy,
            config.data.batching.buffer_size if bounded else None,
            config.data.batching.cost_cache_size if bounded else None,
            training_args.per_device_train_batch_size,
            training_args.world_size,
            training_args.gradient_accumulation_steps,
            config.data.batching.max_samples_per_microbatch,
            config.data.batching.max_padded_tokens,
            config.data.batching.max_vision_patches,
            config.data.min_pixels,
            config.data.max_pixels,
        )

        with self._progress_phase("startup.data", label="data", message="loading"):
            data_center = ShaftDataCenter(
                config.data,
                seed=config.experiment.seed,
                train_sample_budget=resolve_step_sample_budget(
                    config,
                    world_size=training_args.world_size,
                ),
            )
            dataset_bundle = data_center.build_dataset_bundle(SFTDataset)
        train_dataset = dataset_bundle.train_dataset
        train_sampler = dataset_bundle.train_sampler
        train_schedule = dataset_bundle.train_schedule

        resume_checkpoint = resolve_resume_checkpoint(
            config.train.resume_from_checkpoint,
            require_bounded_state=bounded,
        )
        if resume_checkpoint is not None:
            validate_resume_checkpoint(
                resume_checkpoint,
                finetune_mode=config.model.finetune.mode,
            )

        with self._progress_phase("startup.model", label="model", message="loading"):
            artifacts = build_model_tokenizer_processor(
                config,
                init_from_checkpoint=config.train.init_from_checkpoint,
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

        train_batch_sampler: ShaftBoundedBatchSampler | None = None
        bounded_spec: ShaftBoundedBatchingSpec | None = None
        if bounded:
            train_batch_sampler, bounded_spec = _build_bounded_batch_sampler(
                config=config,
                training_args=training_args,
                train_dataset=train_dataset,
                train_schedule=train_schedule,
                artifacts=artifacts,
                resume_checkpoint=resume_checkpoint,
            )
            train_sampler = None

        batching_metadata = build_batching_run_metadata(
            config=config,
            training_args=training_args,
            bounded_spec=bounded_spec,
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

        callbacks: list[Any] = [ShaftBatchingMetadataCallback(batching_metadata)]
        if train_batch_sampler is not None and bounded_spec is not None:
            bounded_resume_contract = build_bounded_resume_contract_fingerprint(
                config=config,
                training_args=training_args,
            )
            callbacks.append(
                ShaftBoundedBatchingCallback(
                    train_batch_sampler,
                    bounded_spec,
                    gradient_accumulation_steps=int(
                        training_args.gradient_accumulation_steps
                    ),
                    resume_contract_fingerprint=bounded_resume_contract,
                )
            )
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
            callbacks=callbacks,
            online_eval_runner=online_eval_runner,
            eval_config=config.eval,
            model_adapter=artifacts.model_adapter,
            finetune_plan=finetune_plan,
        )

        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
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
