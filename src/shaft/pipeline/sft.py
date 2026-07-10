from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any
import uuid

from transformers import TrainingArguments

from shaft.algorithms.base import AlgorithmContext
from shaft.algorithms.registry import ALGORITHM_REGISTRY
from shaft.algorithms import sft as _sft  # noqa: F401
from shaft.config import RuntimeConfig
from shaft.data import (
    SFTCollator,
    SFTDataset,
    ShaftBatchPlanningSignature,
    ShaftCostAwareSampler,
    ShaftDataCenter,
    ShaftFixedBatchPlanningSpec,
    ShaftMMapCostPlanProvider,
    ShaftSampleCostProvider,
    ShaftSamplePlan,
    ShaftSFTSampleCostProvider,
    ShaftCostPlanMaterialization,
    cost_plan_reference_path,
    load_cost_plan_manifest,
    materialize_cost_plan,
    resolve_cost_plan_cache_dir,
    sft_cost_planning_source_fingerprint,
    validate_sft_cost_model_adapter,
    validate_sft_cost_planning_dataset,
    write_cost_plan_reference,
)
from shaft.model import (
    ModelArtifacts,
    build_model_tokenizer_processor,
    resolve_model_adapter_from_config,
)
from shaft.model import summarize_resolved_finetune_plan, write_resolved_finetune_summary
from shaft.model.generation import align_model_generation_config
from shaft.plugins import (
    ExecutionProxy,
    TrainerHookCallback,
    build_hook_manager,
    build_interceptor_manager,
)
from shaft.training.epoch_interval_callback import ShaftEpochIntervalCallback
from shaft.training.batch_planning import (
    ShaftBatchPlanningCallback,
    batch_planning_signature_path,
    validate_batch_planning_resume,
    validate_batch_planning_resume_geometry,
    write_batch_planning_signature,
)
from shaft.training.online_eval import ShaftOnlineEvalRunner
from shaft.training.progress_callback import ShaftProgressCallback
from shaft.training.checkpointing import (
    ensure_hf_export_layout,
    prune_root_output_layout,
    resolve_best_export_dir,
    resolve_resume_checkpoint,
    validate_resume_checkpoint,
    validate_training_state_policy,
)
from shaft.training.distributed import barrier_if_distributed
from shaft.training.distributed import all_gather_objects
from shaft.training.distributed import broadcast_object_from_rank_zero
from shaft.training.distributed import is_rank_zero
from shaft.training.topology import validate_training_topology

from .registry import PIPELINE_REGISTRY, register_pipeline
from .training_args import build_hf_training_args, resolve_step_sample_budget

logger = logging.getLogger(__name__)


def _restore_file_snapshot(path: Path, content: bytes | None) -> None:
    if content is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.rollback")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_cost_aware_run_metadata(
    *,
    output_dir: str | Path,
    materialization: ShaftCostPlanMaterialization,
    signature: ShaftBatchPlanningSignature,
) -> None:
    """Publish two root files transactionally, with reference as commit marker."""

    signature_path = batch_planning_signature_path(output_dir)
    reference_path = cost_plan_reference_path(output_dir)
    previous_signature = (
        signature_path.read_bytes() if signature_path.is_file() else None
    )
    previous_reference = (
        reference_path.read_bytes() if reference_path.is_file() else None
    )
    try:
        write_batch_planning_signature(output_dir, signature)
        write_cost_plan_reference(output_dir, materialization)
    except Exception:
        rollback_errors: list[str] = []
        for path, content in (
            (reference_path, previous_reference),
            (signature_path, previous_signature),
        ):
            try:
                _restore_file_snapshot(path, content)
            except Exception as rollback_error:  # noqa: BLE001
                rollback_errors.append(
                    f"{path}: {type(rollback_error).__name__}: {rollback_error}"
                )
        if rollback_errors:
            raise RuntimeError(
                "Cost-aware run metadata publish failed and rollback was incomplete: "
                f"{rollback_errors!r}."
            )
        raise


def _prepare_and_validate_cost_aware_startup(
    *,
    config: RuntimeConfig,
    plan: ShaftSamplePlan,
    train_dataset: SFTDataset,
    per_device_batch_size: int,
    data_world_size: int,
    gradient_accumulation_steps: int,
) -> tuple[ShaftFixedBatchPlanningSpec, Path]:
    """Fail before model loading when ranks do not share source/cache identity."""

    local_status: dict[str, Any]
    planning_spec: ShaftFixedBatchPlanningSpec | None = None
    local_exception: Exception | None = None
    try:
        planning_spec = ShaftFixedBatchPlanningSpec.from_plan(
            plan,
            per_device_batch_size=per_device_batch_size,
            data_world_size=data_world_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            planning_window=config.data.batching.planning_window,
            seed=config.experiment.seed,
            drop_last=False,
        )
        validate_sft_cost_planning_dataset(train_dataset)
        validate_sft_cost_model_adapter(resolve_model_adapter_from_config(config))
        local_status = {
            "ok": True,
            "planning_spec": planning_spec,
            "source_fingerprint": sft_cost_planning_source_fingerprint(train_dataset),
            "cache_path": str(
                resolve_cost_plan_cache_dir(
                    config.data.batching.cost_plan_cache_dir,
                    record_cache_dir=config.data.record_cache_dir,
                ).resolve()
            ),
        }
    except Exception as exc:  # noqa: BLE001 - every rank must enter the first collective
        local_exception = exc
        local_status = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    statuses = all_gather_objects(local_status)
    failures = [status for status in statuses if not status.get("ok")]
    if failures:
        if local_exception is not None:
            raise local_exception
        raise RuntimeError(
            "Distributed cost-aware preflight failed before model loading: "
            f"{failures!r}."
        )
    planning_specs = {status["planning_spec"] for status in statuses}
    if len(planning_specs) != 1:
        raise ValueError(
            "Distributed fixed batch planning spec differs across ranks: "
            f"{planning_specs!r}."
        )
    source_fingerprints = {
        str(status["source_fingerprint"]) for status in statuses
    }
    if len(source_fingerprints) != 1:
        raise ValueError(
            "Distributed SFT CostPlan source fingerprint differs across ranks: "
            f"{sorted(source_fingerprints)!r}."
        )
    cache_paths = {str(status["cache_path"]) for status in statuses}
    if len(cache_paths) != 1:
        raise ValueError(
            "Distributed CostPlan cache path differs across ranks: "
            f"{sorted(cache_paths)!r}."
        )
    cache_dir = Path(local_status["cache_path"])
    assert planning_spec is not None
    if int(data_world_size) <= 1:
        return planning_spec, cache_dir

    rank_zero_error: Exception | None = None
    rendezvous: dict[str, Any] | None = None
    if is_rank_zero():
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            probe_path = cache_dir / f".shaft-shared-cache-probe.{token}"
            with probe_path.open("xb") as handle:
                handle.write(token.encode("ascii"))
                handle.flush()
                os.fsync(handle.fileno())
            rendezvous = {
                "ok": True,
                "probe_path": str(probe_path),
                "token": token,
            }
        except Exception as exc:  # noqa: BLE001 - every rank must receive the failure
            rank_zero_error = exc
            rendezvous = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    rendezvous = broadcast_object_from_rank_zero(rendezvous)
    if not isinstance(rendezvous, dict) or not bool(rendezvous.get("ok")):
        if rank_zero_error is not None:
            raise rank_zero_error
        raise RuntimeError(
            "Rank-zero shared CostPlan cache probe failed: "
            f"{rendezvous!r}."
        )

    local_error: str | None = None
    try:
        probe_path = Path(str(rendezvous["probe_path"]))
        observed_token = probe_path.read_text(encoding="ascii")
        if observed_token != str(rendezvous["token"]):
            raise RuntimeError("shared cache probe token does not match")
    except Exception as exc:  # noqa: BLE001 - acknowledge failures collectively
        local_error = f"{type(exc).__name__}: {exc}"
    statuses = all_gather_objects({"ok": local_error is None, "error": local_error})
    if is_rank_zero():
        try:
            Path(str(rendezvous["probe_path"])).unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "failed to remove shared CostPlan cache probe %s",
                rendezvous["probe_path"],
                exc_info=True,
            )
    failures = [status for status in statuses if not status.get("ok")]
    if failures:
        raise RuntimeError(
            "CostPlan cache is not the same readable filesystem on every rank: "
            f"{failures!r}. Use the same absolute shared cache mount."
        )
    return planning_spec, cache_dir


def _build_shared_cost_plan_provider(
    *,
    config: RuntimeConfig,
    plan: ShaftSamplePlan,
    train_dataset: SFTDataset,
    artifacts: ModelArtifacts,
    cache_dir: Path,
) -> tuple[ShaftMMapCostPlanProvider, ShaftCostPlanMaterialization | None]:
    """Materialize on global rank zero, then map the same immutable plan everywhere."""

    materialization: ShaftCostPlanMaterialization | None = None
    rank_zero_error: Exception | None = None
    rendezvous: dict[str, Any] | None = None
    if is_rank_zero():
        started = time.perf_counter()
        try:
            runtime_provider: ShaftSampleCostProvider = ShaftSFTSampleCostProvider(
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
                image_size_cache_size=config.data.batching.image_size_cache_size,
            )
            materialization = materialize_cost_plan(
                plan,
                cost_provider=runtime_provider,
                cache_dir=cache_dir,
            )
            rendezvous = {
                "ok": True,
                "manifest_path": str(materialization.manifest_path),
                "manifest_fingerprint": materialization.provider.manifest.fingerprint,
            }
            logger.info(
                "[cost-plan-cache] hit=%s samples=%s bytes=%s elapsed_seconds=%.6f "
                "materialize_seconds=%.6f manifest=%s",
                materialization.cache_hit,
                len(plan),
                materialization.data_bytes,
                time.perf_counter() - started,
                materialization.elapsed_seconds,
                materialization.manifest_path,
            )
        except Exception as exc:  # noqa: BLE001 - the failure must reach every rank
            rank_zero_error = exc
            rendezvous = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        finally:
            if materialization is not None:
                materialization.provider.close()

    rendezvous = broadcast_object_from_rank_zero(rendezvous)
    if not isinstance(rendezvous, dict) or not bool(rendezvous.get("ok")):
        if rank_zero_error is not None:
            raise rank_zero_error
        raise RuntimeError(
            "Rank-zero CostPlan materialization failed: "
            f"{rendezvous.get('error_type', 'Error') if isinstance(rendezvous, dict) else 'Error'}: "
            f"{rendezvous.get('error', 'missing rendezvous payload') if isinstance(rendezvous, dict) else rendezvous}"
        )

    provider: ShaftMMapCostPlanProvider | None = None
    local_error: str | None = None
    try:
        provider = load_cost_plan_manifest(
            str(rendezvous["manifest_path"]),
            plan=plan,
            expected_manifest_fingerprint=str(rendezvous["manifest_fingerprint"]),
            verify_checksum=False,
        )
    except Exception as exc:  # noqa: BLE001 - all ranks must acknowledge mmap readiness
        local_error = f"{type(exc).__name__}: {exc}"
    load_statuses = all_gather_objects(
        {
            "ok": local_error is None,
            "error": local_error,
        }
    )
    failed_statuses = [status for status in load_statuses if not status.get("ok")]
    if failed_statuses:
        if provider is not None:
            provider.close()
        raise RuntimeError(
            "Shared CostPlan is not readable on every rank: "
            f"{failed_statuses!r}. Use the same absolute shared cache mount."
        )
    assert provider is not None
    return provider, materialization


@register_pipeline("shaft_sft")
class ShaftSFTPipeline:
    """HF-first SFT pipeline.

    The pipeline only coordinates modules; business semantics stay in data/model/algorithms.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.interceptor_manager = build_interceptor_manager(config.plugins.interceptors)
        self._cost_plan_provider: ShaftMMapCostPlanProvider | None = None

    def close(self) -> None:
        if self._cost_plan_provider is not None:
            self._cost_plan_provider.close()
            self._cost_plan_provider = None

    def build_training_args(self) -> TrainingArguments:
        return build_hf_training_args(self.config)

    def run(self) -> dict[str, Any]:
        config = self.config
        algorithm_name = str(config.algorithm.name).strip().lower()
        if algorithm_name != "sft":
            raise ValueError(
                f"ShaftSFTPipeline only supports sft, got algorithm={algorithm_name!r}. "
                "Use ShaftRLHFPipeline for dpo/ppo."
            )
        validate_training_state_policy(config)
        validate_training_topology(config)
        training_args = self.build_training_args()
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
        cost_aware_batching = config.data.batching.strategy == "cost_aware"
        planning_spec: ShaftFixedBatchPlanningSpec | None = None
        cost_plan_cache_dir: Path | None = None
        if cost_aware_batching:
            if train_sampler is None or not hasattr(train_sampler, "plan"):
                raise TypeError(
                    "Cost-aware SFT batching requires the Shaft sample-plan sampler."
                )
            planning_spec, cost_plan_cache_dir = (
                _prepare_and_validate_cost_aware_startup(
                    config=config,
                    plan=train_sampler.plan,
                    train_dataset=train_dataset,
                    per_device_batch_size=training_args.per_device_train_batch_size,
                    data_world_size=training_args.world_size,
                    gradient_accumulation_steps=(
                        training_args.gradient_accumulation_steps
                    ),
                )
            )

        resume_checkpoint = resolve_resume_checkpoint(config.train.resume_from_checkpoint)
        if resume_checkpoint is not None:
            validate_resume_checkpoint(
                resume_checkpoint,
                finetune_mode=config.model.finetune.mode,
            )
            if cost_aware_batching:
                assert train_sampler is not None and hasattr(train_sampler, "plan")
                assert planning_spec is not None
                validate_batch_planning_resume_geometry(
                    resume_checkpoint,
                    expected=planning_spec,
                )

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
                write_resolved_finetune_summary(config.experiment.output_dir, freeze_summary)
                logger.info("[startup] resolved freeze summary: %s", freeze_summary.to_log_dict())
        batch_planning_signature = None
        cost_plan_materialization: ShaftCostPlanMaterialization | None = None
        if cost_aware_batching:
            assert train_sampler is not None and hasattr(train_sampler, "plan")
            assert planning_spec is not None
            assert cost_plan_cache_dir is not None
            cost_provider, cost_plan_materialization = _build_shared_cost_plan_provider(
                config=config,
                plan=train_sampler.plan,
                train_dataset=train_dataset,
                artifacts=artifacts,
                cache_dir=cost_plan_cache_dir,
            )
            self._cost_plan_provider = cost_provider
            train_sampler = ShaftCostAwareSampler(
                train_sampler.plan,
                cost_provider=cost_provider,
                spec=planning_spec,
            )
            batch_planning_signature = train_sampler.signature

        if resume_checkpoint is not None and batch_planning_signature is not None:
            validate_batch_planning_resume(
                resume_checkpoint,
                expected=batch_planning_signature,
            )
        if batch_planning_signature is not None:
            publish_error: Exception | None = None
            publish_status: dict[str, Any] | None = None
            if is_rank_zero():
                assert cost_plan_materialization is not None
                try:
                    _publish_cost_aware_run_metadata(
                        output_dir=config.experiment.output_dir,
                        materialization=cost_plan_materialization,
                        signature=batch_planning_signature,
                    )
                    publish_status = {"ok": True}
                except Exception as exc:  # noqa: BLE001 - broadcast before raising
                    publish_error = exc
                    publish_status = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
            publish_status = broadcast_object_from_rank_zero(publish_status)
            if not isinstance(publish_status, dict) or not publish_status.get("ok"):
                if publish_error is not None:
                    raise publish_error
                raise RuntimeError(
                    "Rank-zero cost-aware run metadata publish failed: "
                    f"{publish_status!r}."
                )
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
        hook_manager = build_hook_manager(config.plugins.hooks)
        callbacks = []
        if batch_planning_signature is not None:
            callbacks.append(ShaftBatchPlanningCallback(batch_planning_signature))
        if config.progress.enabled:
            callbacks.append(
                ShaftProgressCallback(
                    leave=config.progress.leave,
                    mininterval=config.progress.mininterval,
                )
            )
        if (
            (config.eval.enabled and config.eval.eval_strategy == "epoch" and int(config.eval.epoch_interval) > 1)
            or (config.train.save_strategy == "epoch" and int(config.train.save_epoch_interval) > 1)
        ):
            callbacks.append(
                ShaftEpochIntervalCallback(
                    eval_epoch_interval=int(config.eval.epoch_interval),
                    save_epoch_interval=int(config.train.save_epoch_interval),
                )
            )
        if hook_manager.hooks:
            callbacks.append(TrainerHookCallback(hook_manager))
        callbacks_or_none = callbacks or None
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
                progress_enabled=config.progress.enabled,
                progress_leave=config.progress.leave,
                progress_mininterval=config.progress.mininterval,
            )
        algorithm_cls = ALGORITHM_REGISTRY.get(algorithm_name)
        algorithm = algorithm_cls()
        trainer = algorithm.build_trainer(
            context=AlgorithmContext(params=dict(config.algorithm.params)),
            train_config=config.train,
            model=artifacts.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if config.eval.enabled else None,
            train_sampler=train_sampler,
            processing_class=artifacts.processor,
            data_collator=collator,
            callbacks=callbacks_or_none,
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
    finally:
        pipeline.close()
