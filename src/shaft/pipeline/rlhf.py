from __future__ import annotations

from contextlib import nullcontext
from functools import partial
import logging
from typing import Any

from transformers import TrainingArguments

from shaft.algorithms.base import AlgorithmContext
from shaft.algorithms.registry import ALGORITHM_REGISTRY
from shaft.algorithms import dpo as _dpo  # noqa: F401
from shaft.algorithms import grpo as _grpo  # noqa: F401
from shaft.algorithms import ppo as _ppo  # noqa: F401
from shaft.config import RuntimeConfig
from shaft.data import (
    DPOCollator,
    DPODataset,
    GRPODataset,
    PPOCollator,
    PPODataset,
    SFTCollator,
    SFTDataset,
    ShaftDataCenter,
)
from shaft.model import build_model_tokenizer_processor, resolve_model_plan
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
    publish_batching_run_metadata,
    validate_batching_resume_contract,
)
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
from shaft.training.distributed import barrier_if_distributed
from shaft.training.distributed import is_rank_zero
from shaft.training.topology import validate_training_topology
from shaft.training.efficiency import invalidate_training_efficiency_summary

from .registry import PIPELINE_REGISTRY, register_pipeline
from .training_args import build_hf_training_args

logger = logging.getLogger(__name__)


@register_pipeline("shaft_rlhf")
class ShaftRLHFPipeline:
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

    def _build_collator(self, algorithm_name: str, *, artifacts):
        common_kwargs = {
            "model_adapter": artifacts.model_adapter,
            "template": artifacts.template,
            "processor": artifacts.processor,
            "tokenizer": artifacts.tokenizer,
            "min_pixels": self.config.data.min_pixels,
            "max_pixels": self.config.data.max_pixels,
            "max_length": self.config.data.max_length,
            "add_eos_token": self.config.data.add_eos_token,
        }
        if algorithm_name == "dpo":
            return DPOCollator(**common_kwargs)
        if algorithm_name == "ppo":
            return PPOCollator(**common_kwargs)
        raise ValueError(f"Unsupported RLHF algorithm: {algorithm_name!r}.")

    def run(self) -> dict[str, Any]:
        config = self.config
        algorithm_name = str(config.algorithm.name).strip().lower()
        if algorithm_name not in {"dpo", "ppo", "grpo"}:
            raise ValueError(
                f"ShaftRLHFPipeline only supports dpo/ppo/grpo, got algorithm={algorithm_name!r}."
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
        resume_checkpoint = resolve_resume_checkpoint(
            config.train.resume_from_checkpoint
        )
        if resume_checkpoint is not None:
            if algorithm_name == "ppo":
                raise ValueError(
                    "TRL PPOTrainer does not support resume_from_checkpoint "
                    "in current Shaft pipeline."
                )
            validate_resume_checkpoint(
                resume_checkpoint,
                finetune_mode=config.model.finetune.mode,
            )
            validate_batching_resume_contract(
                resume_checkpoint,
                expected_contract=batch_contract,
            )
        batching_metadata = build_batching_run_metadata(
            config=config,
            training_args=training_args,
            batch_contract=batch_contract,
        )
        publish_batching_run_metadata(config.experiment.output_dir, batching_metadata)
        logger.info("[batching-metadata] %s", batching_metadata.to_dict())
        with self._progress_phase(
            "startup.model",
            label="model",
            message="loading",
        ):
            artifacts = build_model_tokenizer_processor(
                config,
                init_from_checkpoint=config.train.init_from_checkpoint,
                resolved_model_plan=model_plan,
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
        if algorithm_name == "dpo":
            dataset_cls = DPODataset
        elif algorithm_name == "ppo":
            dataset_cls = PPODataset
        else:
            dataset_cls = SFTDataset
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
            dataset_bundle = data_center.build_dataset_bundle(dataset_cls)
        train_dataset = dataset_bundle.train_dataset
        eval_dataset: Any = dataset_bundle.eval_dataset
        use_named_eval_datasets = bool(
            algorithm_name == "grpo"
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
        if algorithm_name == "grpo":
            grpo_dataset_kwargs = {
                "template": artifacts.template,
                "image_preprocessor": partial(
                    artifacts.model_adapter.prepare_rollout_image,
                    min_pixels=config.data.min_pixels,
                    max_pixels=config.data.max_pixels,
                ),
            }
            train_dataset = GRPODataset(train_dataset, **grpo_dataset_kwargs)
            if eval_dataset is not None and not (
                config.eval.online_metrics_enabled and use_named_eval_datasets
            ):
                eval_dataset = GRPODataset(eval_dataset, **grpo_dataset_kwargs)
        hook_manager = build_hook_manager(config.plugins.hooks)
        callbacks = [ShaftBatchingMetadataCallback(batching_metadata)]
        if self.progress_manager.enabled:
            callbacks.append(ShaftProgressCallback(self.progress_manager))
        if hook_manager.hooks:
            callbacks.append(TrainerHookCallback(hook_manager))
        callbacks_or_none = callbacks or None
        online_eval_runner = None
        if algorithm_name == "grpo" and config.eval.enabled and config.eval.online_metrics_enabled:
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

        algorithm_cls = ALGORITHM_REGISTRY.get(algorithm_name)
        algorithm = algorithm_cls()
        processing_class = artifacts.processor
        algorithm_extra_kwargs: dict[str, Any] = {}
        if algorithm_name == "ppo":
            algorithm_extra_kwargs["model_meta"] = artifacts.model_meta
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
            **algorithm_extra_kwargs,
        }
        if algorithm_name == "dpo":
            trainer_kwargs["train_sampler"] = dataset_bundle.train_sampler
        if algorithm_name == "grpo":
            if dataset_bundle.train_sampler is None:
                raise RuntimeError("GRPO requires a Shaft sample plan from the data center.")
            trainer_kwargs["sample_plan"] = dataset_bundle.train_sampler.plan
            trainer_kwargs["online_eval_runner"] = online_eval_runner
            trainer_kwargs["eval_config"] = config.eval
        if algorithm_name != "grpo":
            trainer_kwargs["data_collator"] = self._build_collator(algorithm_name, artifacts=artifacts)
        trainer = algorithm.build_trainer(
            **trainer_kwargs,
        )

        if algorithm_name == "ppo":
            train_result = trainer.train()
        else:
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
        if train_result is not None and hasattr(train_result, "metrics"):
            return dict(train_result.metrics or {})
        log_history = getattr(getattr(trainer, "state", None), "log_history", None)
        if isinstance(log_history, list):
            for entry in reversed(log_history):
                if isinstance(entry, dict):
                    return dict(entry)
        return {}


def run_rlhf(config: RuntimeConfig) -> dict[str, Any]:
    pipeline_cls = PIPELINE_REGISTRY.get("shaft_rlhf")
    pipeline = pipeline_cls(config)
    runner = ExecutionProxy(
        point="pipeline.rlhf.run",
        target=pipeline.run,
        interceptor_manager=pipeline.interceptor_manager,
    )
    try:
        return runner()
    except BaseException as exc:
        pipeline.progress_manager.record_failure(
            str(exc) or type(exc).__name__
        )
        raise
    finally:
        pipeline.close()
