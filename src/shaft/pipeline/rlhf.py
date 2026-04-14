from __future__ import annotations

from typing import Any

from transformers import TrainingArguments

from shaft.algorithms import ALGORITHM_REGISTRY, AlgorithmContext
from shaft.algorithms import dpo as _dpo  # noqa: F401
from shaft.algorithms import ppo as _ppo  # noqa: F401
from shaft.config import RuntimeConfig
from shaft.data import (
    DPOCollator,
    DPODataset,
    PPOCollator,
    PPODataset,
    MixedDatasetBuilder,
    build_data_source,
    build_offline_pipeline,
    build_online_pipeline,
)
from shaft.model import build_model_tokenizer_processor
from shaft.plugins import (
    ExecutionProxy,
    TrainerHookCallback,
    build_hook_manager,
    build_interceptor_manager,
)
from shaft.training import ShaftProgressCallback
from shaft.training.checkpointing import (
    ensure_hf_export_layout,
    resolve_resume_checkpoint,
    validate_resume_checkpoint,
    validate_training_state_policy,
)
from shaft.training.distributed import barrier_if_distributed

from .registry import PIPELINE_REGISTRY, register_pipeline
from .training_args import build_hf_training_args


@register_pipeline("shaft_rlhf")
class ShaftRLHFPipeline:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.interceptor_manager = build_interceptor_manager(config.plugins.interceptors)

    def build_training_args(self) -> TrainingArguments:
        return build_hf_training_args(self.config)

    def build_datasets(self, algorithm_name: str):
        records_by_dataset_train: dict[str, list[Any]] = {}
        records_by_dataset_val: dict[str, list[Any]] = {}
        weights: dict[str, float] = {}
        dataset_online_pipelines: dict[str, Any] = {}

        for source in self.config.data.datasets:
            if not source.enabled:
                continue
            weights[source.name] = float(source.weight)
            source_impl = build_data_source(source)
            offline_pipeline = build_offline_pipeline(source.offline_transforms)
            train_records = source_impl.load_split("train")
            val_records = source_impl.load_split("val")
            records_by_dataset_train[source.name] = offline_pipeline(train_records)
            records_by_dataset_val[source.name] = offline_pipeline(val_records)
            dataset_online_pipelines[source.name] = build_online_pipeline(source.online_transforms)

        mixer = MixedDatasetBuilder(seed=self.config.experiment.seed)
        mixed_indices = mixer.build_indices(
            records_by_dataset_train,
            weights,
            strategy=self.config.data.mix_strategy,
            shuffle=self.config.data.shuffle,
        )
        mixed_train_records = [
            records_by_dataset_train[dataset_id][row_index]
            for dataset_id, row_index in mixed_indices
        ]
        val_records = []
        for dataset_id in sorted(records_by_dataset_val):
            val_records.extend(records_by_dataset_val[dataset_id])

        def _dataset_aware_online_transform(sample: dict[str, Any]) -> dict[str, Any]:
            dataset_id = str(sample.get("dataset_id", "default"))
            pipeline = dataset_online_pipelines.get(dataset_id)
            if pipeline is None:
                return sample
            return pipeline(sample)

        online_transforms = [_dataset_aware_online_transform]
        if algorithm_name == "dpo":
            return (
                DPODataset(mixed_train_records, online_transforms=online_transforms),
                DPODataset(val_records, online_transforms=online_transforms),
            )
        if algorithm_name == "ppo":
            return (
                PPODataset(mixed_train_records, online_transforms=online_transforms),
                PPODataset(val_records, online_transforms=online_transforms),
            )
        raise ValueError(f"Unsupported RLHF algorithm: {algorithm_name!r}.")

    def _build_collator(self, algorithm_name: str, *, artifacts):
        common_kwargs = {
            "model_meta": artifacts.model_meta,
            "template": artifacts.template,
            "processor": artifacts.processor,
            "tokenizer": artifacts.tokenizer,
            "min_pixels": self.config.data.min_pixels,
            "max_pixels": self.config.data.max_pixels,
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
        if algorithm_name not in {"dpo", "ppo"}:
            raise ValueError(
                f"ShaftRLHFPipeline only supports dpo/ppo, got algorithm={algorithm_name!r}."
            )

        validate_training_state_policy(config)
        artifacts = build_model_tokenizer_processor(
            config,
            init_from_checkpoint=config.sft.train.init_from_checkpoint,
        )
        train_dataset, eval_dataset = self.build_datasets(algorithm_name)
        hook_manager = build_hook_manager(config.plugins.hooks)
        callbacks = []
        if config.progress.enabled:
            callbacks.append(
                ShaftProgressCallback(
                    leave=config.progress.leave,
                    mininterval=config.progress.mininterval,
                )
            )
        if hook_manager.hooks:
            callbacks.append(TrainerHookCallback(hook_manager))
        callbacks_or_none = callbacks or None

        algorithm_cls = ALGORITHM_REGISTRY.get(algorithm_name)
        algorithm = algorithm_cls()
        processing_class = artifacts.processor
        algorithm_extra_kwargs: dict[str, Any] = {}
        if algorithm_name == "ppo":
            algorithm_extra_kwargs["model_meta"] = artifacts.model_meta
        trainer = algorithm.build_trainer(
            context=AlgorithmContext(params=dict(config.algorithm.params)),
            train_config=config.sft.train,
            rlhf_config=getattr(config.rlhf, algorithm_name),
            finetune_mode=config.model.finetune.mode,
            model=artifacts.model,
            args=self.build_training_args(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if config.sft.eval.enabled else None,
            processing_class=processing_class,
            data_collator=self._build_collator(algorithm_name, artifacts=artifacts),
            callbacks=callbacks_or_none,
            **algorithm_extra_kwargs,
        )

        resume_checkpoint = resolve_resume_checkpoint(config.sft.train.resume_from_checkpoint)
        if resume_checkpoint is not None:
            validate_resume_checkpoint(resume_checkpoint, finetune_mode=config.model.finetune.mode)
        if algorithm_name == "ppo":
            if resume_checkpoint is not None:
                raise ValueError("TRL PPOTrainer does not support resume_from_checkpoint in current Shaft pipeline.")
            train_result = trainer.train()
        else:
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        barrier_if_distributed()
        if config.sft.train.save_final_model:
            trainer.save_model()
            ensure_hf_export_layout(
                config.experiment.output_dir,
                finetune_mode=config.model.finetune.mode,
                model_meta=artifacts.model_meta,
            )
        if config.sft.train.save_final_state:
            trainer.save_state()
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
    return runner()
