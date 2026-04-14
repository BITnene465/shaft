from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import TrainingArguments

from shaft.algorithms import ALGORITHM_REGISTRY, AlgorithmContext
from shaft.algorithms import dpo as _dpo  # noqa: F401
from shaft.algorithms import ppo as _ppo  # noqa: F401
from shaft.algorithms import sft as _sft  # noqa: F401
from shaft.config import RuntimeConfig
from shaft.data import (
    build_data_source,
    MixedDatasetBuilder,
    SFTCollator,
    SFTDataset,
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


@register_pipeline("shaft_train")
class ShaftTrainPipeline:
    """HF-first training pipeline.

    The pipeline only coordinates modules; business semantics stay in data/model/algorithms.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.interceptor_manager = build_interceptor_manager(config.plugins.interceptors)

    def build_training_args(self) -> TrainingArguments:
        config = self.config
        sft_train = config.sft.train
        sft_eval = config.sft.eval
        eval_strategy = "no" if not sft_eval.enabled else sft_eval.eval_strategy
        use_bf16 = bool(sft_train.bf16) and torch.cuda.is_available()
        dataloader_num_workers = int(config.data.num_workers)
        return TrainingArguments(
            output_dir=str(Path(config.experiment.output_dir)),
            run_name=config.experiment.run_id or config.experiment.name,
            num_train_epochs=float(sft_train.epochs),
            max_steps=int(sft_train.max_steps),
            per_device_train_batch_size=int(sft_train.per_device_train_batch_size),
            per_device_eval_batch_size=int(sft_eval.per_device_eval_batch_size),
            gradient_accumulation_steps=int(sft_train.gradient_accumulation_steps),
            learning_rate=float(sft_train.learning_rate),
            weight_decay=float(sft_train.weight_decay),
            warmup_ratio=float(sft_train.warmup_ratio),
            lr_scheduler_type=str(sft_train.lr_scheduler_type),
            max_grad_norm=float(sft_train.max_grad_norm),
            bf16=use_bf16,
            use_cpu=bool(sft_train.use_cpu),
            logging_steps=int(sft_train.logging_steps),
            save_strategy=str(sft_train.save_strategy),
            save_steps=int(sft_train.save_steps),
            save_total_limit=int(sft_train.save_total_limit),
            load_best_model_at_end=bool(sft_train.load_best_model_at_end),
            eval_strategy=eval_strategy,
            eval_steps=int(sft_eval.eval_steps),
            metric_for_best_model=str(sft_eval.metric_for_best_model),
            greater_is_better=bool(sft_eval.greater_is_better),
            ddp_find_unused_parameters=bool(sft_train.ddp_find_unused_parameters),
            save_on_each_node=False,
            log_on_each_node=False,
            report_to=list(sft_train.report_to),
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=bool(config.data.pin_memory),
            dataloader_persistent_workers=bool(config.data.persistent_workers and dataloader_num_workers > 0),
            disable_tqdm=True,
            remove_unused_columns=False,
        )

    def build_datasets(self) -> tuple[SFTDataset, SFTDataset]:
        config = self.config
        records_by_dataset_train: dict[str, list[Any]] = {}
        records_by_dataset_val: dict[str, list[Any]] = {}
        weights: dict[str, float] = {}
        dataset_online_pipelines: dict[str, Any] = {}

        for source in config.data.datasets:
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

        mixer = MixedDatasetBuilder(seed=config.experiment.seed)
        mixed_indices = mixer.build_indices(
            records_by_dataset_train,
            weights,
            strategy=config.data.mix_strategy,
            shuffle=config.data.shuffle,
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
        return SFTDataset(mixed_train_records, online_transforms=online_transforms), SFTDataset(
            val_records,
            online_transforms=online_transforms,
        )

    def run(self) -> dict[str, Any]:
        config = self.config
        validate_training_state_policy(config)
        artifacts = build_model_tokenizer_processor(
            config,
            init_from_checkpoint=config.sft.train.init_from_checkpoint,
        )
        train_dataset, eval_dataset = self.build_datasets()
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
        collator = SFTCollator(
            model_meta=artifacts.model_meta,
            template=artifacts.template,
            processor=artifacts.processor,
            tokenizer=artifacts.tokenizer,
            min_pixels=config.data.min_pixels,
            max_pixels=config.data.max_pixels,
            add_eos_token=config.data.add_eos_token,
            include_targets_in_inputs=True,
        )
        algorithm_cls = ALGORITHM_REGISTRY.get(config.algorithm.name)
        algorithm = algorithm_cls()
        trainer = algorithm.build_trainer(
            context=AlgorithmContext(params=dict(config.algorithm.params)),
            train_config=config.sft.train,
            model=artifacts.model,
            args=self.build_training_args(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if config.sft.eval.enabled else None,
            processing_class=artifacts.processor,
            data_collator=collator,
            callbacks=callbacks_or_none,
        )

        resume_checkpoint = resolve_resume_checkpoint(config.sft.train.resume_from_checkpoint)
        if resume_checkpoint is not None:
            validate_resume_checkpoint(resume_checkpoint, finetune_mode=config.model.finetune.mode)
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
        return dict(train_result.metrics or {})


def run_train(config: RuntimeConfig) -> dict[str, Any]:
    pipeline_cls = PIPELINE_REGISTRY.get("shaft_train")
    pipeline = pipeline_cls(config)
    runner = ExecutionProxy(
        point="pipeline.train.run",
        target=pipeline.run,
        interceptor_manager=pipeline.interceptor_manager,
    )
    return runner()
