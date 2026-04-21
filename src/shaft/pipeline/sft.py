from __future__ import annotations

import logging
from typing import Any

from transformers import TrainingArguments

from shaft.algorithms import ALGORITHM_REGISTRY, AlgorithmContext
from shaft.algorithms import sft as _sft  # noqa: F401
from shaft.config import RuntimeConfig
from shaft.data import (
    SFTCollator,
    ShaftDataCenter,
    SFTDataset,
)
from shaft.model import build_model_tokenizer_processor
from shaft.model import summarize_resolved_finetune_plan, write_resolved_finetune_summary
from shaft.plugins import (
    ExecutionProxy,
    TrainerHookCallback,
    build_hook_manager,
    build_interceptor_manager,
)
from shaft.training import ShaftProgressCallback
from shaft.training.online_eval import ShaftOnlineEvalRunner
from shaft.training.checkpointing import (
    ensure_hf_export_layout,
    resolve_best_export_dir,
    resolve_resume_checkpoint,
    validate_resume_checkpoint,
    validate_training_state_policy,
)
from shaft.training.distributed import barrier_if_distributed

from .registry import PIPELINE_REGISTRY, register_pipeline
from .training_args import build_hf_training_args

logger = logging.getLogger(__name__)


@register_pipeline("shaft_sft")
class ShaftSFTPipeline:
    """HF-first SFT pipeline.

    The pipeline only coordinates modules; business semantics stay in data/model/algorithms.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.interceptor_manager = build_interceptor_manager(config.plugins.interceptors)

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
        artifacts = build_model_tokenizer_processor(
            config,
            init_from_checkpoint=config.train.init_from_checkpoint,
        )
        finetune_plan = getattr(artifacts, "finetune_plan", None)
        if finetune_plan is not None:
            freeze_summary = summarize_resolved_finetune_plan(
                artifacts.model,
                finetune=config.model.finetune,
                plan=finetune_plan,
                model_adapter=artifacts.model_adapter,
            )
            write_resolved_finetune_summary(config.experiment.output_dir, freeze_summary)
            logger.info("[startup] resolved freeze summary: %s", freeze_summary.to_log_dict())
        data_center = ShaftDataCenter(config.data, seed=config.experiment.seed)
        dataset_bundle = data_center.build_dataset_bundle(SFTDataset)
        train_dataset = dataset_bundle.train_dataset
        eval_dataset = dataset_bundle.eval_dataset
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
            model_adapter=artifacts.model_adapter,
            template=artifacts.template,
            processor=artifacts.processor,
            tokenizer=artifacts.tokenizer,
            min_pixels=config.data.min_pixels,
            max_pixels=config.data.max_pixels,
            add_eos_token=config.data.add_eos_token,
            include_targets_in_inputs=True,
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
                    add_eos_token=config.data.add_eos_token,
                    include_targets_in_inputs=False,
                ),
            )
        algorithm_cls = ALGORITHM_REGISTRY.get(algorithm_name)
        algorithm = algorithm_cls()
        trainer = algorithm.build_trainer(
            context=AlgorithmContext(params=dict(config.algorithm.params)),
            train_config=config.train,
            model=artifacts.model,
            args=self.build_training_args(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if config.eval.enabled else None,
            train_sampler=dataset_bundle.train_sampler,
            processing_class=artifacts.processor,
            data_collator=collator,
            callbacks=callbacks_or_none,
            online_eval_runner=online_eval_runner,
            model_adapter=artifacts.model_adapter,
            finetune_plan=finetune_plan,
        )

        resume_checkpoint = resolve_resume_checkpoint(config.train.resume_from_checkpoint)
        if resume_checkpoint is not None:
            validate_resume_checkpoint(resume_checkpoint, finetune_mode=config.model.finetune.mode)
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        barrier_if_distributed()
        if config.train.save_final_model:
            best_export_dir = resolve_best_export_dir(config.experiment.output_dir)
            trainer.save_model(output_dir=str(best_export_dir))
            ensure_hf_export_layout(
                best_export_dir,
                finetune_mode=config.model.finetune.mode,
                model_meta=artifacts.model_adapter,
            )
        if config.train.save_final_state:
            trainer.save_state()
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
    return runner()
