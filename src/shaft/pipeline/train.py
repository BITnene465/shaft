from __future__ import annotations

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


@register_pipeline("shaft_train")
class ShaftTrainPipeline:
    """HF-first training pipeline.

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
                f"ShaftTrainPipeline only supports sft, got algorithm={algorithm_name!r}. "
                "Use ShaftRLHFPipeline for dpo/ppo."
            )
        validate_training_state_policy(config)
        artifacts = build_model_tokenizer_processor(
            config,
            init_from_checkpoint=config.sft.train.init_from_checkpoint,
        )
        data_center = ShaftDataCenter(config.data, seed=config.experiment.seed)
        train_dataset, eval_dataset = data_center.build_dataset_pair(SFTDataset)
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
        )
        algorithm_cls = ALGORITHM_REGISTRY.get(algorithm_name)
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
                model_meta=artifacts.model_adapter,
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
