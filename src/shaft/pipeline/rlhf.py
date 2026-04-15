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
    ShaftDataCenter,
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

    def _build_collator(self, algorithm_name: str, *, artifacts):
        common_kwargs = {
            "model_adapter": artifacts.model_adapter,
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
            init_from_checkpoint=config.train.init_from_checkpoint,
        )
        data_center = ShaftDataCenter(config.data, seed=config.experiment.seed)
        dataset_cls = DPODataset if algorithm_name == "dpo" else PPODataset
        train_dataset, eval_dataset = data_center.build_dataset_pair(dataset_cls)
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
            train_config=config.train,
            rlhf_config=getattr(config.rlhf, algorithm_name),
            finetune_mode=config.model.finetune.mode,
            model=artifacts.model,
            args=self.build_training_args(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if config.eval.enabled else None,
            processing_class=processing_class,
            data_collator=self._build_collator(algorithm_name, artifacts=artifacts),
            callbacks=callbacks_or_none,
            **algorithm_extra_kwargs,
        )

        resume_checkpoint = resolve_resume_checkpoint(config.train.resume_from_checkpoint)
        if resume_checkpoint is not None:
            validate_resume_checkpoint(resume_checkpoint, finetune_mode=config.model.finetune.mode)
        if algorithm_name == "ppo":
            if resume_checkpoint is not None:
                raise ValueError("TRL PPOTrainer does not support resume_from_checkpoint in current Shaft pipeline.")
            train_result = trainer.train()
        else:
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        barrier_if_distributed()
        if config.train.save_final_model:
            trainer.save_model()
            ensure_hf_export_layout(
                config.experiment.output_dir,
                finetune_mode=config.model.finetune.mode,
                model_meta=artifacts.model_adapter,
            )
        if config.train.save_final_state:
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
