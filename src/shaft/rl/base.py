from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from shaft.algorithms.base import ShaftTrainerSpec
from shaft.config import RuntimeConfig
from shaft.data import ShaftDatasetBundle
from shaft.observability import ShaftProgressManager


@dataclass(frozen=True, slots=True)
class RLResolvedArgs:
    """Algorithm-specific TRL args behind one RL pipeline-facing shape."""

    trainer_args: Any | None = None
    sample_contract: Any | None = None
    resume_context: dict[str, Any] = field(default_factory=dict)

    def resume_contract_kwargs(self) -> dict[str, Any]:
        return {"algorithm_resume_context": dict(self.resume_context)}


@dataclass(slots=True)
class RLTrainerInputs:
    train_dataset: Any
    eval_dataset: Any
    data_collator: Any | None = None
    eval_data_collator: Any | None = None
    online_eval_runner: Any | None = None
    eval_config: Any | None = None


@dataclass(frozen=True, slots=True)
class RLRuntimeContext:
    config: RuntimeConfig
    training_args: Any
    artifacts: Any
    dataset_bundle: ShaftDatasetBundle[Any]
    resolved_args: RLResolvedArgs
    progress_manager: ShaftProgressManager
    checkpoint_protocol: str


class RLRuntime(ABC):
    """Algorithm-owned behavior consumed by the branch-free RL pipeline."""

    name: str
    dataset_type: type
    input_builder: type
    supports_checkpoint: bool = True
    input_mode: str = "training"

    def validate_config(self, config: RuntimeConfig) -> None:
        _ = config

    @abstractmethod
    def resolve_args(self, config: RuntimeConfig, training_args: Any) -> RLResolvedArgs:
        raise NotImplementedError

    def checkpointing_requested(
        self,
        config: RuntimeConfig,
        *,
        resume_checkpoint: str | None,
    ) -> bool:
        if not self.supports_checkpoint:
            return False
        return bool(
            str(config.train.save_strategy).strip().lower() != "no"
            or resume_checkpoint is not None
        )

    def build_train_sample_budget(
        self,
        *,
        batch_contract: Any,
        resolved_args: RLResolvedArgs,
        training_args: Any,
    ) -> int | None:
        _ = resolved_args
        return batch_contract.finite_sample_plan_size(max_steps=training_args.max_steps)

    def validate_dataset_bundle(
        self,
        bundle: ShaftDatasetBundle[Any],
        *,
        resolved_args: RLResolvedArgs,
        training_args: Any,
        resume_checkpoint: str | None,
    ) -> None:
        _ = bundle, resolved_args, training_args, resume_checkpoint

    def bind_execution_fingerprint(
        self,
        fingerprint: str,
        *,
        resolved_args: RLResolvedArgs,
    ) -> str:
        _ = resolved_args
        return fingerprint

    def input_options(
        self,
        context: RLRuntimeContext,
        *,
        sequence_execution_contract: Any,
    ) -> dict[str, Any]:
        return {
            "min_pixels": context.config.data.min_pixels,
            "max_pixels": context.config.data.max_pixels,
            "max_length": context.config.data.max_length,
            "add_eos_token": context.config.data.add_eos_token,
            "input_mode": self.input_mode,
            "ignore_index": -100,
            "sequence_execution_contract_fingerprint": (
                sequence_execution_contract.fingerprint
            ),
        }

    @abstractmethod
    def build_trainer_inputs(self, context: RLRuntimeContext) -> RLTrainerInputs:
        raise NotImplementedError

    @abstractmethod
    def prepare_trainer(
        self,
        context: RLRuntimeContext,
        *,
        trainer_inputs: RLTrainerInputs,
        callbacks: list[Any] | None,
        finetune_plan: Any,
        resolved_optimizer_plan: Any,
    ) -> ShaftTrainerSpec[Any]:
        raise NotImplementedError

    def train(self, trainer: Any, *, resume_checkpoint: str | None) -> Any:
        if not self.supports_checkpoint:
            return trainer.train()
        return trainer.train(resume_from_checkpoint=resume_checkpoint)
