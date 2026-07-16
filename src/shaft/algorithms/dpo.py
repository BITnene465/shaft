from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.config import DPOConfig, TrainConfig
from shaft.training.trl_trainers import ShaftDPOTrainer

from .base import AlgorithmContext, ShaftTrainerSpec, trainer_spec_contract
from .rlhf_utils import build_reference_model, build_trl_dpo_config
from .registry import register_algorithm

_DPO_ARG_FIELDS = (
    "activation_offloading",
    "beta",
    "disable_dropout",
    "discopop_tau",
    "f_alpha_divergence_coef",
    "f_divergence_type",
    "label_smoothing",
    "ld_alpha",
    "loss_type",
    "loss_weights",
    "max_length",
    "pad_to_multiple_of",
    "pad_token",
    "padding_free",
    "precompute_ref_batch_size",
    "precompute_ref_log_probs",
    "ref_model_mixup_alpha",
    "ref_model_sync_steps",
    "sync_ref_model",
    "truncation_mode",
    "use_weighting",
)


@dataclass
@register_algorithm("dpo")
class DPOAlgorithm:
    name: str = "dpo"

    def prepare_trainer(
        self,
        *,
        context: AlgorithmContext,
        **kwargs: Any,
    ) -> ShaftTrainerSpec[ShaftDPOTrainer]:
        _ = context
        train_config: TrainConfig = kwargs.pop("train_config")
        rlhf_config: DPOConfig = kwargs.pop("rlhf_config")
        training_args = kwargs.pop("args")
        finetune_mode: str = kwargs.pop("finetune_mode")
        model = kwargs.pop("model")
        ref_model = build_reference_model(
            model=model,
            finetune_mode=finetune_mode,
        )
        dpo_args = kwargs.pop("resolved_dpo_args", None)
        if dpo_args is None:
            dpo_args = build_trl_dpo_config(
                train_args=training_args,
                rlhf_config=rlhf_config,
            )
        trainer_kwargs = {
            "model": model,
            "ref_model": ref_model,
            "args": dpo_args,
            "optimizer_name": train_config.optimizer_name,
            "scheduler_name": train_config.scheduler_name,
            "scheduler_num_cycles": train_config.scheduler_num_cycles,
            "scheduler_power": train_config.scheduler_power,
            "adam_beta1": train_config.adam_beta1,
            "adam_beta2": train_config.adam_beta2,
            "adam_epsilon": train_config.adam_epsilon,
            "model_adapter": kwargs.pop("model_adapter"),
            "finetune_plan": kwargs.pop("finetune_plan"),
            "resolved_optimizer_plan": kwargs.pop("resolved_optimizer_plan"),
            "param_group_lrs": dict(train_config.param_group_lrs),
            "no_decay_name_patterns": list(train_config.no_decay_name_patterns),
            **kwargs,
        }
        return ShaftTrainerSpec(
            trainer_cls=ShaftDPOTrainer,
            kwargs=trainer_kwargs,
            contract=trainer_spec_contract(
                algorithm=self.name,
                args=dpo_args,
                train_config=train_config,
                arg_fields=_DPO_ARG_FIELDS,
                extra={
                    "finetune_mode": str(finetune_mode).strip().lower(),
                    "reference_model": "adapter_disabled" if ref_model is None else "frozen_copy",
                },
            ),
        )

    def build_trainer(self, *, context: AlgorithmContext, **kwargs: Any) -> ShaftDPOTrainer:
        return self.prepare_trainer(context=context, **kwargs).build()
