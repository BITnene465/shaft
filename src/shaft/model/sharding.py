from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.config.training import resolve_deepspeed_zero_stage


@dataclass(frozen=True)
class ModelShardingPolicy:
    fsdp_transformer_layer_cls_to_wrap: tuple[str, ...] = ()
    supports_fsdp_activation_checkpointing: bool = True
    supports_deepspeed_zero3: bool = True

    def resolve_fsdp_transformer_layer_cls_to_wrap(self, values: list[str]) -> list[str]:
        normalized = [str(item).strip() for item in values if str(item).strip()]
        if normalized != ["auto"]:
            return normalized
        if not self.fsdp_transformer_layer_cls_to_wrap:
            raise ValueError(
                "FSDP transformer_layer_cls_to_wrap=['auto'] requires a model sharding policy "
                "with fsdp_transformer_layer_cls_to_wrap defaults."
            )
        return list(self.fsdp_transformer_layer_cls_to_wrap)

    def validate_distributed_config(
        self,
        train: Any,
        *,
        finetune: Any | None = None,
    ) -> None:
        distributed = train.distributed
        strategy = str(distributed.strategy).strip().lower()
        zero_stage = (
            resolve_deepspeed_zero_stage(train.distributed.deepspeed)
            if strategy == "deepspeed"
            else 0
        )
        if strategy == "deepspeed" and zero_stage == 3 and not self.supports_deepspeed_zero3:
            raise ValueError(
                "This model execution policy does not support DeepSpeed ZeRO-3; "
                "use FSDP until the model's routed-expert leaf-module contract is implemented."
            )
        if (
            strategy == "deepspeed"
            and finetune is not None
            and tuple(getattr(finetune, "target_parameters", ()) or ())
            and zero_stage == 3
        ):
            raise ValueError(
                "DeepSpeed ZeRO-3 cannot inject PEFT target_parameters while "
                "Transformers constructs partitioned parameters. Use FSDP for "
                "fused-parameter LoRA, or use full finetuning with ZeRO-3."
            )
        if strategy != "fsdp":
            return
        finetune_mode = (
            None
            if finetune is None
            else str(getattr(finetune, "mode", "")).strip().lower()
        )
        if finetune_mode in {"lora", "dora", "qlora"}:
            if str(distributed.fsdp.state_dict_type).strip().lower() != "full_state_dict":
                raise ValueError(
                    "FSDP PEFT currently requires state_dict_type='full_state_dict' "
                    "so checkpoints contain one complete standard adapter for exact resume."
                )
            if bool(train.load_best_model_at_end):
                raise ValueError(
                    "FSDP PEFT currently requires load_best_model_at_end=false because "
                    "the upstream adapter-only best-model loader restores an incomplete "
                    "local DTensor shard."
                )
        if (
            not self.supports_fsdp_activation_checkpointing
            and bool(distributed.fsdp.activation_checkpointing)
        ):
            raise ValueError(
                "This model execution policy requires "
                "train.distributed.fsdp.activation_checkpointing=false; use "
                "train.gradient_checkpointing for model-side recomputation."
            )
