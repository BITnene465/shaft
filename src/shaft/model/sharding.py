from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShardingPolicy:
    fsdp_transformer_layer_cls_to_wrap: tuple[str, ...] = ()

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

