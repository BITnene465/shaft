from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch

from .types import (
    ShaftAuxiliaryLossTerm,
    ShaftEvalAuxiliaryMetric,
    ShaftEvalAuxiliaryStatistic,
    TrainingObjectivePolicy,
)


def _iter_wrapped_models(model: Any) -> Iterator[Any]:
    pending = [model]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        for attribute in ("module", "base_model", "model"):
            nested = getattr(candidate, attribute, None)
            if nested is not None and id(nested) not in seen:
                pending.append(nested)


def _resolve_moe_text_config(model: Any) -> Any:
    for candidate in _iter_wrapped_models(model):
        config = getattr(candidate, "config", None)
        if config is None:
            continue
        text_config = getattr(config, "text_config", None)
        for config_candidate in (text_config, config):
            if config_candidate is None:
                continue
            if getattr(config_candidate, "num_experts", None) is not None:
                return config_candidate
    raise ValueError("Qwen VL MoE SFT requires a resolvable MoE text config.")


def _resolve_moe_config_value(model: Any, name: str) -> Any:
    value = getattr(_resolve_moe_text_config(model), name, None)
    if value is None:
        raise ValueError(f"Qwen VL MoE SFT requires model.config.text_config.{name}.")
    return value


def _expected_router_layer_count(model: Any) -> int:
    config = _resolve_moe_text_config(model)
    num_hidden_layers = int(getattr(config, "num_hidden_layers", 0))
    if num_hidden_layers <= 0:
        raise ValueError("Qwen VL MoE SFT requires num_hidden_layers > 0.")
    raw_sparse_step = getattr(config, "decoder_sparse_step", None)
    if raw_sparse_step is None:
        return num_hidden_layers
    sparse_step = int(raw_sparse_step)
    if sparse_step <= 0:
        raise ValueError("Qwen VL MoE SFT requires decoder_sparse_step > 0.")
    mlp_only_layers = {
        int(layer_index)
        for layer_index in (getattr(config, "mlp_only_layers", None) or ())
    }
    if any(layer_index < 0 or layer_index >= num_hidden_layers for layer_index in mlp_only_layers):
        raise ValueError("Qwen VL MoE SFT received an out-of-range mlp_only_layers entry.")
    return sum(
        1
        for layer_index in range(num_hidden_layers)
        if layer_index not in mlp_only_layers
        and (layer_index + 1) % sparse_step == 0
    )


def _validated_router_logits(model: Any, outputs: Any, *, stage: str) -> tuple[torch.Tensor, ...]:
    router_logits = getattr(outputs, "router_logits", None)
    if not isinstance(router_logits, tuple) or not router_logits:
        raise RuntimeError(
            f"Qwen VL MoE {stage} did not return per-layer router_logits after "
            "output_router_logits=True."
        )
    expected_layers = _expected_router_layer_count(model)
    if len(router_logits) != expected_layers:
        raise RuntimeError(
            f"Qwen VL MoE {stage} returned an incomplete router trace: "
            f"expected_layers={expected_layers}, actual_layers={len(router_logits)}."
        )
    num_experts = int(_resolve_moe_config_value(model, "num_experts"))
    for layer_index, layer_logits in enumerate(router_logits):
        if (
            not torch.is_tensor(layer_logits)
            or layer_logits.ndim != 2
            or int(layer_logits.shape[-1]) != num_experts
        ):
            actual_shape = None if not torch.is_tensor(layer_logits) else tuple(layer_logits.shape)
            raise RuntimeError(
                "Qwen VL MoE router trace has an invalid layer tensor: "
                f"layer={layer_index}, expected_num_experts={num_experts}, "
                f"actual_shape={actual_shape}."
            )
    return router_logits


def _resolve_router_aux_loss_coefficient(model: Any) -> float:
    return float(_resolve_moe_config_value(model, "router_aux_loss_coef"))


@dataclass(frozen=True)
class QwenVLMoeTrainingObjectivePolicy(TrainingObjectivePolicy):
    """Enable and combine the upstream Qwen VL MoE router-balancing objective."""

    def auxiliary_loss_names(self) -> tuple[str, ...]:
        return ("router_aux_loss",)

    def prepare_sft_forward_inputs(
        self,
        *,
        model: Any,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        _ = model
        prepared = dict(inputs)
        prepared["output_router_logits"] = True
        return prepared

    def resolve_sft_auxiliary_loss_terms(
        self,
        *,
        model: Any,
        outputs: Any,
        inputs: dict[str, Any],
    ) -> tuple[ShaftAuxiliaryLossTerm, ...]:
        _ = inputs
        _validated_router_logits(model, outputs, stage="training")
        auxiliary_loss = getattr(outputs, "aux_loss", None)
        if not torch.is_tensor(auxiliary_loss):
            raise RuntimeError(
                "Qwen VL MoE forward did not return aux_loss after "
                "output_router_logits=True. Verify the Transformers model variant."
            )
        return (
            ShaftAuxiliaryLossTerm(
                name="router_aux_loss",
                value=auxiliary_loss,
                coefficient=_resolve_router_aux_loss_coefficient(model),
            ),
        )

    def resolve_sft_eval_auxiliary_statistics(
        self,
        *,
        model: Any,
        outputs: Any,
        inputs: dict[str, Any],
    ) -> tuple[ShaftEvalAuxiliaryStatistic, ...]:
        router_logits = _validated_router_logits(model, outputs, stage="eval")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            input_ids = inputs.get("input_ids")
            if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
                raise ValueError(
                    "Qwen VL MoE eval statistics require a 2-D attention_mask "
                    "or input_ids tensor."
                )
            attention_mask = torch.ones_like(input_ids)
        if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
            raise ValueError(
                "Qwen VL MoE eval statistics require a 2-D attention_mask."
            )

        batch_size, sequence_length = map(int, attention_mask.shape)
        num_experts = int(_resolve_moe_config_value(model, "num_experts"))
        top_k = int(_resolve_moe_config_value(model, "num_experts_per_tok"))
        if num_experts <= 0 or top_k <= 0 or top_k > num_experts:
            raise ValueError(
                "Qwen VL MoE eval statistics received an invalid expert topology: "
                f"num_experts={num_experts}, top_k={top_k}."
            )

        device = router_logits[0].device
        valid = attention_mask.to(device=device, dtype=torch.float32)
        expert_counts = torch.zeros(
            (batch_size, top_k, num_experts),
            device=device,
            dtype=torch.float32,
        )
        router_probability_sums = torch.zeros(
            (batch_size, num_experts),
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            for layer_index, layer_logits in enumerate(router_logits):
                if not torch.is_tensor(layer_logits) or layer_logits.shape != (
                    batch_size * sequence_length,
                    num_experts,
                ):
                    actual_shape = (
                        None
                        if not torch.is_tensor(layer_logits)
                        else tuple(layer_logits.shape)
                    )
                    raise ValueError(
                        "Qwen VL MoE router_logits shape does not match the padded "
                        "eval batch: "
                        f"layer={layer_index}, expected="
                        f"{(batch_size * sequence_length, num_experts)}, "
                        f"actual={actual_shape}."
                    )
                routing_weights = torch.softmax(
                    layer_logits.detach().to(
                        device=device,
                        dtype=torch.float32,
                    ),
                    dim=-1,
                ).reshape(batch_size, sequence_length, num_experts)
                selected_experts = torch.topk(
                    routing_weights,
                    top_k,
                    dim=-1,
                ).indices
                router_probability_sums += (
                    routing_weights * valid.unsqueeze(-1)
                ).sum(dim=1)
                for slot in range(top_k):
                    expert_counts[:, slot, :].scatter_add_(
                        1,
                        selected_experts[:, :, slot],
                        valid,
                    )
        valid_layer_tokens = (
            valid.sum(dim=1, keepdim=True) * float(len(router_logits))
        )
        return (
            ShaftEvalAuxiliaryStatistic(
                name="router_global_balance",
                coefficient=_resolve_router_aux_loss_coefficient(model),
                coefficient_key="router_aux_loss",
                components={
                    "expert_counts": expert_counts,
                    "router_probability_sums": router_probability_sums,
                    "valid_layer_tokens": valid_layer_tokens,
                },
            ),
        )

    def finalize_sft_eval_auxiliary_statistics(
        self,
        statistics: tuple[ShaftEvalAuxiliaryStatistic, ...],
    ) -> tuple[ShaftEvalAuxiliaryMetric, ...]:
        if len(statistics) != 1 or statistics[0].name != "router_global_balance":
            raise ValueError(
                "Qwen VL MoE eval requires exactly one router_global_balance "
                "statistic."
            )
        statistic = statistics[0]
        components = statistic.components
        expected_components = {
            "expert_counts",
            "router_probability_sums",
            "valid_layer_tokens",
        }
        if set(components) != expected_components:
            raise ValueError(
                "Qwen VL MoE eval statistic components are incomplete: "
                f"expected={sorted(expected_components)}, actual={sorted(components)}."
            )
        expert_counts = components["expert_counts"].sum(dim=0)
        router_probability_sums = components["router_probability_sums"].sum(dim=0)
        valid_layer_tokens = components["valid_layer_tokens"].sum()
        if float(valid_layer_tokens.item()) <= 0.0:
            value = valid_layer_tokens * 0.0
        else:
            tokens_per_expert = expert_counts / valid_layer_tokens
            probability_per_expert = router_probability_sums / valid_layer_tokens
            value = (
                tokens_per_expert
                * probability_per_expert.unsqueeze(0)
            ).sum() * float(expert_counts.shape[-1])
        return (
            ShaftEvalAuxiliaryMetric(
                name=statistic.name,
                value=value,
                coefficient_key=statistic.coefficient_key,
                coefficient=statistic.coefficient,
            ),
        )
