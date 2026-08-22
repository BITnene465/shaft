from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from shaft.plugins import Registry

LossFn = Callable[..., torch.Tensor]
LOSS_REGISTRY: Registry[LossFn] = Registry("loss")

_DEFAULT_MAX_CROSS_ENTROPY_TOKENS_PER_CHUNK = 512


def register_loss(name: str):
    return LOSS_REGISTRY.register(name)


def build_loss(loss_name: str) -> LossFn:
    normalized = str(loss_name).strip().lower()
    return LOSS_REGISTRY.get(normalized)


def _extract_loss(outputs: Any) -> torch.Tensor | None:
    maybe_loss = getattr(outputs, "loss", None)
    if isinstance(maybe_loss, torch.Tensor):
        return maybe_loss
    if isinstance(outputs, dict):
        loss = outputs.get("loss")
        if isinstance(loss, torch.Tensor):
            return loss
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor) and first.ndim == 0:
            return first
    return None


def _extract_logits(outputs: Any) -> torch.Tensor | None:
    logits = getattr(outputs, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(outputs, dict):
        maybe = outputs.get("logits")
        if isinstance(maybe, torch.Tensor):
            return maybe
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if isinstance(item, torch.Tensor) and item.ndim >= 2:
                return item
    return None


@register_loss("auto")
def auto_loss(
    *,
    outputs: Any,
    labels: torch.Tensor | None,
    ignore_index: int = -100,
    loss_scale: torch.Tensor | None = None,
    normalization_denominator: torch.Tensor | int | float | None = None,
    component_output: dict[str, torch.Tensor] | None = None,
    **_: Any,
) -> torch.Tensor:
    if loss_scale is not None or normalization_denominator is not None:
        logits = _extract_logits(outputs)
        if logits is None or labels is None:
            raise ValueError(
                "auto loss with explicit normalization requires outputs.logits and labels."
            )
        return causal_lm_cross_entropy(
            logits=logits,
            labels=labels,
            ignore_index=ignore_index,
            loss_scale=loss_scale,
            normalization_denominator=normalization_denominator,
            component_output=component_output,
        )
    maybe_loss = _extract_loss(outputs)
    if isinstance(maybe_loss, torch.Tensor):
        return maybe_loss
    logits = _extract_logits(outputs)
    if logits is None or labels is None:
        raise ValueError("auto loss requires model outputs.loss or (outputs.logits and labels).")
    return causal_lm_cross_entropy(logits=logits, labels=labels, ignore_index=ignore_index)


@register_loss("causal_lm")
def causal_lm_loss(
    *,
    outputs: Any,
    labels: torch.Tensor | None,
    ignore_index: int = -100,
    loss_scale: torch.Tensor | None = None,
    normalization_denominator: torch.Tensor | int | float | None = None,
    component_output: dict[str, torch.Tensor] | None = None,
    **_: Any,
) -> torch.Tensor:
    logits = _extract_logits(outputs)
    if logits is None or labels is None:
        raise ValueError("causal_lm loss requires outputs.logits and labels.")
    return causal_lm_cross_entropy(
        logits=logits,
        labels=labels,
        ignore_index=ignore_index,
        loss_scale=loss_scale,
        normalization_denominator=normalization_denominator,
        component_output=component_output,
    )


def _cross_entropy_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _iter_cross_entropy_chunks(
    *,
    batch_size: int,
    sequence_length: int,
    max_tokens_per_chunk: int,
):
    """Yield 2D slices whose flattened token count is bounded by the chunk cap."""

    batch_chunk_size = min(batch_size, max_tokens_per_chunk)
    for batch_start in range(0, batch_size, batch_chunk_size):
        batch_end = min(batch_size, batch_start + batch_chunk_size)
        current_batch_size = batch_end - batch_start
        sequence_chunk_size = max(1, max_tokens_per_chunk // current_batch_size)
        for sequence_start in range(0, sequence_length, sequence_chunk_size):
            sequence_end = min(sequence_length, sequence_start + sequence_chunk_size)
            yield batch_start, batch_end, sequence_start, sequence_end


def _cross_entropy_token_chunk(
    *,
    chunk_logits: torch.Tensor,
    chunk_labels: torch.Tensor,
    ignore_index: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    vocab_size = int(chunk_logits.shape[-1])
    return F.cross_entropy(
        chunk_logits.reshape(-1, vocab_size).to(dtype=compute_dtype),
        chunk_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(chunk_labels)


class _MemoryEfficientCausalLMCrossEntropy(torch.autograd.Function):
    """Bound CE workspace without retaining one vocabulary-sized buffer per chunk.

    The forward computes token chunks without an autograd graph. The backward recomputes
    one chunk at a time and writes its gradient into the full logits gradient. Saving only
    the original logits plus token-sized metadata avoids the ``tokens x vocabulary``
    log-softmax activation that caused long-sequence training OOMs.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        shift_labels: torch.Tensor,
        weights: torch.Tensor,
        denominator: torch.Tensor,
        ignore_index: int,
        max_tokens_per_chunk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = shift_labels.shape
        compute_dtype = _cross_entropy_compute_dtype(logits.dtype)
        per_example_numerator = torch.zeros(
            batch_size,
            device=logits.device,
            dtype=compute_dtype,
        )

        for batch_start, batch_end, sequence_start, sequence_end in _iter_cross_entropy_chunks(
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_tokens_per_chunk=max_tokens_per_chunk,
        ):
            chunk_logits = logits[
                batch_start:batch_end,
                sequence_start:sequence_end,
                :,
            ]
            chunk_labels = shift_labels[
                batch_start:batch_end,
                sequence_start:sequence_end,
            ]
            chunk_weights = weights[
                batch_start:batch_end,
                sequence_start:sequence_end,
            ]
            token_loss = _cross_entropy_token_chunk(
                chunk_logits=chunk_logits,
                chunk_labels=chunk_labels,
                ignore_index=ignore_index,
                compute_dtype=compute_dtype,
            )
            weighted_token_loss = token_loss * chunk_weights
            per_example_numerator[batch_start:batch_end] += weighted_token_loss.sum(dim=-1)

        ctx.save_for_backward(logits, shift_labels, weights, denominator)
        ctx.ignore_index = int(ignore_index)
        ctx.max_tokens_per_chunk = int(max_tokens_per_chunk)
        ctx.mark_non_differentiable(per_example_numerator)
        return per_example_numerator.sum() / denominator, per_example_numerator

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        grad_loss: torch.Tensor | None,
        _grad_per_example_numerator: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, None, None, None, None, None]:
        logits, shift_labels, weights, denominator = ctx.saved_tensors
        if grad_loss is None:
            return None, None, None, None, None, None

        batch_size, sequence_length = shift_labels.shape
        compute_dtype = _cross_entropy_compute_dtype(logits.dtype)
        grad_logits = torch.zeros_like(logits)

        with torch.enable_grad():
            for batch_start, batch_end, sequence_start, sequence_end in _iter_cross_entropy_chunks(
                batch_size=batch_size,
                sequence_length=sequence_length,
                max_tokens_per_chunk=ctx.max_tokens_per_chunk,
            ):
                chunk_logits = (
                    logits[
                        batch_start:batch_end,
                        sequence_start:sequence_end,
                        :,
                    ]
                    .detach()
                    .to(dtype=compute_dtype)
                    .requires_grad_(True)
                )
                chunk_labels = shift_labels[
                    batch_start:batch_end,
                    sequence_start:sequence_end,
                ]
                chunk_weights = weights[
                    batch_start:batch_end,
                    sequence_start:sequence_end,
                ]
                token_loss = _cross_entropy_token_chunk(
                    chunk_logits=chunk_logits,
                    chunk_labels=chunk_labels,
                    ignore_index=ctx.ignore_index,
                    compute_dtype=compute_dtype,
                )
                chunk_objective = (token_loss * chunk_weights).sum() / denominator
                (chunk_gradient,) = torch.autograd.grad(
                    chunk_objective,
                    chunk_logits,
                    grad_outputs=grad_loss.to(dtype=chunk_objective.dtype),
                )
                grad_logits[
                    batch_start:batch_end,
                    sequence_start:sequence_end,
                    :,
                ].copy_(chunk_gradient.to(dtype=logits.dtype))

        return grad_logits, None, None, None, None, None


def causal_lm_cross_entropy(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    loss_scale: torch.Tensor | None = None,
    normalization_denominator: torch.Tensor | int | float | None = None,
    component_output: dict[str, torch.Tensor] | None = None,
    max_tokens_per_chunk: int = _DEFAULT_MAX_CROSS_ENTROPY_TOKENS_PER_CHUNK,
) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "causal LM cross entropy requires [batch, sequence, vocabulary] logits and [batch, sequence] labels."
        )
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("causal LM logits and labels must align on batch and sequence dimensions.")
    if int(labels.shape[-1]) < 2:
        raise ValueError("causal LM cross entropy requires sequence length >= 2.")
    max_tokens_per_chunk = int(max_tokens_per_chunk)
    if max_tokens_per_chunk <= 0:
        raise ValueError("max_tokens_per_chunk must be > 0.")

    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels.ne(int(ignore_index))
    compute_dtype = _cross_entropy_compute_dtype(logits.dtype)
    if loss_scale is None:
        weights = valid_mask.to(device=logits.device, dtype=compute_dtype)
    else:
        if tuple(loss_scale.shape) != tuple(labels.shape):
            raise ValueError("loss_scale must align with labels.")
        shift_loss_scale = (
            loss_scale[:, 1:]
            .contiguous()
            .to(
                device=logits.device,
                dtype=compute_dtype,
            )
        )
        weights = shift_loss_scale * valid_mask.to(device=logits.device, dtype=compute_dtype)
    local_denominator = weights.sum()
    denom = (
        local_denominator
        if normalization_denominator is None
        else torch.as_tensor(
            normalization_denominator,
            device=logits.device,
            dtype=compute_dtype,
        )
    )
    if float(denom.detach().item()) <= 0:
        if component_output is not None:
            component_output["numerator"] = torch.zeros(
                int(labels.shape[0]),
                device=logits.device,
                dtype=compute_dtype,
            )
            component_output["denominator"] = weights.sum(dim=-1)
        return logits[:, :-1, :].sum() * 0.0

    loss, per_example_numerator = _MemoryEfficientCausalLMCrossEntropy.apply(
        logits,
        shift_labels,
        weights,
        denom,
        int(ignore_index),
        max_tokens_per_chunk,
    )
    if component_output is not None:
        component_output["numerator"] = per_example_numerator
        component_output["denominator"] = weights.sum(dim=-1)
    return loss
