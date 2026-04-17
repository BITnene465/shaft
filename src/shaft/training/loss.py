from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from shaft.plugins import Registry

LossFn = Callable[..., torch.Tensor]
LOSS_REGISTRY: Registry[LossFn] = Registry("loss")


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
    **_: Any,
) -> torch.Tensor:
    if loss_scale is not None:
        logits = _extract_logits(outputs)
        if logits is None or labels is None:
            raise ValueError("auto loss with loss_scale requires outputs.logits and labels.")
        return causal_lm_cross_entropy(
            logits=logits,
            labels=labels,
            ignore_index=ignore_index,
            loss_scale=loss_scale,
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
    )


def causal_lm_cross_entropy(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    loss_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = int(shift_logits.shape[-1])
    token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=int(ignore_index),
        reduction="none",
    ).view_as(shift_labels)
    valid_mask = shift_labels.ne(int(ignore_index))
    if loss_scale is None:
        weights = valid_mask.to(dtype=token_loss.dtype)
    else:
        shift_loss_scale = loss_scale[..., 1:].contiguous().to(device=token_loss.device, dtype=token_loss.dtype)
        weights = shift_loss_scale * valid_mask.to(dtype=token_loss.dtype)
    denom = weights.sum()
    if float(denom.detach().item()) <= 0:
        return token_loss.sum() * 0.0
    return (token_loss * weights).sum() / denom
