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
    **_: Any,
) -> torch.Tensor:
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
    **_: Any,
) -> torch.Tensor:
    logits = _extract_logits(outputs)
    if logits is None or labels is None:
        raise ValueError("causal_lm loss requires outputs.logits and labels.")
    return causal_lm_cross_entropy(logits=logits, labels=labels, ignore_index=ignore_index)


def causal_lm_cross_entropy(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    vocab_size = int(logits.shape[-1])
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=int(ignore_index),
    )
