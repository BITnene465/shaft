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


@register_loss("auto")
def auto_loss(
    *,
    outputs: Any,
    labels: torch.Tensor | None,
    ignore_index: int = -100,
    **_: Any,
) -> torch.Tensor:
    maybe_loss = getattr(outputs, "loss", None)
    if isinstance(maybe_loss, torch.Tensor):
        return maybe_loss
    logits = getattr(outputs, "logits", None)
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
    logits = getattr(outputs, "logits", None)
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
