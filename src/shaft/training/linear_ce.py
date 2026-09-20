"""Loss-only LM-head execution. The causal objective remains owned by Shaft."""

from importlib.metadata import version
from functools import cache
from collections.abc import Callable

import torch

from .loss import _causal_lm_weights_and_denominator


@cache
def load_fused_linear_ce() -> Callable[..., torch.Tensor]:
    try:
        # Older versions have different per-token backward contracts. Do not silently
        # accept them for weighted objectives.
        if version("liger-kernel") != "0.8.3":
            raise ImportError("Shaft fused linear CE requires liger-kernel==0.8.3.")
        from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
    except ImportError as exc:
        raise ImportError("Install Shaft's [fused-ce] extra to enable fused_linear_ce.") from exc
    return liger_fused_linear_cross_entropy


def linear_causal_cross_entropy(
    *,
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Linear,
    labels: torch.Tensor,
    ignore_index: int = -100,
    loss_scale: torch.Tensor | None = None,
    normalization_denominator: torch.Tensor | int | float | None = None,
) -> torch.Tensor:
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError("Fused linear CE requires [B,L,H] hidden states and [B,L] labels.")
    if hidden_states.shape[:2] != labels.shape or labels.shape[1] < 2:
        raise ValueError("Fused linear CE requires aligned sequences of length >= 2.")
    if torch.is_grad_enabled() and not hidden_states.requires_grad:
        # Liger 0.8.3's sum path computes head gradients together with input
        # gradients. Do not silently accept head-only training on a frozen backbone.
        raise ValueError(
            "Fused linear CE requires backbone gradients; head-only training is unsupported."
        )
    shift_labels, weights, denominator = _causal_lm_weights_and_denominator(
        labels=labels,
        loss_scale=loss_scale,
        ignore_index=ignore_index,
        normalization_denominator=normalization_denominator,
        device=hidden_states.device,
        compute_dtype=torch.float64 if hidden_states.dtype == torch.float64 else torch.float32,
    )
    valid = shift_labels.ne(ignore_index)
    if not bool(valid.any()) or float(denominator.detach()) <= 0:
        # Keep both the backbone and head (including tied weights) in the DDP graph.
        zero = hidden_states.sum() * 0 + lm_head.weight.sum() * 0
        return zero if lm_head.bias is None else zero + lm_head.bias.sum() * 0
    hidden = hidden_states[:, :-1][valid].contiguous()
    targets = shift_labels[valid].contiguous()
    kernel = load_fused_linear_ce()
    losses = kernel(
        hidden,
        lm_head.weight,
        targets,
        bias=lm_head.bias,
        ignore_index=ignore_index,
        reduction="sum" if loss_scale is None else "none",
        accum_dtype=torch.float32,
    )
    numerator = losses if loss_scale is None else (losses * weights[valid]).sum()
    return numerator / denominator
