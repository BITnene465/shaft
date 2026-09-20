"""SFT reduction contracts, independent of CE kernel and HF GA conventions."""

import torch


LOSS_NORMALIZATIONS = frozenset({"global_token", "rank_token", "microbatch_token"})


def validate_loss_normalization(value: str) -> str:
    if not isinstance(value, str) or value not in LOSS_NORMALIZATIONS:
        raise ValueError(
            f"Unsupported train.loss_normalization={value!r}; "
            f"expected one of {sorted(LOSS_NORMALIZATIONS)}."
        )
    return value


def supervision_mass(labels, loss_scale=None, ignore_index=-100):
    valid = labels[..., 1:].ne(ignore_index).to(dtype=torch.float32)
    if loss_scale is not None:
        if loss_scale.shape != labels.shape:
            raise ValueError("loss_scale must align with labels.")
        valid = valid * loss_scale[..., 1:].to(device=labels.device, dtype=torch.float32)
    return valid.sum()
