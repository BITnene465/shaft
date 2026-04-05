from __future__ import annotations

import torch


def compute_weighted_token_ce_loss(
    model_outputs,
    batch: dict,
    *,
    ignore_index: int = -100,
) -> object:
    loss_weights = batch.get("loss_weights")
    if loss_weights is None:
        raise ValueError("Weighted token loss requires batch['loss_weights'].")

    logits = model_outputs.logits
    labels = batch["labels"].to(logits.device)
    loss_weights = loss_weights.to(logits.device)
    if loss_weights.shape != labels.shape:
        raise ValueError(
            "Weighted token loss requires batch loss_weights to align with labels. "
            f"loss_weights={tuple(loss_weights.shape)}, labels={tuple(labels.shape)}."
        )

    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = loss_weights[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
    token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    ).view_as(shift_labels)
    valid_mask = (shift_labels != ignore_index).to(token_loss.dtype)
    weighted_loss = token_loss * shift_weights * valid_mask
    denom = (shift_weights * valid_mask).sum().clamp_min(1.0)
    return weighted_loss.sum() / denom
