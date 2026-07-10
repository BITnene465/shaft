from __future__ import annotations

import pytest
import torch

from shaft.training.loss import (
    LOSS_REGISTRY,
    auto_loss,
    build_loss,
    causal_lm_cross_entropy,
    causal_lm_loss,
)
from tests.support.training import DummyOutput as _DummyOutput


pytestmark = pytest.mark.component


def test_loss_functions() -> None:
    assert LOSS_REGISTRY.has("auto")
    assert LOSS_REGISTRY.has("causal_lm")
    assert build_loss("auto") is auto_loss
    logits = torch.randn(2, 3, 8)
    labels = torch.tensor([[1, 2, -100], [3, 4, 5]])
    out = _DummyOutput(loss=None, logits=logits)
    loss = causal_lm_loss(outputs=out, labels=labels, ignore_index=-100)
    assert isinstance(loss, torch.Tensor)
    assert float(loss) > 0.0

    out2 = _DummyOutput(loss=torch.tensor(1.25), logits=logits)
    loss2 = auto_loss(outputs=out2, labels=labels, ignore_index=-100)
    assert float(loss2) == pytest.approx(1.25)


def test_causal_lm_cross_entropy_supports_weighted_loss_scale() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 5.0],
                [0.0, 5.0, 0.0],
                [5.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    weighted = causal_lm_cross_entropy(
        logits=logits,
        labels=labels,
        loss_scale=torch.tensor([[0.0, 0.5, 1.0, 1.0]], dtype=torch.float32),
    )
    unweighted = causal_lm_cross_entropy(logits=logits, labels=labels)
    assert isinstance(weighted, torch.Tensor)
    assert isinstance(unweighted, torch.Tensor)
    assert weighted.ndim == 0
    assert unweighted.ndim == 0
    assert float(weighted) < float(unweighted)


def test_causal_lm_cross_entropy_includes_last_eos_and_shift_is_exact() -> None:
    vocab_size = 8
    labels = torch.tensor([[-100, 3, 4, 2]], dtype=torch.long)
    perfect_logits = torch.full((1, 4, vocab_size), -10.0, dtype=torch.float32)
    perfect_logits[0, 0, 3] = 10.0
    perfect_logits[0, 1, 4] = 10.0
    perfect_logits[0, 2, 2] = 10.0

    misaligned_logits = torch.full((1, 4, vocab_size), -10.0, dtype=torch.float32)
    misaligned_logits[0, 0, 4] = 10.0
    misaligned_logits[0, 1, 2] = 10.0
    misaligned_logits[0, 2, 0] = 10.0

    perfect_loss = causal_lm_cross_entropy(logits=perfect_logits, labels=labels)
    misaligned_loss = causal_lm_cross_entropy(logits=misaligned_logits, labels=labels)

    assert float(perfect_loss) < 1e-3
    assert float(misaligned_loss) > 1.0


def test_global_denominator_is_invariant_to_microbatch_split() -> None:
    torch.manual_seed(17)
    labels = torch.tensor(
        [
            [0, 1, 2, -100],
            [0, 2, 1, 2],
        ],
        dtype=torch.long,
    )
    loss_scale = torch.tensor(
        [
            [0.0, 0.5, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    global_denominator = float(
        (loss_scale[:, 1:] * labels[:, 1:].ne(-100)).sum()
    )
    full_logits = torch.randn(2, 4, 5, requires_grad=True)
    split_logits = full_logits.detach().clone().requires_grad_(True)

    full_loss = causal_lm_cross_entropy(
        logits=full_logits,
        labels=labels,
        loss_scale=loss_scale,
    )
    split_loss = sum(
        causal_lm_cross_entropy(
            logits=split_logits[row : row + 1],
            labels=labels[row : row + 1],
            loss_scale=loss_scale[row : row + 1],
            normalization_denominator=global_denominator,
        )
        for row in range(2)
    )

    full_loss.backward()
    split_loss.backward()

    assert split_loss.detach() == pytest.approx(float(full_loss.detach()))
    assert torch.allclose(split_logits.grad, full_logits.grad, atol=1e-7, rtol=1e-6)
