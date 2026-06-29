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
