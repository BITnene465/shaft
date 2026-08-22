from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

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
    global_denominator = float((loss_scale[:, 1:] * labels[:, 1:].ne(-100)).sum())
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


def test_chunked_causal_lm_cross_entropy_matches_reference_value_and_gradient() -> None:
    torch.manual_seed(31)
    labels = torch.tensor(
        [
            [-100, 1, 2, 3, -100, 4],
            [-100, 3, 1, 2, 4, 0],
        ],
        dtype=torch.long,
    )
    loss_scale = torch.tensor(
        [
            [0.0, 0.5, 1.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 0.25, 1.0, 1.5, 2.0],
        ],
        dtype=torch.float64,
    )
    normalization_denominator = torch.tensor(12.0, dtype=torch.float64)
    reference_logits = torch.randn(2, 6, 7, dtype=torch.float64, requires_grad=True)
    chunked_logits = reference_logits.detach().clone().requires_grad_(True)

    shift_labels = labels[:, 1:]
    reference_token_loss = F.cross_entropy(
        reference_logits[:, :-1, :].reshape(-1, 7),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    weights = loss_scale[:, 1:] * shift_labels.ne(-100)
    weighted_reference = reference_token_loss * weights
    reference_loss = weighted_reference.sum() / normalization_denominator

    components: dict[str, torch.Tensor] = {}
    chunked_loss = causal_lm_cross_entropy(
        logits=chunked_logits,
        labels=labels,
        loss_scale=loss_scale,
        normalization_denominator=normalization_denominator,
        component_output=components,
        max_tokens_per_chunk=3,
    )

    torch.testing.assert_close(chunked_loss, reference_loss)
    torch.testing.assert_close(
        components["numerator"],
        weighted_reference.detach().sum(dim=-1),
    )
    torch.testing.assert_close(
        components["denominator"],
        weights.sum(dim=-1),
    )

    reference_loss.backward()
    chunked_loss.backward()
    torch.testing.assert_close(chunked_logits.grad, reference_logits.grad)


def test_chunked_causal_lm_cross_entropy_bounds_each_ce_workspace(monkeypatch) -> None:
    torch.manual_seed(37)
    logits = torch.randn(2, 6, 11, requires_grad=True)
    labels = torch.tensor(
        [
            [-100, 1, 2, 3, 4, 5],
            [-100, 5, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )
    calls: list[int] = []
    original_cross_entropy = F.cross_entropy

    def recording_cross_entropy(input_tensor, *args, **kwargs):
        calls.append(int(input_tensor.shape[0]))
        return original_cross_entropy(input_tensor, *args, **kwargs)

    monkeypatch.setattr("shaft.training.loss.F.cross_entropy", recording_cross_entropy)
    loss = causal_lm_cross_entropy(
        logits=logits,
        labels=labels,
        max_tokens_per_chunk=3,
    )
    forward_call_count = len(calls)
    loss.backward()

    assert forward_call_count > 1
    assert len(calls) == 2 * forward_call_count
    assert max(calls) <= 3
