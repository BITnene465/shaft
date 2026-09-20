from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from shaft.training.linear_ce import linear_causal_cross_entropy
from shaft.config.training import TrainLigerConfig


@pytest.fixture(autouse=True)
def reference_kernel(monkeypatch):
    # CPU contract tests use an independent dense oracle; CUDA tests exercise Liger itself.
    def kernel(hidden, weight, target, *, bias, ignore_index, reduction, accum_dtype):
        return F.cross_entropy(
            F.linear(hidden, weight, bias),
            target,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    monkeypatch.setattr("shaft.training.linear_ce.load_fused_linear_ce", lambda: kernel)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("external_denominator", [None, 17.0])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_linear_ce_matches_dense_loss_and_gradients(weighted, external_denominator, all_ignored):
    torch.manual_seed(73)
    hidden = torch.randn(2, 7, 5, dtype=torch.float64, requires_grad=True)
    head = torch.nn.Linear(5, 11, dtype=torch.float64)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_head = deepcopy(head)
    labels = torch.tensor([[-100, -100, 3, 2, 10, -100, -100], [-100, 2, 3, 4, 5, 6, 1]])
    if all_ignored:
        labels.fill_(-100)
    scales = torch.rand(2, 7, dtype=torch.float64) if weighted else None
    weights = labels[:, 1:].ne(-100).double()
    if weighted:
        weights *= scales[:, 1:]
    denom = weights.sum() if external_denominator is None else external_denominator
    logits = reference_head(reference_hidden)
    losses = F.cross_entropy(
        logits[:, :-1].reshape(-1, 11),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(2, 6)
    expected = (losses * weights).sum() / max(float(denom), 1e-20)
    actual = linear_causal_cross_entropy(
        hidden_states=hidden,
        lm_head=head,
        labels=labels,
        loss_scale=scales,
        normalization_denominator=external_denominator,
    )
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    for parameter, reference in zip(head.parameters(), reference_head.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, reference.grad)


def test_linear_ce_never_projects_ignored_positions(monkeypatch):
    hidden = torch.randn(2, 20, 8, requires_grad=True)
    head = torch.nn.Linear(8, 31, bias=False)
    labels = torch.full((2, 20), -100)
    labels[:, -4:] = 2
    sizes = []
    original = F.linear

    def observe(inputs, *args, **kwargs):
        sizes.append(inputs.shape[0])
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(F, "linear", observe)
    linear_causal_cross_entropy(
        hidden_states=hidden,
        lm_head=head,
        labels=labels,
    ).backward()
    assert sizes == [8]  # Only supervised positions reach the fused kernel.


@pytest.mark.parametrize("liger", [False, True])
def test_sft_algorithm_routes_fused_ce_and_records_contract(tmp_path, liger):
    from shaft.algorithms.base import AlgorithmContext
    from shaft.algorithms.sft import SFTAlgorithm
    from shaft.config import TrainConfig
    from tests.support.pipeline import build_fake_model_artifacts
    from tests.support.training import build_training_args

    spec = SFTAlgorithm().prepare_trainer(
        context=AlgorithmContext(params={}),
        train_config=TrainConfig(liger=TrainLigerConfig(
            fused_linear_ce=True, rms_norm=liger, swiglu=liger,
        )),
        model_adapter=build_fake_model_artifacts().model_adapter,
        resolved_optimizer_plan=None,
        args=build_training_args(tmp_path),
    )
    assert spec.kwargs["liger_config"].fused_linear_ce is True
    assert spec.contract["extra"]["liger"] == {
        "fused_linear_ce": True, "rms_norm": liger, "swiglu": liger,
    }


def test_fused_ce_rejects_unsupported_model_policies():
    from shaft.model.types import TrainingObjectivePolicy
    from shaft.model.qwen35_loss import Qwen35VLTrainingObjectivePolicy

    with pytest.raises(ValueError, match="does not support"):
        TrainingObjectivePolicy().enable_fused_linear_ce(torch.nn.Linear(3, 4))
    with pytest.raises(ValueError, match="native dense Qwen3.5"):
        Qwen35VLTrainingObjectivePolicy().enable_fused_linear_ce(torch.nn.Linear(3, 4))


def test_fused_ce_rejects_head_only_training():
    with pytest.raises(ValueError, match="head-only training"):
        linear_causal_cross_entropy(
            hidden_states=torch.randn(2, 3, 5),
            lm_head=torch.nn.Linear(5, 7),
            labels=torch.ones(2, 3, dtype=torch.long),
        )
