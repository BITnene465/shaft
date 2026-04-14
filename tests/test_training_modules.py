from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from transformers import TrainingArguments

from shaft.training.loss import LOSS_REGISTRY, auto_loss, build_loss, causal_lm_loss
from shaft.training.muon import Muon
from shaft.training.optimizer import OPTIMIZER_REGISTRY, build_optimizer
from shaft.training.scheduler import SCHEDULER_REGISTRY, build_scheduler
from shaft.training.trainer import ShaftSFTTrainer


class _DummyOutput:
    def __init__(self, loss: torch.Tensor | None = None, logits: torch.Tensor | None = None):
        self.loss = loss
        self.logits = logits


class _TinyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, 8)
        self.fc = torch.nn.Linear(8, vocab_size)

    def forward(self, input_ids=None, labels=None, **kwargs):
        _ = kwargs
        hidden = self.emb(input_ids)
        logits = self.fc(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return _DummyOutput(loss=loss, logits=logits)


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


def test_optimizer_and_scheduler() -> None:
    assert OPTIMIZER_REGISTRY.has("adamw_torch")
    assert OPTIMIZER_REGISTRY.has("muon")
    assert SCHEDULER_REGISTRY.has("cosine")
    assert SCHEDULER_REGISTRY.has("cosine_with_restarts")
    assert SCHEDULER_REGISTRY.has("polynomial")
    model = _TinyModel()
    args = TrainingArguments(
        output_dir="/tmp/shaft_training_modules",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to=[],
    )
    optimizer = build_optimizer(
        model=model,
        args=args,
        optimizer_name="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
    )
    assert isinstance(optimizer, torch.optim.Optimizer)
    scheduler = build_scheduler(
        scheduler_name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=10,
    )
    assert scheduler is not None

    scheduler_restart = build_scheduler(
        scheduler_name="cosine_with_restarts",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=10,
        num_cycles=2.0,
    )
    assert scheduler_restart is not None

    scheduler_poly = build_scheduler(
        scheduler_name="polynomial",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=10,
        power=2.0,
    )
    assert scheduler_poly is not None

    muon = build_optimizer(
        model=model,
        args=args,
        optimizer_name="muon",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
    )
    assert isinstance(muon, Muon)


def test_shaft_trainer_uses_custom_components() -> None:
    model = _TinyModel()
    args = TrainingArguments(
        output_dir="/tmp/shaft_trainer_smoke",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to=[],
    )
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset=[],
        data_collator=lambda x: x,
        loss_name="causal_lm",
        optimizer_name="adamw_torch",
        scheduler_name="linear",
        scheduler_num_cycles=2.0,
        scheduler_power=1.5,
    )
    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], device=device),
        "labels": torch.tensor([[1, 2, 3]], device=device),
    }
    with patch("shaft.training.trainer.build_optimizer") as mocked_build_optim:
        mocked_build_optim.return_value = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer.create_optimizer()
        mocked_build_optim.assert_called_once()
    with patch("shaft.training.trainer.build_scheduler") as mocked_build_sched:
        mocked_build_sched.return_value = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
        trainer.create_scheduler(10)
        mocked_build_sched.assert_called_once()
        _, kwargs = mocked_build_sched.call_args
        assert kwargs["num_cycles"] == pytest.approx(2.0)
        assert kwargs["power"] == pytest.approx(1.5)
    loss = trainer.compute_loss(model, inputs)
    assert isinstance(loss, torch.Tensor)
