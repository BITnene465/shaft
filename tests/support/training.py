from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import TrainingArguments


class DummyOutput:
    def __init__(self, loss: torch.Tensor | None = None, logits: torch.Tensor | None = None):
        self.loss = loss
        self.logits = logits


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, 8)
        self.fc = torch.nn.Linear(8, vocab_size)
        self.config = type("Cfg", (), {"hidden_size": 8})()
        self.last_forward_kwargs = None
        self.last_forward_labels = None

    def forward(self, input_ids=None, labels=None, **kwargs):
        self.last_forward_kwargs = dict(kwargs)
        self.last_forward_labels = labels
        hidden = self.emb(input_ids)
        logits = self.fc(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return DummyOutput(loss=loss, logits=logits)


def build_training_args(output_dir: str | Path, **overrides: Any) -> TrainingArguments:
    values: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": 1e-3,
        "per_device_train_batch_size": 1,
        "use_cpu": True,
        "report_to": [],
    }
    values.update(overrides)
    return TrainingArguments(**values)


class StaticOnlineEvalRunner:
    def __init__(self, metrics: dict[str, float]):
        self.metrics = dict(metrics)

    def evaluate(self, trainer, *, eval_dataset, metric_key_prefix="eval"):
        _ = trainer, eval_dataset, metric_key_prefix
        return dict(self.metrics)


def eval_loop_output(metrics: dict[str, float], *, num_samples: int = 1) -> SimpleNamespace:
    return SimpleNamespace(metrics=dict(metrics), num_samples=num_samples)


def capture_trainer_logs(trainer) -> list[dict[str, float]]:
    logged: list[dict[str, float]] = []
    trainer.log = lambda metrics, start_time=None: logged.append(dict(metrics))  # type: ignore[method-assign]
    return logged
