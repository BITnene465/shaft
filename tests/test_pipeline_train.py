from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from shaft.config import RuntimeConfig, load_config
from shaft.pipeline import run_train


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    eos_token = "</s>"

    def __call__(self, texts, add_special_tokens=False, return_attention_mask=False):
        _ = add_special_tokens, return_attention_mask
        return {"input_ids": [[1] for _ in texts]}


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = messages, tokenize, add_generation_prompt
        return "prompt"

    def __call__(self, text, images, padding=True, return_tensors="pt", **kwargs):
        _ = text, images, padding, return_tensors, kwargs
        return {
            "input_ids": torch.tensor([[1], [1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1], [1]], dtype=torch.long),
            "pixel_values": torch.randn(2, 3, 2, 2),
        }


class _FakeModel(torch.nn.Module):
    def forward(self, **kwargs):
        _ = kwargs
        return type("Out", (), {"loss": torch.tensor(0.1)})


class _FakeTrainResult:
    metrics = {"train_loss": 0.1}


class _FakeTrainer:
    last_kwargs = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeTrainer.last_kwargs = kwargs

    def train(self, resume_from_checkpoint=None):
        _ = resume_from_checkpoint
        return _FakeTrainResult()

    def save_model(self):
        return None

    def save_state(self):
        return None


def _write_config(tmp_path: Path, *, hooks: list[str] | None = None) -> RuntimeConfig:
    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl = tmp_path / "val.jsonl"
    image = tmp_path / "img.png"

    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    train_jsonl.write_text(
        f'{{"image_path":"{image}","target_text":"{{}}","user_prompt":"x"}}\n',
        encoding="utf-8",
    )
    val_jsonl.write_text(
        f'{{"image_path":"{image}","target_text":"{{}}","user_prompt":"x"}}\n',
        encoding="utf-8",
    )
    hooks_yaml = f"  hooks: {hooks}\n" if hooks is not None else ""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
experiment:
  name: test
  output_dir: {tmp_path}/out
data:
  datasets:
    - name: ds
      train_path: {train_jsonl}
      val_path: {val_jsonl}
algorithm:
  name: sft
plugins:
{hooks_yaml if hooks_yaml else '  hooks: []'}
train:
  epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-5
  use_cpu: true
  report_to: ["none"]
eval:
  enabled: false
""",
        encoding="utf-8",
    )
    return load_config(cfg_path)


def test_run_train_smoke(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    with patch("shaft.pipeline.train.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = type(
            "Artifacts",
            (),
            {"model": _FakeModel(), "tokenizer": _FakeTokenizer(), "processor": _FakeProcessor()},
        )()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            metrics = run_train(config)
    assert "train_loss" in metrics


def test_hooks_are_wired_into_trainer_callbacks(tmp_path: Path) -> None:
    config = _write_config(tmp_path, hooks=["log_on_save"])
    with patch("shaft.pipeline.train.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = type(
            "Artifacts",
            (),
            {"model": _FakeModel(), "tokenizer": _FakeTokenizer(), "processor": _FakeProcessor()},
        )()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            _ = run_train(config)
    callbacks = _FakeTrainer.last_kwargs.get("callbacks")
    assert callbacks is not None
    assert len(callbacks) >= 1
