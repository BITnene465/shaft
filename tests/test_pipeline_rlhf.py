from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from shaft.config import load_config
from shaft.data import DPODataset, ShaftDatasetBundle
from shaft.model import build_model_meta
from shaft.pipeline import run_rlhf
from shaft.template import build_template


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    eos_token = "</s>"


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


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


def _write_common_image(base_dir: Path) -> Path:
    image_path = base_dir / "image.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_path)
    return image_path


def _write_dpo_config(base_dir: Path) -> Path:
    image_path = _write_common_image(base_dir)
    train_jsonl = base_dir / "train_dpo.jsonl"
    val_jsonl = base_dir / "val_dpo.jsonl"
    row = {
        "image_path": str(image_path),
        "chosen_text": "{\"ok\":1}",
        "rejected_text": "{\"ok\":0}",
        "user_prompt": "return json",
    }
    train_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    val_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = base_dir / "config_dpo.yaml"
    cfg.write_text(
        f"""
experiment:
  name: smoke-dpo
  output_dir: {base_dir}/outputs_dpo
  seed: 7
model:
  model_type: smoke_vlm
  finetune:
    mode: lora
    target_modules: ["all-linear"]
algorithm:
  name: dpo
data:
  datasets:
    - dataset_name: dpo_ds
      source_type: jsonl_dpo
      train_path: {train_jsonl}
      val_path: {val_jsonl}
  num_workers: 0
  persistent_workers: false
  pin_memory: false
  min_pixels:
  max_pixels:
train:
  epochs: 1
  max_steps: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-3
  optimizer_name: adamw_torch
  scheduler_name: linear
  loss_name: auto
  logging_steps: 1
  save_strategy: no
  report_to: ["none"]
  load_best_model_at_end: false
  save_final_model: false
  save_final_state: false
  bf16: false
  use_cpu: true
eval:
  enabled: false
rlhf:
  enabled: true
  dpo:
    precompute_ref_log_probs: false
""",
        encoding="utf-8",
    )
    return cfg


def _write_ppo_config(base_dir: Path) -> Path:
    image_path = _write_common_image(base_dir)
    train_jsonl = base_dir / "train_ppo.jsonl"
    val_jsonl = base_dir / "val_ppo.jsonl"
    row = {
        "image_path": str(image_path),
        "prompt": "return json",
        "user_prompt": "return json",
    }
    train_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    val_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = base_dir / "config_ppo.yaml"
    cfg.write_text(
        f"""
experiment:
  name: smoke-ppo
  output_dir: {base_dir}/outputs_ppo
  seed: 7
model:
  model_type: smoke_vlm
  finetune:
    mode: lora
    target_modules: ["all-linear"]
algorithm:
  name: ppo
data:
  datasets:
    - dataset_name: ppo_ds
      source_type: jsonl_ppo
      train_path: {train_jsonl}
      val_path: {val_jsonl}
  num_workers: 0
  persistent_workers: false
  pin_memory: false
  min_pixels:
  max_pixels:
train:
  epochs: 1
  max_steps: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-3
  optimizer_name: adamw_torch
  scheduler_name: linear
  loss_name: auto
  logging_steps: 1
  save_strategy: no
  report_to: ["none"]
  load_best_model_at_end: false
  save_final_model: false
  save_final_state: false
  bf16: false
  use_cpu: true
eval:
  enabled: false
rlhf:
  enabled: true
  ppo:
    response_length: 4
    num_ppo_epochs: 1
    num_mini_batches: 1
    local_rollout_forward_batch_size: 1
    num_sample_generations: 0
    allow_untrained_reward_model: true
    allow_text_only_multimodal_ppo: true
""",
        encoding="utf-8",
    )
    return cfg


def test_run_rlhf_dpo_smoke(tmp_path: Path) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    metrics = run_rlhf(cfg)
    assert "train_loss" in metrics


def test_run_rlhf_ppo_smoke(tmp_path: Path) -> None:
    cfg = load_config(_write_ppo_config(tmp_path))
    metrics = run_rlhf(cfg)
    assert "episode" in metrics
    assert "objective/rlhf_reward" in metrics


def test_run_rlhf_uses_data_center_for_dpo(tmp_path: Path) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    fake_train_dataset = object()
    fake_eval_dataset = object()
    fake_train_sampler = object()
    captured = {}

    class _FakeDataCenter:
        def __init__(self, data_config, *, seed):
            captured["data_config"] = data_config
            captured["seed"] = seed

        def build_dataset_bundle(self, dataset_cls):
            captured["dataset_cls"] = dataset_cls
            return ShaftDatasetBundle(
                train_dataset=fake_train_dataset,
                eval_dataset=fake_eval_dataset,
                train_sampler=fake_train_sampler,
            )

    with patch("shaft.pipeline.rlhf.ShaftDataCenter", _FakeDataCenter):
        with patch("shaft.pipeline.rlhf.build_model_tokenizer_processor") as mocked_builder:
            mocked_builder.return_value = type(
                "Artifacts",
                (),
                {
                    "model": _FakeModel(),
                    "tokenizer": _FakeTokenizer(),
                    "processor": _FakeProcessor(),
                    "model_meta": build_model_meta("smoke_vlm"),
                    "model_adapter": build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM"),
                    "template": build_template("smoke_vlm"),
                },
            )()
            with patch("shaft.algorithms.dpo.ShaftDPOTrainer", _FakeTrainer):
                _ = run_rlhf(cfg)

    assert captured["data_config"] is cfg.data
    assert captured["seed"] == cfg.experiment.seed
    assert captured["dataset_cls"] is DPODataset
    assert _FakeTrainer.last_kwargs["train_dataset"] is fake_train_dataset
    assert _FakeTrainer.last_kwargs["train_sampler"] is fake_train_sampler
    assert _FakeTrainer.last_kwargs["eval_dataset"] is None
