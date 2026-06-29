from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from shaft.config import load_config
from shaft.data import DPODataset, GRPODataset, SFTDataset, ShaftDatasetBundle
from shaft.pipeline import run_rlhf
from shaft.pipeline.training_args import build_hf_training_args
from tests.support.pipeline import FakePipelineTrainer as _FakeTrainer
from tests.support.pipeline import build_fake_model_artifacts as _build_fake_model_artifacts
from tests.support.rlhf import write_common_image as _write_common_image
from tests.support.rlhf import write_dpo_config as _write_dpo_config
from tests.support.rlhf import write_grpo_config as _write_grpo_config


pytestmark = pytest.mark.component


def test_run_rlhf_builds_training_args_before_model_for_deepspeed(tmp_path: Path) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    cfg.train.distributed.strategy = "deepspeed"
    cfg.train.distributed.deepspeed.config = {
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "zero_optimization": {"stage": 3},
    }
    call_order: list[str] = []

    def _fake_build_training_args(runtime_config):
        assert runtime_config is cfg
        call_order.append("training_args")
        return build_hf_training_args(runtime_config)

    def _fake_build_model(runtime_config, *, init_from_checkpoint=None):
        assert runtime_config is cfg
        assert init_from_checkpoint is None
        call_order.append("model")
        assert call_order == ["training_args", "model"]
        return _build_fake_model_artifacts()

    with patch("shaft.pipeline.rlhf.build_hf_training_args", _fake_build_training_args):
        with patch("shaft.pipeline.rlhf.build_model_tokenizer_processor", _fake_build_model):
            with patch("shaft.algorithms.dpo.ShaftDPOTrainer", _FakeTrainer):
                _ = run_rlhf(cfg)

    assert call_order == ["training_args", "model"]
    assert _FakeTrainer.last_kwargs["args"].deepspeed == cfg.train.distributed.deepspeed.config


def test_run_rlhf_rank_nonzero_skips_run_level_file_ops(tmp_path: Path) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    cfg.train.save_final_model = True
    cfg.train.save_final_state = True

    with patch("shaft.algorithms.grpo.ShaftGRPOTrainer", _FakeTrainer):
        with patch("shaft.pipeline.rlhf.is_rank_zero", return_value=False):
            with patch("shaft.pipeline.rlhf.ensure_hf_export_layout") as mocked_ensure:
                with patch("shaft.pipeline.rlhf.prune_root_output_layout") as mocked_prune:
                    metrics = run_rlhf(cfg)

    assert "train_loss" in metrics
    mocked_ensure.assert_not_called()
    mocked_prune.assert_not_called()


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
            mocked_builder.return_value = _build_fake_model_artifacts()
            with patch("shaft.algorithms.dpo.ShaftDPOTrainer", _FakeTrainer):
                _ = run_rlhf(cfg)

    assert captured["data_config"] is cfg.data
    assert captured["seed"] == cfg.experiment.seed
    assert captured["dataset_cls"] is DPODataset
    assert _FakeTrainer.last_kwargs["train_dataset"] is fake_train_dataset
    assert _FakeTrainer.last_kwargs["train_sampler"] is fake_train_sampler
    assert _FakeTrainer.last_kwargs["eval_dataset"] is None
    assert _FakeTrainer.last_kwargs["model_adapter"] is mocked_builder.return_value.model_adapter
    assert _FakeTrainer.last_kwargs["finetune_plan"] is None


def test_run_rlhf_uses_sft_dataset_for_grpo(tmp_path: Path) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
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
            mocked_builder.return_value = _build_fake_model_artifacts()
            with patch("shaft.algorithms.grpo.ShaftGRPOTrainer", _FakeTrainer):
                _ = run_rlhf(cfg)

    assert captured["dataset_cls"] is SFTDataset
    assert isinstance(_FakeTrainer.last_kwargs["train_dataset"], GRPODataset)
    assert _FakeTrainer.last_kwargs["train_dataset"].dataset is fake_train_dataset
    assert "train_sampler" not in _FakeTrainer.last_kwargs
    assert "finetune_mode" not in _FakeTrainer.last_kwargs
    assert "data_collator" not in _FakeTrainer.last_kwargs
    assert _FakeTrainer.last_kwargs["model_adapter"] is mocked_builder.return_value.model_adapter
    assert _FakeTrainer.last_kwargs["finetune_plan"] is None


def test_run_rlhf_wires_grpo_online_eval_runner_with_named_eval_datasets(
    tmp_path: Path,
) -> None:
    image_path = _write_common_image(tmp_path)
    train_jsonl = tmp_path / "train_grpo.jsonl"
    val_jsonl = tmp_path / "val_grpo.jsonl"
    row = {
        "image_path": str(image_path),
        "target_text": "{\"ok\":1}",
        "user_prompt": "return json",
    }
    train_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    val_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    config_path = tmp_path / "config_grpo_eval.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke-grpo-eval
  output_dir: {tmp_path}/outputs_grpo_eval
  seed: 7
model:
  model_type: smoke_vlm
  finetune:
    mode: lora
    target_modules: ["all-linear"]
algorithm:
  name: grpo
data:
  datasets:
    - dataset_name: grpo_ds
      source_type: jsonl_sft
      train_path: {train_jsonl}
      val_path: {val_jsonl}
  mix_refresh: static
  num_workers: 0
  persistent_workers: false
  pin_memory: false
train:
  epochs: 1
  max_steps: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-3
  save_strategy: no
  report_to: ["none"]
  load_best_model_at_end: false
  save_final_model: false
  save_final_state: false
  bf16: false
  use_cpu: true
eval:
  enabled: true
  loss_metrics_enabled: false
  online_metrics_enabled: true
  metric_for_best_model: eval_final_score
  datasets:
    grpo_ds:
      prediction_codec: json_any
      target_adapter: target_text
      target_adapter_params:
        codec: json_any
      metrics:
        - name: parse_success
        - name: exact_match
      primary_metric: exact_match
rlhf:
  enabled: true
  grpo:
    num_generations: 2
    max_completion_length: 8
    reward_functions:
      - name: exact_match
        codec: json_any
""",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    fake_train_dataset = object()
    fake_eval_dataset = object()
    fake_eval_datasets_by_name = {"grpo_ds": fake_eval_dataset}

    class _FakeDataCenter:
        def __init__(self, data_config, *, seed):
            _ = data_config, seed

        def build_dataset_bundle(self, dataset_cls):
            assert dataset_cls is SFTDataset
            return ShaftDatasetBundle(
                train_dataset=fake_train_dataset,
                eval_dataset=object(),
                eval_datasets_by_name=fake_eval_datasets_by_name,
            )

    with patch("shaft.pipeline.rlhf.ShaftDataCenter", _FakeDataCenter):
        with patch("shaft.pipeline.rlhf.build_model_tokenizer_processor") as mocked_builder:
            mocked_builder.return_value = _build_fake_model_artifacts()
            with patch("shaft.algorithms.grpo.ShaftGRPOTrainer", _FakeTrainer):
                _ = run_rlhf(cfg)

    assert isinstance(_FakeTrainer.last_kwargs["train_dataset"], GRPODataset)
    assert _FakeTrainer.last_kwargs["eval_dataset"] is fake_eval_datasets_by_name
    assert _FakeTrainer.last_kwargs["online_eval_runner"] is not None
    assert _FakeTrainer.last_kwargs["eval_config"] is cfg.eval
