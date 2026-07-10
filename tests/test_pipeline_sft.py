from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from shaft.config import load_config
from shaft.data import SFTDataset, ShaftDatasetBundle
from shaft.model.finetune_plan import resolved_finetune_summary_path
from shaft.pipeline import run_sft
from tests.support.pipeline import FakePipelineModel as _FakeModel
from tests.support.pipeline import FakePipelineTrainer as _FakeTrainer
from tests.support.pipeline import build_fake_model_artifacts as _build_fake_model_artifacts
from tests.support.pipeline import write_sft_pipeline_config as _write_config


pytestmark = pytest.mark.component


def test_run_sft_smoke(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    fake_model = _FakeModel()
    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts(
            model=fake_model,
            include_finetune_plan=True,
        )
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            metrics = run_sft(config)
    assert "train_loss" in metrics
    assert fake_model.generation_config.do_sample is False
    assert fake_model.generation_config.temperature == 1.0
    assert fake_model.generation_config.top_p == 1.0
    assert fake_model.generation_config.top_k == 50
    assert fake_model.generation_config.eos_token_id == [2, 99]
    assert fake_model.generation_config.bos_token_id is None
    assert fake_model.generation_config.pad_token_id == 0
    assert fake_model.config.eos_token_id == 2
    assert fake_model.config.bos_token_id is None
    assert fake_model.config.pad_token_id == 0
    assert fake_model.config.text_config.eos_token_id == 99
    assert fake_model.config.text_config.bos_token_id == 98
    assert fake_model.config.text_config.pad_token_id == 97
    assert resolved_finetune_summary_path(config.experiment.output_dir).exists()


def test_run_sft_rank_nonzero_skips_run_level_file_ops(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.save_final_model = True
    config.train.save_final_state = True

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts(include_finetune_plan=True)
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            with patch("shaft.pipeline.sft.is_rank_zero", return_value=False):
                with patch("shaft.pipeline.sft.ensure_hf_export_layout") as mocked_ensure:
                    with patch("shaft.pipeline.sft.prune_root_output_layout") as mocked_prune:
                        metrics = run_sft(config)

    assert "train_loss" in metrics
    mocked_ensure.assert_not_called()
    mocked_prune.assert_not_called()


def test_run_sft_wires_loss_scale_into_train_collator(tmp_path: Path) -> None:
    config = _write_config(tmp_path, loss_scale="all")
    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            _ = run_sft(config)
    collator = _FakeTrainer.last_kwargs["data_collator"]
    assert collator.loss_scale_name == "all"


def test_hooks_are_wired_into_trainer_callbacks(tmp_path: Path) -> None:
    config = _write_config(tmp_path, hooks=["log_on_save"])
    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            _ = run_sft(config)
    callbacks = _FakeTrainer.last_kwargs.get("callbacks")
    assert callbacks is not None
    assert len(callbacks) >= 1


def test_run_sft_uses_data_center(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    fake_train_dataset = object()
    fake_eval_dataset = object()
    fake_train_sampler = object()
    captured = {}

    class _FakeDataCenter:
        def __init__(self, data_config, *, seed, train_sample_budget):
            captured["data_config"] = data_config
            captured["seed"] = seed
            captured["train_sample_budget"] = train_sample_budget

        def build_dataset_bundle(self, dataset_cls):
            captured["dataset_cls"] = dataset_cls
            return ShaftDatasetBundle(
                train_dataset=fake_train_dataset,
                eval_dataset=fake_eval_dataset,
                train_sampler=fake_train_sampler,
            )

    with patch("shaft.pipeline.sft.ShaftDataCenter", _FakeDataCenter):
        with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
            mocked_builder.return_value = _build_fake_model_artifacts()
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                _ = run_sft(config)

    assert captured["data_config"] is config.data
    assert captured["seed"] == config.experiment.seed
    assert captured["train_sample_budget"] == 1
    assert captured["dataset_cls"] is SFTDataset
    assert _FakeTrainer.last_kwargs["train_dataset"] is fake_train_dataset
    assert _FakeTrainer.last_kwargs["train_sampler"] is fake_train_sampler
    assert _FakeTrainer.last_kwargs["eval_dataset"] is None
    assert _FakeTrainer.last_kwargs["model_adapter"] is mocked_builder.return_value.model_adapter
    assert _FakeTrainer.last_kwargs["finetune_plan"] is None
    callbacks = _FakeTrainer.last_kwargs.get("callbacks")
    assert callbacks is not None
    assert all(callback.__class__.__name__ != "ShaftMixRefreshCallback" for callback in callbacks)


def test_run_sft_wires_online_eval_runner(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl = tmp_path / "val.jsonl"
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    train_jsonl.write_text(
        f'{{"image_path":"{image}","target_text":"{{\\"ok\\":1}}","user_prompt":"x"}}\n',
        encoding="utf-8",
    )
    val_jsonl.write_text(
        f'{{"image_path":"{image}","target_text":"{{\\"ok\\":1}}","user_prompt":"x"}}\n',
        encoding="utf-8",
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
experiment:
  name: test-online-eval
  output_dir: {tmp_path}/out
data:
  datasets:
    - dataset_name: ds
      train_path: {train_jsonl}
      val_path: {val_jsonl}
algorithm:
  name: sft
train:
  duration:
    unit: steps
    value: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-5
  save_epoch_interval: 2
  use_cpu: true
  report_to: ["none"]
  load_best_model_at_end: false
  save_final_model: false
  save_final_state: false
eval:
  enabled: true
  epoch_interval: 2
  online_metrics_enabled: true
  metric_for_best_model: eval_final_score
  greater_is_better: true
  datasets:
    ds:
      prediction_codec: json_object
      target_adapter: target_text
      target_adapter_params:
        codec: json_object
      metrics:
        - name: parse_success
        - name: exact_match
      primary_metric: exact_match
      normalizer:
        type: identity
      weight: 1.0
""",
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            _ = run_sft(config)
    assert isinstance(_FakeTrainer.last_kwargs["eval_dataset"], dict)
    assert set(_FakeTrainer.last_kwargs["eval_dataset"].keys()) == {"ds"}
    assert _FakeTrainer.last_kwargs["online_eval_runner"] is not None
    assert _FakeTrainer.last_kwargs["eval_config"] is config.eval
    assert _FakeTrainer.last_kwargs["online_eval_runner"].prompt_collator.padding_side == "left"
    callbacks = _FakeTrainer.last_kwargs.get("callbacks")
    assert callbacks is not None
    assert any(callback.__class__.__name__ == "ShaftEpochIntervalCallback" for callback in callbacks)


def test_run_sft_keeps_merged_eval_dataset_for_legacy_eval_loss_mode(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl = tmp_path / "val.jsonl"
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    train_jsonl.write_text(
        f'{{"image_path":"{image}","target_text":"{{\\"ok\\":1}}","user_prompt":"x"}}\n',
        encoding="utf-8",
    )
    val_jsonl.write_text(
        f'{{"image_path":"{image}","target_text":"{{\\"ok\\":1}}","user_prompt":"x"}}\n',
        encoding="utf-8",
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
experiment:
  name: test-legacy-eval-loss
  output_dir: {tmp_path}/out
data:
  datasets:
    - dataset_name: ds
      train_path: {train_jsonl}
      val_path: {val_jsonl}
algorithm:
  name: sft
train:
  duration:
    unit: steps
    value: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-5
  use_cpu: true
  report_to: ["none"]
  load_best_model_at_end: false
  save_final_model: false
  save_final_state: false
eval:
  enabled: true
  loss_metrics_enabled: false
  online_metrics_enabled: false
  metric_for_best_model: eval_loss
  datasets:
    ds:
      weight: 1.0
""",
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            _ = run_sft(config)
    assert not isinstance(_FakeTrainer.last_kwargs["eval_dataset"], dict)
    assert _FakeTrainer.last_kwargs["online_eval_runner"] is None


def test_run_sft_keeps_merged_eval_dataset_when_eval_policies_are_absent(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl = tmp_path / "val.jsonl"
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    train_jsonl.write_text(
        f'{{"image_path":"{image}","target_text":"{{\\"ok\\":1}}","user_prompt":"x"}}\n',
        encoding="utf-8",
    )
    val_jsonl.write_text(
        f'{{"image_path":"{image}","target_text":"{{\\"ok\\":1}}","user_prompt":"x"}}\n',
        encoding="utf-8",
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
experiment:
  name: test-plain-eval-loss
  output_dir: {tmp_path}/out
data:
  datasets:
    - dataset_name: ds
      train_path: {train_jsonl}
      val_path: {val_jsonl}
algorithm:
  name: sft
train:
  duration:
    unit: steps
    value: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-5
  use_cpu: true
  report_to: ["none"]
  load_best_model_at_end: false
  save_final_model: false
  save_final_state: false
eval:
  enabled: true
  online_metrics_enabled: false
  metric_for_best_model: eval_loss
""",
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            _ = run_sft(config)
    assert not isinstance(_FakeTrainer.last_kwargs["eval_dataset"], dict)
    assert _FakeTrainer.last_kwargs["online_eval_runner"] is None
