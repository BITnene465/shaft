from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from shaft.config import load_config
from shaft.data import DPODataset, GRPODataset, SFTDataset, ShaftDatasetBundle
from shaft.model import build_model_meta
from shaft.algorithms.rlhf_utils import build_trl_dpo_config, build_trl_grpo_config
from shaft.pipeline import run_rlhf
from shaft.pipeline.training_args import build_hf_training_args
from shaft.template import build_template


def _fsdp_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value)


def _fsdp_option_values(value) -> list[str]:
    if isinstance(value, bool):
        return []
    return [getattr(option, "value", str(option)) for option in value]


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

    def save_model(self, *args, **kwargs):
        _ = args, kwargs
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


def _write_grpo_config(base_dir: Path) -> Path:
    image_path = _write_common_image(base_dir)
    train_jsonl = base_dir / "train_grpo.jsonl"
    val_jsonl = base_dir / "val_grpo.jsonl"
    row = {
        "image_path": str(image_path),
        "target_text": "{\"ok\":1}",
        "user_prompt": "return json",
    }
    train_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    val_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = base_dir / "config_grpo.yaml"
    cfg.write_text(
        f"""
experiment:
  name: smoke-grpo
  output_dir: {base_dir}/outputs_grpo
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
  grpo:
    num_generations: 2
    max_completion_length: 8
    reward_functions:
      - name: exact_match
        codec: json_any
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


def test_run_rlhf_grpo_smoke(tmp_path: Path) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    with patch("shaft.algorithms.grpo.ShaftGRPOTrainer", _FakeTrainer):
        metrics = run_rlhf(cfg)
    assert "train_loss" in metrics


def test_dpo_trl_config_preserves_deepspeed_args(tmp_path: Path) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    cfg.train.distributed.strategy = "deepspeed"
    cfg.train.distributed.deepspeed.config = {
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "zero_optimization": {"stage": 2},
    }

    train_args = build_hf_training_args(cfg)
    dpo_args = build_trl_dpo_config(train_args=train_args, rlhf_config=cfg.rlhf.dpo)

    assert dpo_args.deepspeed == cfg.train.distributed.deepspeed.config
    assert getattr(dpo_args, "hf_deepspeed_config", None) is not None


def test_grpo_trl_config_preserves_fsdp_args(tmp_path: Path) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    cfg.train.distributed.strategy = "fsdp"
    cfg.train.distributed.fsdp.auto_wrap_policy = "none"

    train_args = build_hf_training_args(cfg)
    grpo_args = build_trl_grpo_config(train_args=train_args, rlhf_config=cfg.rlhf.grpo)

    assert _fsdp_enabled(grpo_args.fsdp) is True
    option_values = _fsdp_option_values(grpo_args.fsdp)
    if option_values:
        assert option_values == ["full_shard"]
    assert grpo_args.fsdp_config["activation_checkpointing"] is True


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
        return type(
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


def test_run_rlhf_wires_grpo_online_eval_runner_with_named_eval_datasets(tmp_path: Path) -> None:
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
            with patch("shaft.algorithms.grpo.ShaftGRPOTrainer", _FakeTrainer):
                _ = run_rlhf(cfg)

    assert isinstance(_FakeTrainer.last_kwargs["train_dataset"], GRPODataset)
    assert _FakeTrainer.last_kwargs["eval_dataset"] is fake_eval_datasets_by_name
    assert _FakeTrainer.last_kwargs["online_eval_runner"] is not None
    assert _FakeTrainer.last_kwargs["eval_config"] is cfg.eval


def test_grpo_dataset_applies_image_pixel_budget() -> None:
    image = Image.new("RGB", (100, 50), color=(255, 255, 255))

    class _SingleImageDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "image": image,
                "target_text": "{\"ok\":1}",
                "user_prompt": "return json",
                "dataset_name": "grpo_ds",
                "sample_id": "sample-1",
                "image_path": "/tmp/sample.png",
                "extra": {},
            }

    dataset = GRPODataset(
        _SingleImageDataset(),
        template=build_template("smoke_vlm"),
        max_pixels=2000,
    )

    sample = dataset[0]

    assert sample["image"].size[0] * sample["image"].size[1] <= 2000
    assert sample["image"].size != image.size
    assert image.size == (100, 50)
