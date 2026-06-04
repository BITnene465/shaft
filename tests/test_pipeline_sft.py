from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from shaft.config import FinetuneConfig, RuntimeConfig, load_config
from shaft.data import SFTDataset, ShaftDatasetBundle
from shaft.model import build_model_meta
from shaft.model.finetune_plan import build_resolved_finetune_plan, resolved_finetune_summary_path
from shaft.pipeline.training_args import (
    _build_deepspeed_arg,
    _build_fsdp_args,
    build_hf_training_args,
)
from shaft.pipeline import run_sft
from shaft.template import build_template
from shaft.training.topology import validate_training_topology


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
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.generation_config = SimpleNamespace(
            use_cache=False,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
        )

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


def _write_config(
    tmp_path: Path,
    *,
    hooks: list[str] | None = None,
    loss_scale: str = "default",
) -> RuntimeConfig:
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
    - dataset_name: ds
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
  loss_scale: {loss_scale}
  use_cpu: true
  report_to: ["none"]
  load_best_model_at_end: false
  save_final_model: false
  save_final_state: false
eval:
  enabled: false
""",
        encoding="utf-8",
    )
    return load_config(cfg_path)


def test_run_sft_smoke(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    fake_model = _FakeModel()
    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = type(
            "Artifacts",
            (),
            {
                "model": fake_model,
                "tokenizer": _FakeTokenizer(),
                "processor": _FakeProcessor(),
                "model_meta": build_model_meta("smoke_vlm"),
                "model_adapter": adapter,
                "template": build_template("smoke_vlm"),
                "finetune_plan": build_resolved_finetune_plan(
                    _FakeModel(),
                    FinetuneConfig(mode="full"),
                    model_adapter=adapter,
                ),
            },
        )()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            metrics = run_sft(config)
    assert "train_loss" in metrics
    assert fake_model.generation_config.do_sample is False
    assert fake_model.generation_config.temperature == 1.0
    assert fake_model.generation_config.top_p == 1.0
    assert fake_model.generation_config.top_k == 50
    assert resolved_finetune_summary_path(config.experiment.output_dir).exists()


def test_run_sft_rank_nonzero_skips_run_level_file_ops(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.save_final_model = True
    config.train.save_final_state = True
    adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = type(
            "Artifacts",
            (),
            {
                "model": _FakeModel(),
                "tokenizer": _FakeTokenizer(),
                "processor": _FakeProcessor(),
                "model_meta": build_model_meta("smoke_vlm"),
                "model_adapter": adapter,
                "template": build_template("smoke_vlm"),
                "finetune_plan": build_resolved_finetune_plan(
                    _FakeModel(),
                    FinetuneConfig(mode="full"),
                    model_adapter=adapter,
                ),
            },
        )()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            with patch("shaft.pipeline.sft.is_rank_zero", return_value=False):
                with patch("shaft.pipeline.sft.ensure_hf_export_layout") as mocked_ensure:
                    with patch("shaft.pipeline.sft.prune_root_output_layout") as mocked_prune:
                        metrics = run_sft(config)

    assert "train_loss" in metrics
    mocked_ensure.assert_not_called()
    mocked_prune.assert_not_called()


def test_build_hf_training_args_supports_gradient_checkpointing(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.gradient_checkpointing = True

    args = build_hf_training_args(config)

    assert args.gradient_checkpointing is True


def test_build_hf_training_args_supports_fsdp_strategy(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.distributed.strategy = "fsdp"
    config.train.distributed.fsdp.transformer_layer_cls_to_wrap = ["auto"]

    args = build_hf_training_args(config)

    assert _fsdp_enabled(args.fsdp) is True
    option_values = _fsdp_option_values(args.fsdp)
    if option_values:
        assert option_values == ["full_shard", "auto_wrap"]
    assert args.fsdp_config["transformer_layer_cls_to_wrap"] == [
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLVisionBlock",
    ]
    assert args.fsdp_config["activation_checkpointing"] is True
    assert args.fsdp_config["state_dict_type"] == "full_state_dict"


def test_fsdp_activation_checkpointing_disables_trainer_gradient_checkpointing(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.train.gradient_checkpointing = True
    config.train.distributed.strategy = "fsdp"
    config.train.distributed.fsdp.activation_checkpointing = True
    config.train.distributed.fsdp.transformer_layer_cls_to_wrap = ["auto"]

    args = build_hf_training_args(config)

    assert args.gradient_checkpointing is False
    assert args.fsdp_config["activation_checkpointing"] is True


def test_build_hf_training_args_resolves_qwen36vl_fsdp_auto_layers(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "models/Qwen3.6-27B"
    config.train.distributed.strategy = "fsdp"
    config.train.distributed.fsdp.transformer_layer_cls_to_wrap = ["auto"]

    args = build_hf_training_args(config)

    assert _fsdp_enabled(args.fsdp) is True
    assert args.fsdp_config["transformer_layer_cls_to_wrap"] == [
        "Qwen3_5DecoderLayer",
        "Qwen3_5VisionBlock",
    ]


def test_qwen36_sft_27b_fsdp_example_config_loads() -> None:
    config = load_config(Path("configs/train/qwen36_sft_27b_fsdp_example.yaml"))

    assert config.model.model_type == "qwen36vl"
    assert config.model.template == "qwen35vl"
    assert config.model.finetune.mode == "lora"
    assert config.train.distributed.strategy == "fsdp"

    args = build_hf_training_args(config)
    assert _fsdp_enabled(args.fsdp) is True
    assert args.fsdp_config["activation_checkpointing"] is False
    assert args.gradient_checkpointing is True
    assert args.fsdp_config["transformer_layer_cls_to_wrap"] == [
        "Qwen3_5DecoderLayer",
        "Qwen3_5VisionBlock",
    ]


def test_fsdp_auto_layers_require_model_default(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.model.model_type = "unknown_model"
    config.train.distributed.strategy = "fsdp"
    config.train.distributed.fsdp.transformer_layer_cls_to_wrap = ["auto"]

    try:
        _build_fsdp_args(config)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("FSDP auto layer resolution should require a registered default")

    assert "transformer_layer_cls_to_wrap=['auto']" in message


def test_deepspeed_training_arg_prefers_inline_config(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config_path = "configs/deepspeed/zero3_bf16.json"
    config.train.distributed.deepspeed.config = {"zero_optimization": {"stage": 3}}

    assert _build_deepspeed_arg(config) == {"zero_optimization": {"stage": 3}}


def test_deepspeed_training_arg_uses_config_path(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config_path = "configs/deepspeed/zero3_bf16.json"

    assert _build_deepspeed_arg(config) == "configs/deepspeed/zero3_bf16.json"


def test_build_hf_training_args_supports_deepspeed_strategy(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config = {
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "zero_optimization": {"stage": 2},
    }

    args = build_hf_training_args(config)

    assert args.deepspeed == config.train.distributed.deepspeed.config
    assert getattr(args, "hf_deepspeed_config", None) is not None
    assert _fsdp_enabled(args.fsdp) is False


def test_build_hf_training_args_resets_deepspeed_state_for_non_deepspeed(tmp_path: Path) -> None:
    from transformers.integrations.deepspeed import deepspeed_config

    deepspeed_config_payload = {
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "zero_optimization": {"stage": 2},
    }
    deepspeed_dir = tmp_path / "deepspeed"
    deepspeed_dir.mkdir()
    deepspeed_runtime = _write_config(deepspeed_dir)
    deepspeed_runtime.train.distributed.strategy = "deepspeed"
    deepspeed_runtime.train.distributed.deepspeed.config = deepspeed_config_payload
    _ = build_hf_training_args(deepspeed_runtime)
    assert deepspeed_config()["zero_optimization"]["stage"] == 2

    ddp_dir = tmp_path / "ddp"
    ddp_dir.mkdir()
    ddp_runtime = _write_config(ddp_dir)
    ddp_args = build_hf_training_args(ddp_runtime)

    assert ddp_args.deepspeed is None
    assert deepspeed_config() is None


def test_run_sft_builds_training_args_before_model_for_deepspeed(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config = {
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "zero_optimization": {"stage": 3},
    }
    call_order: list[str] = []
    training_args = object()

    def _fake_build_training_args(runtime_config):
        assert runtime_config is config
        call_order.append("training_args")
        return training_args

    def _fake_build_model(runtime_config, *, init_from_checkpoint=None):
        assert runtime_config is config
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

    with patch("shaft.pipeline.sft.build_hf_training_args", _fake_build_training_args):
        with patch("shaft.pipeline.sft.build_model_tokenizer_processor", _fake_build_model):
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                _ = run_sft(config)

    assert call_order == ["training_args", "model"]
    assert _FakeTrainer.last_kwargs["args"] is training_args


def test_training_topology_rejects_single_process_data_parallel(monkeypatch, tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.use_cpu = False
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    try:
        validate_training_topology(config)
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("validate_training_topology should reject single-process multi-GPU training")

    assert "torch.nn.DataParallel" in message
    assert "CUDA_VISIBLE_DEVICES=1" in message


def test_training_topology_allows_distributed_launch(monkeypatch, tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.use_cpu = False
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    validate_training_topology(config)


def test_run_sft_wires_loss_scale_into_train_collator(tmp_path: Path) -> None:
    config = _write_config(tmp_path, loss_scale="all")
    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
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
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            _ = run_sft(config)
    collator = _FakeTrainer.last_kwargs["data_collator"]
    assert collator.loss_scale_name == "all"


def test_hooks_are_wired_into_trainer_callbacks(tmp_path: Path) -> None:
    config = _write_config(tmp_path, hooks=["log_on_save"])
    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
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

    with patch("shaft.pipeline.sft.ShaftDataCenter", _FakeDataCenter):
        with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
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
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                _ = run_sft(config)

    assert captured["data_config"] is config.data
    assert captured["seed"] == config.experiment.seed
    assert captured["dataset_cls"] is SFTDataset
    assert _FakeTrainer.last_kwargs["train_dataset"] is fake_train_dataset
    assert _FakeTrainer.last_kwargs["train_sampler"] is fake_train_sampler
    assert _FakeTrainer.last_kwargs["eval_dataset"] is None
    assert _FakeTrainer.last_kwargs["model_adapter"] is mocked_builder.return_value.model_adapter
    assert _FakeTrainer.last_kwargs["finetune_plan"] is None


def test_run_sft_wires_data_center_train_sampler(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    fake_train_dataset = object()
    fake_eval_dataset = object()
    fake_train_sampler = object()

    class _FakeDataCenter:
        def __init__(self, data_config, *, seed):
            _ = data_config, seed

        def build_dataset_bundle(self, dataset_cls):
            _ = dataset_cls
            return ShaftDatasetBundle(
                train_dataset=fake_train_dataset,
                eval_dataset=fake_eval_dataset,
                train_sampler=fake_train_sampler,
            )

    with patch("shaft.pipeline.sft.ShaftDataCenter", _FakeDataCenter):
        with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
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
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                _ = run_sft(config)

    assert _FakeTrainer.last_kwargs["train_sampler"] is fake_train_sampler
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
  epochs: 1
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
  epochs: 1
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
  epochs: 1
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
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            _ = run_sft(config)
    assert not isinstance(_FakeTrainer.last_kwargs["eval_dataset"], dict)
    assert _FakeTrainer.last_kwargs["online_eval_runner"] is None
