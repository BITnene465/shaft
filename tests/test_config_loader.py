from __future__ import annotations

import math
from pathlib import Path

import pytest

from shaft.config import RuntimeConfig, TrainDDPConfig, TrainDistributedConfig, load_config
from shaft.data import load_prompt_source_pool
from tests.support.configs import load_config_from_yaml


pytestmark = pytest.mark.component


V5_7_DATASETS = (
    "grounding_layout",
    "shape_context_reconstruction",
    "line_context_reconstruction",
    "line_context_points",
    "image_context_reconstruction",
)
V5_7_CONFIGS = (
    "banana_sft_4b_v5_7.yaml",
    "banana_sft_4b_v5_7_trial.yaml",
    "banana_sft_27b_qwen36_v5_7_full_zero3.yaml",
    "banana_sft_27b_qwen36_v5_7_lora.yaml",
    "banana_sft_27b_qwen36_v5_7_qlora.yaml",
    "banana_sft_27b_qwen36_v5_7_re_full_zero3.yaml",
)

V5_8_FORMULATIONS = {
    "grounding_layout": ("labels", "boxes", "objects"),
    "shape_context_reconstruction": ("appearance", "geometry", "reconstruction"),
    "line_context_reconstruction": ("appearance", "points", "reconstruction"),
    "line_context_points": ("points",),
    "image_context_reconstruction": ("image_type",),
}
V5_8_FORMULATION_WEIGHTS = {
    "grounding_layout": (1.0, 1.0, 4.0),
    "shape_context_reconstruction": (1.0, 1.0, 4.0),
    "line_context_reconstruction": (1.0, 1.0, 4.0),
    "line_context_points": (1.0,),
    "image_context_reconstruction": (1.0,),
}


@pytest.mark.parametrize("filename", ("sft_4b.yaml", "dpo_4b.yaml", "grpo_4b.yaml"))
def test_checkpoint_enabled_example_configs_declare_media_snapshot(filename: str) -> None:
    config = load_config(Path("configs/train") / filename)

    assert config.train.save_strategy != "no"
    assert config.data.media_snapshot_id == "example-media-v1"


@pytest.mark.parametrize("filename", V5_7_CONFIGS)
def test_v5_7_training_configs_resolve_complete_dataset_and_prompt_contracts(
    filename: str,
) -> None:
    config = load_config(Path("configs/train") / filename)

    assert tuple(dataset.dataset_name for dataset in config.data.datasets) == V5_7_DATASETS
    assert tuple(config.data.catalog_names) == V5_7_DATASETS
    assert set(config.data.prompt_sources) == set(V5_7_DATASETS)
    assert all(not dataset.use_for_eval for dataset in config.data.datasets)
    assert config.eval.enabled is False
    assert config.data.media_snapshot_id == "banana-v5.7-media-v2"

    for dataset_name, source in config.data.prompt_sources.items():
        pool = load_prompt_source_pool(source.path)
        assert pool.formulations
        expected_version = (
            "v5.3" if dataset_name == "image_context_reconstruction" else "v5.7"
        )
        assert pool.version == expected_version


def test_v5_8_preparation_config_freezes_formulation_and_prompt_contracts() -> None:
    config = load_config(Path("configs/train/banana_sft_4b_v5_8_preparation.yaml"))

    assert tuple(dataset.dataset_name for dataset in config.data.datasets) == tuple(
        V5_8_FORMULATIONS
    )
    assert tuple(config.data.catalog_names) == tuple(V5_8_FORMULATIONS)
    assert all(not dataset.train_paths for dataset in config.data.datasets)
    assert all(not dataset.use_for_eval for dataset in config.data.datasets)
    assert config.eval.enabled is False
    assert config.data.media_snapshot_id == "banana-v5.8-preparation"
    dataset_weights = {
        dataset.dataset_name: dataset.weight for dataset in config.data.datasets
    }
    assert dataset_weights["line_context_reconstruction"] == pytest.approx(6.0)
    assert dataset_weights["line_context_points"] == pytest.approx(2.0)

    system_prompts = set()
    for dataset_name, expected_formulations in V5_8_FORMULATIONS.items():
        source = config.data.prompt_sources[dataset_name]
        pool = load_prompt_source_pool(source.path)
        formulation_ids = tuple(item.formulation_id for item in pool.formulations)

        assert pool.version == "v5.8"
        assert pool.explicit_formulations is True
        assert formulation_ids == expected_formulations
        assert tuple(source.formulation_sources) == expected_formulations
        assert tuple(
            formulation.sampling_weight for formulation in pool.formulations
        ) == V5_8_FORMULATION_WEIGHTS[dataset_name]
        for formulation in pool.formulations:
            assert tuple(prompt.variant_id for prompt in formulation.prompt_variants) == (
                "detailed",
                "concise",
            )
            system_prompts.update(
                prompt.system_prompt for prompt in formulation.prompt_variants
            )
    assert len(system_prompts) == 1
    assert next(iter(system_prompts)).startswith(
        "You are a specialized model for editable visual layout understanding"
    )


def test_load_minimal_config(tmp_path: Path) -> None:
    payload = """
experiment:
  name: demo
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    cfg = load_config_from_yaml(tmp_path, payload)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.experiment.name == "demo"
    assert len(cfg.data.datasets) == 1
    assert cfg.data.datasets[0].dataset_name == "ds1"
    assert cfg.model.finetune.mode == "full"
    assert cfg.model.attn_implementation is None
    assert cfg.model.experts_implementation is None
    assert isinstance(cfg.train.distributed, TrainDistributedConfig)
    assert isinstance(cfg.train.distributed.ddp, TrainDDPConfig)
    assert cfg.train.distributed.strategy == "ddp"
    assert cfg.train.distributed.ddp.static_graph is False
    assert cfg.progress.display == "auto"
    assert cfg.progress.width == 72
    assert cfg.progress.refresh_interval == pytest.approx(0.5)
    assert cfg.progress.log_interval == pytest.approx(30.0)
    assert cfg.progress.leave_completed is False
    assert cfg.progress.persist is True
    assert cfg.data.schedule.mixing == "weighted"
    assert cfg.data.schedule.shuffle is True
    assert cfg.data.prompt_sources == {}
    assert cfg.data.batching.grouping == "none"
    assert cfg.data.batching.cardinality == "fixed"
    assert cfg.data.batching.packing.mode == "none"
    assert cfg.data.batching.layout == "padded"


def test_schedule_and_data_flags_parse_quoted_booleans(tmp_path: Path) -> None:
    payload = """
data:
  schedule:
    shuffle: "false"
  pin_memory: "false"
  persistent_workers: "false"
  add_eos_token: "false"
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
      use_for_eval: "false"
eval:
  enabled: "false"
"""
    cfg = load_config_from_yaml(tmp_path, payload)

    assert cfg.data.schedule.shuffle is False
    assert cfg.data.pin_memory is False
    assert cfg.data.persistent_workers is False
    assert cfg.data.add_eos_token is False
    assert cfg.data.datasets[0].use_for_eval is False


def test_normalization(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: SFT
data:
  schedule:
    mixing: WEIGHTED
  record_cache_dir: .cache/records
  max_length: 4096
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
      help: "  demo dataset  "
      tags: [" a ", "", "b"]
train:
  scheduler_name: auto
  lr_scheduler_type: LINEAR
  loss_scale: ALL
  gradient_checkpointing: true
  distributed:
    strategy: FSDP
    fsdp:
      sharding_strategy: FULL_SHARD
      auto_wrap_policy: TRANSFORMER
      transformer_layer_cls_to_wrap: [" auto ", "auto"]
      activation_checkpointing: true
      state_dict_type: FULL_STATE_DICT
      backward_prefetch: BACKWARD_PRE
  save_epoch_interval: 2
  param_group_lrs:
    Language_Model: 1.0e-5
    Vision_Tower: 2.5e-5
  no_decay_name_patterns: [" Embed_Tokens.Weight ", "", "LM_HEAD.WEIGHT", "embed_tokens.weight"]
eval:
  epoch_interval: 3
model:
  experts_implementation: GROUPED_MM
  finetune:
    mode: DORA
progress:
  display: INTERACTIVE
  width: 80
  refresh_interval: 0.75
  log_interval: 45
  leave_completed: true
  persist: false
"""
    cfg = load_config_from_yaml(tmp_path, payload)
    assert cfg.algorithm.name == "sft"
    assert cfg.data.schedule.mixing == "weighted"
    assert cfg.data.record_cache_dir == str((tmp_path / ".cache/records").resolve())
    assert cfg.data.max_length == 4096
    assert cfg.train.scheduler_name == "linear"
    assert cfg.train.loss_scale == "all"
    assert cfg.train.gradient_checkpointing is True
    assert cfg.train.distributed.strategy == "fsdp"
    assert cfg.train.distributed.fsdp.sharding_strategy == "full_shard"
    assert cfg.train.distributed.fsdp.auto_wrap_policy == "transformer"
    assert cfg.train.distributed.fsdp.transformer_layer_cls_to_wrap == ["auto"]
    assert cfg.train.distributed.fsdp.state_dict_type == "full_state_dict"
    assert cfg.train.distributed.fsdp.backward_prefetch == "backward_pre"
    assert cfg.train.save_epoch_interval == 2
    assert cfg.eval.epoch_interval == 3
    assert cfg.train.param_group_lrs == {
        "language_model": pytest.approx(1.0e-5),
        "vision_tower": pytest.approx(2.5e-5),
    }
    assert cfg.train.no_decay_name_patterns == ["embed_tokens.weight", "lm_head.weight"]
    assert cfg.model.finetune.mode == "dora"
    assert cfg.model.experts_implementation == "grouped_mm"
    assert cfg.progress.display == "interactive"
    assert cfg.progress.width == 80
    assert cfg.progress.refresh_interval == pytest.approx(0.75)
    assert cfg.progress.log_interval == pytest.approx(45.0)
    assert cfg.progress.leave_completed is True
    assert cfg.progress.persist is False
    assert cfg.data.datasets[0].help == "demo dataset"
    assert cfg.data.datasets[0].tags == ["a", "b"]


def test_experts_implementation_is_rejected_outside_qwen_vl_profiles(
    tmp_path: Path,
) -> None:
    payload = """
model:
  model_type: smoke_vlm
  experts_implementation: grouped_mm
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
"""

    with pytest.raises(ValueError, match="only by Qwen VL MoE profiles"):
        load_config_from_yaml(tmp_path, payload)


def test_sft_auxiliary_loss_weights_are_normalized(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: SFT
  params:
    auxiliary_loss_weights:
      Router_Aux_Loss: 0.002
      Disabled_Loss: -0.0
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    cfg = load_config_from_yaml(tmp_path, payload)

    assert cfg.algorithm.params == {
        "auxiliary_loss_weights": {
            "disabled_loss": 0.0,
            "router_aux_loss": pytest.approx(0.002),
        }
    }
    assert math.copysign(
        1.0,
        cfg.algorithm.params["auxiliary_loss_weights"]["disabled_loss"],
    ) == 1.0

    empty_cfg = load_config_from_yaml(
        tmp_path,
        """
algorithm:
  name: sft
  params:
    auxiliary_loss_weights: {}
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
""",
        filename="empty-auxiliary-weights.yaml",
    )
    assert empty_cfg.algorithm.params == {}


@pytest.mark.parametrize(
    ("algorithm_name", "weights", "message"),
    [
        ("sft", "[]", "must be a mapping"),
        ("sft", "{router_aux_loss: -0.1}", "must be finite and >= 0"),
        ("sft", "{router_aux_loss: .nan}", "must be finite and >= 0"),
        ("sft", "{router_aux_loss: true}", "must be a number"),
        ("sft", "{router_aux_loss: '0.1'}", "must be a number"),
        ("sft", "{' ': 0.1}", "empty term name"),
        (
            "sft",
            "{Router_Aux_Loss: 0.1, router_aux_loss: 0.2}",
            "duplicate normalized term name",
        ),
        ("dpo", "{router_aux_loss: 0.1}", "not consumed.*auxiliary_loss_weights"),
    ],
)
def test_invalid_auxiliary_loss_weights_are_rejected(
    tmp_path: Path,
    algorithm_name: str,
    weights: str,
    message: str,
) -> None:
    source_type = "jsonl_sft" if algorithm_name == "sft" else "jsonl_dpo"
    payload = f"""
algorithm:
  name: {algorithm_name}
  params:
    auxiliary_loss_weights: {weights}
data:
  datasets:
    - dataset_name: ds1
      source_type: {source_type}
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises((TypeError, ValueError), match=message):
        load_config_from_yaml(tmp_path, payload)


def test_unknown_builtin_sft_algorithm_param_is_rejected(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: sft
  params:
    auxiliary_loss_weight: 0.1
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match="Unknown algorithm.params keys.*auxiliary_loss_weight"):
        load_config_from_yaml(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("display", "sometimes", "progress.display"),
        ("width", "39", "progress.width"),
        ("refresh_interval", "0", "progress.refresh_interval"),
        ("refresh_interval", ".nan", "progress.refresh_interval"),
        ("log_interval", "-1", "progress.log_interval"),
        ("log_interval", ".inf", "progress.log_interval"),
    ],
)
def test_invalid_progress_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    payload = f"""
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
progress:
  {field}: {value}
"""

    with pytest.raises(ValueError, match=message):
        load_config_from_yaml(tmp_path, payload)
