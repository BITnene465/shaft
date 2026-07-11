from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from shaft.config import load_config
from shaft.data import (
    SFTDataset,
    ShaftCostAwareSampler,
    ShaftDatasetBundle,
    ShaftDynamicBatchSampler,
    ShaftMMapCostPlanProvider,
    ShaftRowInvariantCostProvider,
    ShaftSampleCost,
    cost_plan_reference_path,
)
from shaft.model.finetune_plan import resolved_finetune_summary_path
from shaft.pipeline import run_sft
from shaft.training.batch_planning import (
    ShaftBatchPlanningCallback,
    batch_planning_signature_path,
    load_batch_planning_signature,
)
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


def test_run_sft_initializes_seed_before_model_and_adapter_build(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.experiment.seed = 137
    torch.manual_seed(config.experiment.seed + 1)

    def build_seeded_artifacts(*args, **kwargs):
        _ = args, kwargs
        assert torch.initial_seed() == config.experiment.seed
        return _build_fake_model_artifacts()

    with patch(
        "shaft.pipeline.sft.build_model_tokenizer_processor",
        side_effect=build_seeded_artifacts,
    ):
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            _ = run_sft(config)


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


def test_run_sft_replaces_sample_sampler_with_cost_aware_plan(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.data.batching.strategy = "cost_aware"
    config.data.batching.planning_window = 7
    config.data.batching.image_size_cache_size = 3
    config.data.batching.cost_plan_cache_dir = str(tmp_path / "cost-plans")

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch("shaft.pipeline.sft.ShaftSFTSampleCostProvider") as mocked_provider:
            mocked_provider.return_value = ShaftRowInvariantCostProvider(
                {
                    ("ds", 0): ShaftSampleCost(
                        llm_tokens=4,
                        supervised_tokens=2,
                        vision_patches=16,
                        exact=True,
                    )
                },
                fingerprint="pipeline-test-cost-v1",
            )
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                _ = run_sft(config)

    train_sampler = _FakeTrainer.last_kwargs["train_sampler"]
    assert isinstance(train_sampler, ShaftCostAwareSampler)
    assert train_sampler.planner.planning_window == 7
    assert train_sampler.planner.data_world_size == 1
    assert train_sampler.planner.per_device_batch_size == 1
    assert isinstance(train_sampler.planner.cost_provider, ShaftMMapCostPlanProvider)
    assert train_sampler.planner.cost_provider.closed is True
    provider_kwargs = mocked_provider.call_args.kwargs
    assert provider_kwargs["dataset"] is _FakeTrainer.last_kwargs["train_dataset"]
    assert provider_kwargs["model_adapter"] is mocked_builder.return_value.model_adapter
    assert provider_kwargs["image_size_cache_size"] == 3
    assert cost_plan_reference_path(config.experiment.output_dir).is_file()
    assert load_batch_planning_signature(config.experiment.output_dir) == train_sampler.signature
    assert any(
        isinstance(callback, ShaftBatchPlanningCallback)
        for callback in _FakeTrainer.last_kwargs["callbacks"]
    )


def test_run_sft_wires_dynamic_cost_aware_batch_sampler(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.data.batching.strategy = "dynamic_cost_aware"
    config.data.batching.planning_window = 4
    config.data.batching.max_samples_per_microbatch = 2
    config.data.batching.max_padded_tokens = 8
    config.data.batching.cost_plan_cache_dir = str(tmp_path / "cost-plans")

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch(
            "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
            return_value=ShaftRowInvariantCostProvider(
                {
                    ("ds", 0): ShaftSampleCost(
                        llm_tokens=4,
                        supervised_tokens=2,
                        vision_patches=16,
                        exact=True,
                    )
                },
                fingerprint="pipeline-dynamic-cost-v1",
            ),
        ):
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                _ = run_sft(config)

    assert _FakeTrainer.last_kwargs["train_sampler"] is None
    batch_sampler = _FakeTrainer.last_kwargs["train_batch_sampler"]
    assert isinstance(batch_sampler, ShaftDynamicBatchSampler)
    assert batch_sampler.planner.spec.target_samples == 1
    assert batch_sampler.planner.spec.optimizer_step_count == 1
    assert batch_sampler.signature.strategy == "dynamic_cost_aware"
    assert isinstance(batch_sampler.planner.cost_provider, ShaftMMapCostPlanProvider)
    assert batch_sampler.planner.cost_provider.closed is True
    assert load_batch_planning_signature(config.experiment.output_dir) == (
        batch_sampler.signature
    )


def test_run_sft_resolves_dynamic_token_target_sample_count(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.data.batching.strategy = "dynamic_cost_aware"
    config.data.batching.planning_window = 3
    config.data.batching.max_samples_per_microbatch = 3
    config.data.batching.max_padded_tokens = 12
    config.data.batching.cost_plan_cache_dir = str(tmp_path / "cost-plans")
    config.train.optimizer_batch.target_supervised_tokens = 3

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch(
            "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
            return_value=ShaftRowInvariantCostProvider(
                {
                    ("ds", 0): ShaftSampleCost(
                        llm_tokens=4,
                        supervised_tokens=2,
                        vision_patches=16,
                        exact=True,
                    )
                },
                fingerprint="pipeline-dynamic-token-target-v1",
            ),
        ):
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                _ = run_sft(config)

    batch_sampler = _FakeTrainer.last_kwargs["train_batch_sampler"]
    assert isinstance(batch_sampler, ShaftDynamicBatchSampler)
    assert batch_sampler.planned_sample_count == 2
    assert batch_sampler.planned_optimizer_batch_samples is None


def test_dynamic_plan_preflight_rejects_oversize_before_trainer_and_publish(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.data.batching.strategy = "dynamic_cost_aware"
    config.data.batching.planning_window = 1
    config.data.batching.max_samples_per_microbatch = 1
    config.data.batching.max_padded_tokens = 3
    config.data.batching.cost_plan_cache_dir = str(tmp_path / "cost-plans")

    class _UnexpectedTrainer(_FakeTrainer):
        def __init__(self, **kwargs):
            _ = kwargs
            raise AssertionError("trainer must not be built")

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch(
            "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
            return_value=ShaftRowInvariantCostProvider(
                {
                    ("ds", 0): ShaftSampleCost(
                        llm_tokens=4,
                        supervised_tokens=2,
                        vision_patches=16,
                        exact=True,
                    )
                },
                fingerprint="pipeline-dynamic-oversize-v1",
            ),
        ):
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _UnexpectedTrainer):
                with pytest.raises(ValueError, match="oversize sample"):
                    run_sft(config)

    assert not cost_plan_reference_path(config.experiment.output_dir).exists()
    assert not batch_planning_signature_path(config.experiment.output_dir).exists()


def test_run_sft_closes_mmap_provider_when_trainer_raises(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.data.batching.strategy = "cost_aware"
    config.data.batching.cost_plan_cache_dir = str(tmp_path / "cost-plans")

    class _RaisingTrainer(_FakeTrainer):
        def train(self, resume_from_checkpoint=None):
            _ = resume_from_checkpoint
            raise RuntimeError("injected trainer failure")

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch(
            "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
            return_value=ShaftRowInvariantCostProvider(
                {
                    ("ds", 0): ShaftSampleCost(
                        llm_tokens=4,
                        supervised_tokens=2,
                        vision_patches=16,
                        exact=True,
                    )
                },
                fingerprint="trainer-failure-cost-v1",
            ),
        ):
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _RaisingTrainer):
                with pytest.raises(RuntimeError, match="injected trainer failure"):
                    run_sft(config)

    train_sampler = _RaisingTrainer.last_kwargs["train_sampler"]
    assert isinstance(train_sampler.planner.cost_provider, ShaftMMapCostPlanProvider)
    assert train_sampler.planner.cost_provider.closed is True


def test_run_sft_nonzero_rank_maps_reference_without_runtime_cost_rebuild(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.data.batching.strategy = "cost_aware"
    config.data.batching.cost_plan_cache_dir = str(tmp_path / "cost-plans")
    static_provider = ShaftRowInvariantCostProvider(
        {
            ("ds", 0): ShaftSampleCost(
                llm_tokens=4,
                supervised_tokens=2,
                vision_patches=16,
                exact=True,
            )
        },
        fingerprint="pipeline-shared-cost-v1",
    )

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch(
            "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
            return_value=static_provider,
        ):
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                run_sft(config)

    reference = json.loads(
        cost_plan_reference_path(config.experiment.output_dir).read_text(encoding="utf-8")
    )
    rendezvous = {
        "ok": True,
        "manifest_path": reference["manifest_path"],
        "manifest_fingerprint": reference["manifest_fingerprint"],
    }

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch("shaft.pipeline.sft.ShaftSFTSampleCostProvider") as mocked_provider:
            with patch("shaft.pipeline.sft.is_rank_zero", return_value=False):
                with patch(
                    "shaft.pipeline.sft.broadcast_object_from_rank_zero",
                    return_value=rendezvous,
                ):
                    with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                        run_sft(config)

    mocked_provider.assert_not_called()
    train_sampler = _FakeTrainer.last_kwargs["train_sampler"]
    assert isinstance(train_sampler.planner.cost_provider, ShaftMMapCostPlanProvider)


def test_failed_resume_does_not_replace_durable_cost_plan_reference(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.data.batching.strategy = "cost_aware"
    config.data.batching.cost_plan_cache_dir = str(tmp_path / "cost-plans")

    def row_provider(fingerprint: str) -> ShaftRowInvariantCostProvider:
        return ShaftRowInvariantCostProvider(
            {
                ("ds", 0): ShaftSampleCost(
                    llm_tokens=4,
                    supervised_tokens=2,
                    vision_patches=16,
                    exact=True,
                )
            },
            fingerprint=fingerprint,
        )

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch(
            "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
            return_value=row_provider("resume-reference-v1"),
        ):
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                run_sft(config)

    reference_path = cost_plan_reference_path(config.experiment.output_dir)
    original_reference = reference_path.read_bytes()
    config.train.resume_from_checkpoint = str(tmp_path / "checkpoint-1")

    with patch(
        "shaft.pipeline.sft.resolve_resume_checkpoint",
        return_value=str(tmp_path / "checkpoint-1"),
    ):
        with patch("shaft.pipeline.sft.validate_resume_checkpoint"):
            with patch("shaft.pipeline.sft.validate_batch_planning_resume_geometry"):
                with patch(
                    "shaft.pipeline.sft.build_model_tokenizer_processor"
                ) as mocked_builder:
                    mocked_builder.return_value = _build_fake_model_artifacts()
                    with patch(
                        "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
                        return_value=row_provider("resume-reference-v2"),
                    ):
                        with patch(
                            "shaft.pipeline.sft.validate_batch_planning_resume",
                            side_effect=ValueError("full planning signature changed"),
                        ):
                            with pytest.raises(
                                ValueError,
                                match="full planning signature changed",
                            ):
                                run_sft(config)

    assert reference_path.read_bytes() == original_reference


def test_failed_metadata_publish_rolls_back_reference_and_signature(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.data.batching.strategy = "cost_aware"
    config.data.batching.cost_plan_cache_dir = str(tmp_path / "cost-plans")

    def row_provider(fingerprint: str) -> ShaftRowInvariantCostProvider:
        return ShaftRowInvariantCostProvider(
            {
                ("ds", 0): ShaftSampleCost(
                    llm_tokens=4,
                    supervised_tokens=2,
                    vision_patches=16,
                    exact=True,
                )
            },
            fingerprint=fingerprint,
        )

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch(
            "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
            return_value=row_provider("publish-v1"),
        ):
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                run_sft(config)

    reference_path = cost_plan_reference_path(config.experiment.output_dir)
    signature_path = batch_planning_signature_path(config.experiment.output_dir)
    original_reference = reference_path.read_bytes()
    original_signature = signature_path.read_bytes()

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        mocked_builder.return_value = _build_fake_model_artifacts()
        with patch(
            "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
            return_value=row_provider("publish-v2"),
        ):
            with patch(
                "shaft.pipeline.sft.write_cost_plan_reference",
                side_effect=OSError("reference publish failed"),
            ):
                with pytest.raises(OSError, match="reference publish failed"):
                    run_sft(config)

    assert reference_path.read_bytes() == original_reference
    assert signature_path.read_bytes() == original_signature


def test_cost_aware_geometry_fails_before_model_loading(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.data.batching.strategy = "cost_aware"
    config.data.batching.planning_window = 0

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as mocked_builder:
        with pytest.raises(ValueError, match="planning_window must be > 0"):
            run_sft(config)

    mocked_builder.assert_not_called()


def test_cost_plan_local_preflight_failure_enters_collective_before_raising(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.data.batching.strategy = "cost_aware"
    gathered: list[dict] = []

    def gather_status(status):
        gathered.append(status)
        return [status]

    with patch(
        "shaft.pipeline.sft.sft_cost_planning_source_fingerprint",
        side_effect=OSError("source snapshot unreadable"),
    ):
        with patch(
            "shaft.pipeline.sft.all_gather_objects",
            side_effect=gather_status,
        ):
            with patch(
                "shaft.pipeline.sft.build_model_tokenizer_processor"
            ) as mocked_builder:
                with pytest.raises(OSError, match="source snapshot unreadable"):
                    run_sft(config)

    assert gathered == [
        {"ok": False, "error": "OSError: source snapshot unreadable"}
    ]
    mocked_builder.assert_not_called()


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
