from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from shaft.data import ShaftPlannedBatchSampler, ShaftSampleCost
from shaft.model import SequenceExecutionPolicy
from shaft.model.finetune_plan import resolved_finetune_summary_path
from shaft.observability import PROGRESS_SNAPSHOT_FILENAME
from shaft.pipeline import run_sft
from shaft.plugins.hooks import hook
from shaft.training.batch_planning import (
    BATCHING_METADATA_CALLBACK_NAME,
    BATCH_PLANNING_CALLBACK_NAME,
    ShaftBatchingMetadataCallback,
    ShaftBatchPlanningCallback,
    batching_run_metadata_path,
    load_batching_run_metadata,
    write_batch_planning_checkpoint_completion,
)
from tests.support.pipeline import FakePipelineModel as _FakeModel
from tests.support.pipeline import FakePipelineTrainer as _FakeTrainer
from tests.support.pipeline import build_fake_model_artifacts as _build_fake_model_artifacts
from tests.support.pipeline import write_sft_pipeline_config as _write_base_config


pytestmark = pytest.mark.component


def _write_config(tmp_path: Path, **kwargs):
    config = _write_base_config(tmp_path, **kwargs)
    config.model.model_type = "smoke_vlm"
    config.model.model_name_or_path = "models/Smoke-VLM"
    return config


class _CountingProvider:
    fingerprint = "bounded-pipeline-cost-v1"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, sample_ref):
        self.calls.append(int(sample_ref.context.draw_id))
        return ShaftSampleCost(
            llm_tokens=2,
            supervised_tokens=1,
            vision_patches=4,
            exact=True,
        )


def _enable_bounded(config, *, steps: int = 2) -> None:
    config.data.batching.grouping = "bounded_cost"
    config.data.batching.cardinality = "fixed"
    config.data.batching.layout = "padded"
    config.data.batching.buffer_size = 8
    config.data.batching.cost_cache_size = 16
    config.data.batching.max_tokens_per_microbatch = 16
    config.data.batching.resource_budgets = {"vision_patches": 32}
    config.data.media_snapshot_id = "pipeline-fixture-v1"
    config.data.schedule.mixing = "concat"
    config.data.schedule.shuffle = False
    config.train.duration.value = steps


def _enable_length_grouping(config, *, steps: int = 2) -> None:
    config.data.max_length = 8
    config.data.batching.grouping = "length"
    config.data.batching.cardinality = "fixed"
    config.data.batching.packing.mode = "none"
    config.data.batching.layout = "padded"
    config.data.batching.buffer_size = 8
    config.data.batching.cost_cache_size = 16
    config.data.batching.max_tokens_per_microbatch = None
    config.data.batching.resource_budgets = {"vision_patches": 32}
    config.data.media_snapshot_id = "pipeline-fixture-v1"
    config.data.schedule.mixing = "concat"
    config.data.schedule.shuffle = False
    config.train.duration.value = steps


def test_run_sft_smoke(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    fake_model = _FakeModel()
    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as builder:
        builder.return_value = _build_fake_model_artifacts(
            model=fake_model,
            include_finetune_plan=True,
        )
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            metrics = run_sft(config)

    assert metrics["train_loss"] == pytest.approx(0.1)
    assert fake_model.generation_config.do_sample is False
    assert fake_model.generation_config.temperature == 1.0
    assert fake_model.generation_config.eos_token_id == [2, 99]
    assert resolved_finetune_summary_path(config.experiment.output_dir).exists()
    progress = json.loads(
        (Path(config.experiment.output_dir) / PROGRESS_SNAPSHOT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert progress["tasks"]["startup.data"]["state"] == "succeeded"
    assert progress["tasks"]["startup.model"]["state"] == "succeeded"


def test_run_sft_initializes_seed_before_model_build(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.experiment.seed = 137
    torch.manual_seed(999)

    def build_seeded_artifacts(*args, **kwargs):
        _ = args, kwargs
        assert torch.initial_seed() == 137
        return _build_fake_model_artifacts()

    with patch(
        "shaft.pipeline.sft.build_model_tokenizer_processor",
        side_effect=build_seeded_artifacts,
    ):
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            run_sft(config)


def test_run_sft_rejects_resume_contract_before_data_or_model_load(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.train.resume_from_checkpoint = str(tmp_path / "checkpoint-1")

    with patch(
        "shaft.pipeline.sft.resolve_resume_checkpoint",
        return_value=config.train.resume_from_checkpoint,
    ):
        with patch("shaft.pipeline.sft.validate_resume_checkpoint"):
            with patch(
                "shaft.pipeline.sft.validate_batching_resume_contract",
                side_effect=ValueError("batch contract drift"),
            ):
                with patch("shaft.pipeline.sft.ShaftDataCenter") as data_center:
                    with patch(
                        "shaft.pipeline.sft.build_model_tokenizer_processor"
                    ) as build_model:
                        with pytest.raises(ValueError, match="batch contract drift"):
                            run_sft(config)

    data_center.assert_not_called()
    build_model.assert_not_called()


def test_run_sft_wires_loss_scale_and_hooks(tmp_path: Path) -> None:
    hook_name = f"pipeline-hook-{tmp_path.name}"

    @hook("after_step", name=hook_name)
    def _test_hook(state):
        _ = state

    config = _write_config(tmp_path, loss_scale="all", hooks=[hook_name])
    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as builder:
        builder.return_value = _build_fake_model_artifacts()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            run_sft(config)

    kwargs = _FakeTrainer.last_kwargs
    assert kwargs["data_collator"].loss_scale_name == "all"
    assert any(type(callback).__name__ == "TrainerHookCallback" for callback in kwargs["callbacks"])


def test_bounded_pipeline_has_no_full_plan_preflight_or_cost_plan_sidecar(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    _enable_bounded(config, steps=3)
    provider = _CountingProvider()

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as builder:
        builder.return_value = _build_fake_model_artifacts()
        with patch(
            "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
            return_value=provider,
        ):
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                run_sft(config)

    batch_sampler = _FakeTrainer.last_kwargs["train_batch_sampler"]
    assert isinstance(batch_sampler, ShaftPlannedBatchSampler)
    assert provider.calls == []
    first_global_batches = []
    iterator = iter(batch_sampler)
    for _ in range(batch_sampler.spec.data_world_size):
        first_global_batches.append(next(iterator))
    assert all(first_global_batches)
    assert provider.calls == list(range(config.data.batching.buffer_size))
    assert not list(Path(config.experiment.output_dir).glob("*cost_plan*"))
    metadata = load_batching_run_metadata(config.experiment.output_dir)
    assert metadata.grouping == "bounded_cost"
    assert metadata.cardinality == "fixed"
    assert metadata.packing == "none"
    assert metadata.layout == "padded"
    assert metadata.per_device_train_batch_size == 1
    assert metadata.media_snapshot_id == "pipeline-fixture-v1"
    assert metadata.batch_contract_fingerprint
    assert metadata.planner_spec_fingerprint == batch_sampler.spec.fingerprint
    assert any(
        isinstance(callback, ShaftBatchPlanningCallback)
        for callback in _FakeTrainer.last_kwargs["callbacks"]
    )


def test_length_pipeline_uses_the_unified_lazy_planner(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    _enable_length_grouping(config, steps=3)
    provider = _CountingProvider()

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as builder:
        builder.return_value = _build_fake_model_artifacts()
        with patch(
            "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
            return_value=provider,
        ):
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                run_sft(config)

    batch_sampler = _FakeTrainer.last_kwargs["train_batch_sampler"]
    assert isinstance(batch_sampler, ShaftPlannedBatchSampler)
    assert batch_sampler.spec.grouping == "length"
    assert batch_sampler.spec.packing == "none"
    assert batch_sampler.spec.layout == "padded"
    assert batch_sampler.spec.max_sequence_length == 8
    assert batch_sampler.spec.max_tokens_per_microbatch == 8
    assert provider.calls == []
    first_local_batch = next(iter(batch_sampler))
    assert first_local_batch
    assert provider.calls == list(range(config.data.batching.buffer_size))

    metadata = load_batching_run_metadata(config.experiment.output_dir)
    assert metadata.grouping == "length"
    assert metadata.max_sequence_length == 8
    assert metadata.max_tokens_per_microbatch is None
    assert metadata.planner_spec_fingerprint == batch_sampler.spec.fingerprint


def test_varlen_pipeline_preflights_model_policy_and_configures_train_collator(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    _enable_length_grouping(config, steps=1)
    config.data.batching.packing.mode = "greedy"
    config.data.batching.layout = "varlen"
    config.eval.enabled = True
    provider = _CountingProvider()

    class _RecordingSequencePolicy(SequenceExecutionPolicy):
        def __init__(self) -> None:
            self.calls = []

        def validate_runtime(self, *, model, contract) -> None:
            self.calls.append((model, contract))

        def build_contract(self, **kwargs):
            from shaft.model import ShaftSequenceExecutionContract

            return ShaftSequenceExecutionContract(
                **kwargs,
                capability_signature=("recording-sequence-policy-v1",),
            )

    policy = _RecordingSequencePolicy()
    artifacts = _build_fake_model_artifacts()
    artifacts.model_adapter = replace(
        artifacts.model_adapter,
        sequence_execution_policy=policy,
    )

    with patch(
        "shaft.pipeline.sft.resolve_model_adapter_from_config",
        return_value=artifacts.model_adapter,
    ):
        with patch(
            "shaft.pipeline.sft.build_model_tokenizer_processor",
            return_value=artifacts,
        ) as build_model:
            with patch(
                "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
                return_value=provider,
            ):
                with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                    run_sft(config)

    assert len(policy.calls) == 1
    validated_model, validated_contract = policy.calls[0]
    assert validated_model is artifacts.model
    assert validated_contract.layout == "varlen"
    assert validated_contract.device_type == "cpu"
    loaded_contract = build_model.call_args.kwargs["sequence_execution_contract"]
    assert loaded_contract.layout == "varlen"
    assert loaded_contract.capability_signature == ("recording-sequence-policy-v1",)
    collator = _FakeTrainer.last_kwargs["data_collator"]
    assert collator.layout == "varlen"
    assert collator.packing_mode == "greedy"
    eval_collator = _FakeTrainer.last_kwargs["eval_data_collator"]
    assert eval_collator.layout == "padded"
    assert eval_collator.packing_mode == "none"


def test_varlen_pipeline_rejects_unsupported_execution_before_data_or_model_load(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    _enable_length_grouping(config, steps=1)
    config.data.batching.packing.mode = "greedy"
    config.data.batching.layout = "varlen"
    unsupported_adapter = replace(
        _build_fake_model_artifacts().model_adapter,
        sequence_execution_policy=SequenceExecutionPolicy(),
    )

    with patch(
        "shaft.pipeline.sft.resolve_model_adapter_from_config",
        return_value=unsupported_adapter,
    ):
        with patch("shaft.pipeline.sft.ShaftDataCenter") as data_center:
            with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as build_model:
                with pytest.raises(ValueError, match="does not support varlen layout"):
                    run_sft(config)

    data_center.assert_not_called()
    build_model.assert_not_called()


def test_bounded_pipeline_resume_loads_committed_state_and_disables_hf_skip(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    _enable_bounded(config, steps=3)

    def run_with_provider():
        provider = _CountingProvider()
        with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as builder:
            builder.return_value = _build_fake_model_artifacts()
            with patch(
                "shaft.pipeline.sft.ShaftSFTSampleCostProvider",
                return_value=provider,
            ):
                with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                    run_sft(config)
        return _FakeTrainer.last_kwargs["train_batch_sampler"]

    initial_sampler = run_with_provider()
    _ = list(initial_sampler)
    committed = initial_sampler.commit_global_microstep(1)
    initial_callback = next(
        callback
        for callback in _FakeTrainer.last_kwargs["callbacks"]
        if isinstance(callback, ShaftBatchPlanningCallback)
    )
    metadata_callback = next(
        callback
        for callback in _FakeTrainer.last_kwargs["callbacks"]
        if isinstance(callback, ShaftBatchingMetadataCallback)
    )
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "stateful_callbacks": {
                    BATCH_PLANNING_CALLBACK_NAME: initial_callback.state(),
                    BATCHING_METADATA_CALLBACK_NAME: metadata_callback.state(),
                },
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    if initial_sampler.spec.data_world_size == 1:
        (checkpoint / "rng_state.pth").write_bytes(b"rng")
    else:
        for rank in range(initial_sampler.spec.data_world_size):
            (checkpoint / f"rng_state_{rank}.pth").write_bytes(b"rng")
    write_batch_planning_checkpoint_completion(checkpoint)
    config.train.resume_from_checkpoint = str(checkpoint)

    resumed_sampler = run_with_provider()

    assert resumed_sampler.initial_state == committed
    assert _FakeTrainer.last_kwargs["args"].ignore_data_skip is True


def test_fixed_pipeline_keeps_plain_sample_sampler(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as builder:
        builder.return_value = _build_fake_model_artifacts()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            run_sft(config)

    assert _FakeTrainer.last_kwargs["train_sampler"] is not None
    assert _FakeTrainer.last_kwargs["train_batch_sampler"] is None
    assert batching_run_metadata_path(config.experiment.output_dir).is_file()
