from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from shaft.data import ShaftBoundedBatchSampler, ShaftSampleCost
from shaft.model.finetune_plan import resolved_finetune_summary_path
from shaft.observability import PROGRESS_SNAPSHOT_FILENAME
from shaft.pipeline import run_sft
from shaft.plugins.hooks import hook
from shaft.training.batch_planning import (
    BOUNDED_BATCHING_CALLBACK_NAME,
    ShaftBoundedBatchingCallback,
    batching_run_metadata_path,
    load_batching_run_metadata,
)
from tests.support.pipeline import FakePipelineModel as _FakeModel
from tests.support.pipeline import FakePipelineTrainer as _FakeTrainer
from tests.support.pipeline import build_fake_model_artifacts as _build_fake_model_artifacts
from tests.support.pipeline import write_sft_pipeline_config as _write_config


pytestmark = pytest.mark.component


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
    config.data.batching.strategy = "bounded_cost_aware"
    config.data.batching.buffer_size = 8
    config.data.batching.cost_cache_size = 16
    config.data.batching.max_samples_per_microbatch = 4
    config.data.batching.max_padded_tokens = 16
    config.data.batching.max_vision_patches = 32
    config.data.media_snapshot_id = "pipeline-fixture-v1"
    config.data.mix_strategy = "concat"
    config.data.shuffle = False
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
    assert isinstance(batch_sampler, ShaftBoundedBatchSampler)
    assert provider.calls == []
    first_global_batches = []
    iterator = iter(batch_sampler)
    for _ in range(batch_sampler.spec.data_world_size):
        first_global_batches.append(next(iterator))
    assert all(first_global_batches)
    assert provider.calls == list(range(config.data.batching.buffer_size))
    assert not list(Path(config.experiment.output_dir).glob("*cost_plan*"))
    metadata = load_batching_run_metadata(config.experiment.output_dir)
    assert metadata.strategy == "bounded_cost_aware"
    assert metadata.media_snapshot_id == "pipeline-fixture-v1"
    assert metadata.contract_fingerprint == batch_sampler.spec.fingerprint
    assert any(
        isinstance(callback, ShaftBoundedBatchingCallback)
        for callback in _FakeTrainer.last_kwargs["callbacks"]
    )


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
        if isinstance(callback, ShaftBoundedBatchingCallback)
    )
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "stateful_callbacks": {
                    BOUNDED_BATCHING_CALLBACK_NAME: initial_callback.state()
                },
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"model")
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
