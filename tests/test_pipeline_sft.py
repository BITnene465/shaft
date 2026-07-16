from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from shaft.config import (
    EvalDatasetPolicyConfig,
    EvalMetricConfig,
    resolve_eval_input_policy,
)
from shaft.data import (
    ShaftDatasetBundle,
    ShaftPlannedBatchSampler,
    ShaftSampleCost,
    ShaftSampleSampler,
)
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
    build_batch_contract,
)
from shaft.training.checkpointing import (
    ResolvedResumeCheckpoint,
    ShaftCheckpointProtocol,
    commit_training_checkpoint,
    resolve_resume_checkpoint_generation,
)
from shaft.training.resume_contract import build_training_resume_contract
from shaft.pipeline.training_args import build_hf_training_args
from tests.support.pipeline import FakePipelineModel as _FakeModel
from tests.support.pipeline import FakePipelineTrainer as _FakeTrainer
from tests.support.pipeline import build_fake_model_artifacts as _build_fake_model_artifacts
from tests.support.pipeline import write_sft_pipeline_config as _write_base_config


pytestmark = pytest.mark.component


def _resolved_plan_for_adapter(adapter):
    return SimpleNamespace(
        model_adapter=adapter,
        fingerprint="test-model-plan-v1",
        artifact_identity=SimpleNamespace(complete=True),
        build_sequence_execution_contract=adapter.build_sequence_execution_contract,
    )


def _write_config(tmp_path: Path, **kwargs):
    config = _write_base_config(tmp_path, **kwargs)
    config.model.model_type = "smoke_vlm"
    config.model.model_name_or_path = "models/Smoke-VLM"
    config.data.media_snapshot_id = "pipeline-fixture-v1"
    return config


def _resolved_resume(path: str | Path, *, step: int = 1) -> ResolvedResumeCheckpoint:
    return ResolvedResumeCheckpoint(
        path=Path(path),
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        global_step=step,
        generation_fingerprint="a" * 64,
        commit_fingerprint="b" * 64,
        stat_guard=(),
    )


def test_sft_checkpoint_step_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 2, "global_step": 2}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        resolve_resume_checkpoint_generation(
            checkpoint,
            protocol=ShaftCheckpointProtocol.BACKEND_NATIVE,
        )


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


@pytest.mark.parametrize(
    ("save_strategy", "expected_immutable"),
    [("steps", True), ("no", False)],
)
def test_sft_pipeline_materializes_model_identity_only_when_checkpointable(
    tmp_path: Path,
    save_strategy: str,
    expected_immutable: bool,
) -> None:
    config = _write_config(tmp_path)
    config.train.save_strategy = save_strategy
    from shaft.pipeline import sft as sft_pipeline

    with patch(
        "shaft.pipeline.sft.resolve_model_plan",
        wraps=sft_pipeline.resolve_model_plan,
    ) as resolver:
        with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as builder:
            builder.return_value = _build_fake_model_artifacts()
            with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                run_sft(config)

    assert resolver.call_args.kwargs["require_immutable_artifact"] is expected_immutable


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
        "shaft.pipeline.sft.resolve_resume_checkpoint_generation",
        return_value=_resolved_resume(config.train.resume_from_checkpoint),
    ):
        with patch("shaft.pipeline.sft.validate_resume_checkpoint"):
            with patch(
                "shaft.pipeline.sft.load_checkpoint_batching_metadata",
                return_value=SimpleNamespace(training_resume_contract=object()),
            ):
                with patch(
                    "shaft.pipeline.sft.build_training_resume_preflight_contract",
                    return_value=SimpleNamespace(fingerprint="preflight-v1"),
                ):
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


def test_sft_resume_drift_fails_before_local_weight_hashing(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    model_dir = tmp_path / "local-hf"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)
    training_args = build_hf_training_args(config)
    batch_contract = build_batch_contract(config=config, training_args=training_args)
    checkpoint_contract = build_training_resume_contract(
        config=config,
        training_args=training_args,
        batch_contract_fingerprint=batch_contract.fingerprint,
        train_input_contract_fingerprint="stored-train-input",
        data_execution_fingerprint="stored-data-execution",
        model_plan_fingerprint="stored-model-plan",
        resolved_finetune_plan_fingerprint="stored-finetune-plan",
        resolved_optimizer_plan_fingerprint="stored-optimizer-plan",
    )
    config.train.learning_rate *= 2
    config.train.resume_from_checkpoint = str(tmp_path / "checkpoint-1")

    def reject_drift(path, *, expected_training_resume_contract, **kwargs):
        _ = path, kwargs
        assert expected_training_resume_contract.fingerprint != checkpoint_contract.fingerprint
        raise ValueError("Training resume contract changed across exact resume")

    with (
        patch(
            "shaft.pipeline.sft.resolve_resume_checkpoint_generation",
            return_value=_resolved_resume(config.train.resume_from_checkpoint),
        ),
        patch("shaft.pipeline.sft.validate_resume_checkpoint"),
        patch(
            "shaft.pipeline.sft.load_checkpoint_batching_metadata",
            return_value=SimpleNamespace(training_resume_contract=checkpoint_contract),
        ),
        patch(
            "shaft.pipeline.sft.validate_batching_resume_contract",
            side_effect=reject_drift,
        ),
        patch(
            "shaft.pipeline.sft.resolve_model_plan",
            side_effect=AssertionError("model identity resolved before cheap preflight"),
        ),
        patch(
            "shaft.model.artifact_identity._file_sha256",
            side_effect=AssertionError("weights hashed before cheap preflight"),
        ),
    ):
        with pytest.raises(ValueError, match="Training resume contract changed"):
            run_sft(config)


def test_run_sft_rejects_invalid_epoch_sharding_before_model_load(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    sharding_calls: list[dict[str, object]] = []

    def _reject_epoch_sharding(self, **kwargs):
        _ = self
        sharding_calls.append(dict(kwargs))
        raise ValueError("unequal per-rank train step counts")

    with patch.object(
        ShaftSampleSampler,
        "validate_epoch_sharding",
        new=_reject_epoch_sharding,
    ):
        with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as build_model:
            with pytest.raises(ValueError, match="unequal per-rank train step counts"):
                run_sft(config)

    assert len(sharding_calls) == 1
    build_model.assert_not_called()


def test_run_sft_rejects_incomplete_data_identity_before_model_load(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)

    class _IncompleteDataCenter:
        def __init__(self, data_config, *, seed, train_sample_budget):
            _ = data_config, seed, train_sample_budget

        def build_dataset_bundle(self, dataset_cls):
            _ = dataset_cls
            return ShaftDatasetBundle(
                train_dataset=object(),
                eval_dataset=None,
                train_execution_fingerprint="incomplete-data-v1",
                train_execution_contract_complete=False,
                train_execution_incomplete_reasons=("missing_media_snapshot_id",),
                train_stream_fingerprint="incomplete-stream-v1",
            )

    with patch("shaft.pipeline.sft.ShaftDataCenter", _IncompleteDataCenter):
        with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as build_model:
            with pytest.raises(ValueError, match="before model loading"):
                run_sft(config)

    build_model.assert_not_called()


def test_run_sft_wires_loss_scale_and_hooks(tmp_path: Path) -> None:
    hook_name = f"pipeline-hook-{tmp_path.name}"

    @hook("after_step", name=hook_name, trajectory_neutral=True)
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


def test_sft_loss_and_online_eval_share_resolved_dataset_pixel_budget(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _write_config(tmp_path)
    config.eval.enabled = True
    config.eval.online_metrics_enabled = True
    config.eval.min_pixels = 200
    config.eval.max_pixels = 2000
    config.eval.datasets = {
        "ds": EvalDatasetPolicyConfig(
            min_pixels=300,
            max_pixels=3000,
            metrics=[EvalMetricConfig(name="parse_success")],
            primary_metric="parse_success",
        )
    }

    with caplog.at_level("INFO", logger="shaft.training.eval_policy"):
        with patch(
            "shaft.pipeline.sft.resolve_eval_input_policy",
            wraps=resolve_eval_input_policy,
        ) as resolve_policy:
            with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as builder:
                builder.return_value = _build_fake_model_artifacts()
                with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
                    run_sft(config)

    resolve_policy.assert_called_once()
    loss_collator = _FakeTrainer.last_kwargs["eval_data_collator"]
    generation_collator = _FakeTrainer.last_kwargs["online_eval_runner"].prompt_collator
    assert loss_collator.input_mode == "training"
    assert loss_collator.padding_side == "right"
    assert generation_collator.input_mode == "generation"
    assert generation_collator.padding_side == "left"
    assert loss_collator._resolve_pixel_budget(["ds"]) == (300, 3000)
    assert generation_collator._resolve_pixel_budget(["ds"]) == (300, 3000)
    assert "[eval-input] default=200:2000 datasets=ds=300:3000" in caplog.text


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
    assert metadata.sample_execution_fingerprint
    assert metadata.train_input_contract is not None
    assert metadata.train_input_contract.exact_resume_safe is True
    assert (
        dict(metadata.train_input_contract.input_options)["sequence_execution_contract_fingerprint"]
        == _FakeTrainer.last_kwargs["efficiency_monitor"].contract.sequence_contract_fingerprint
    )
    assert batch_sampler.schedule.fingerprint == batch_sampler.spec.sample_schedule_fingerprint
    assert (
        metadata.sample_execution_fingerprint
        == _FakeTrainer.last_kwargs["efficiency_monitor"].contract.sample_execution_fingerprint
    )
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
        "shaft.pipeline.sft.resolve_model_plan",
        return_value=_resolved_plan_for_adapter(artifacts.model_adapter),
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
        "shaft.pipeline.sft.resolve_model_plan",
        return_value=_resolved_plan_for_adapter(unsupported_adapter),
    ):
        with patch("shaft.pipeline.sft.ShaftDataCenter") as data_center:
            with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as build_model:
                with pytest.raises(ValueError, match="does not support varlen layout"):
                    run_sft(config)

    data_center.assert_not_called()
    build_model.assert_not_called()


def test_pipeline_revokes_stale_efficiency_summary_before_model_resolution(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_summary = output_dir / "shaft_training_efficiency.json"
    stale_summary.write_text('{"stale": true}\n', encoding="utf-8")

    with patch(
        "shaft.pipeline.sft.resolve_model_plan",
        side_effect=ValueError("invalid model artifact"),
    ):
        with pytest.raises(ValueError, match="invalid model artifact"):
            run_sft(config)

    assert not stale_summary.exists()


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
    commit_training_checkpoint(
        checkpoint,
        world_size=initial_sampler.spec.data_world_size,
        requires_grad_scaler=False,
    )
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


def test_fixed_weighted_unshuffled_pipeline_uses_finite_plan_execution_identity(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.data.schedule.mixing = "weighted"
    config.data.schedule.shuffle = False

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as builder:
        builder.return_value = _build_fake_model_artifacts()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            run_sft(config)

    train_sampler = _FakeTrainer.last_kwargs["train_sampler"]
    efficiency_monitor = _FakeTrainer.last_kwargs["efficiency_monitor"]
    assert train_sampler is not None
    assert efficiency_monitor is not None
    execution_fingerprint = efficiency_monitor.contract.sample_execution_fingerprint
    stream_fingerprint = efficiency_monitor.contract.sample_stream_fingerprint
    assert len(execution_fingerprint) == 64
    assert len(stream_fingerprint) == 64
    assert execution_fingerprint != train_sampler.plan.fingerprint
    assert train_sampler.plan.stream_fingerprint != train_sampler.plan.fingerprint
    assert stream_fingerprint != execution_fingerprint


def test_fixed_weighted_pipeline_publishes_versioned_sample_execution_identity(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.data.schedule.mixing = "weighted"
    config.data.schedule.shuffle = True

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor") as builder:
        builder.return_value = _build_fake_model_artifacts()
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", _FakeTrainer):
            run_sft(config)

    train_sampler = _FakeTrainer.last_kwargs["train_sampler"]
    metadata = load_batching_run_metadata(config.experiment.output_dir)
    assert train_sampler is not None
    assert train_sampler.plan.schedule.ticket_block_size > 0
    assert metadata.sample_execution_fingerprint
    assert (
        metadata.sample_execution_fingerprint
        == _FakeTrainer.last_kwargs["efficiency_monitor"].contract.sample_execution_fingerprint
    )
