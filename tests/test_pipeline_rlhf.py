from __future__ import annotations

import json
from importlib.metadata import version as distribution_version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from shaft.config import EvalDatasetPolicyConfig, load_config
from shaft.algorithms.rlhf_utils import build_trl_dpo_config
from shaft.data import (
    DPODataset,
    GRPODataset,
    SFTDataset,
    ShaftDatasetBundle,
    ShaftGroupedSampleContract,
    ShaftSampleSampler,
)
from shaft.pipeline import run_rlhf
from shaft.training.batch_planning import (
    ShaftBatchingMetadataCallback,
    load_batching_run_metadata,
    build_batch_contract,
)
from shaft.training.resume_contract import build_training_resume_contract
from shaft.training.checkpointing import (
    ResolvedResumeCheckpoint,
    ShaftCheckpointProtocol,
)
from shaft.pipeline.training_args import build_hf_training_args
from shaft.training.progress_callback import ShaftProgressCallback
from tests.support.pipeline import FakePipelineTrainer as _FakeTrainer
from tests.support.pipeline import build_fake_model_artifacts as _build_fake_model_artifacts
from tests.support.rlhf import write_common_image as _write_common_image
from tests.support.rlhf import write_dpo_config as _write_dpo_config
from tests.support.rlhf import write_grpo_config as _write_grpo_config


pytestmark = pytest.mark.component


def _resolved_resume(path: str | Path, *, step: int = 1) -> ResolvedResumeCheckpoint:
    return ResolvedResumeCheckpoint(
        path=Path(path),
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        global_step=step,
        generation_fingerprint="a" * 64,
        commit_fingerprint="b" * 64,
        stat_guard=(),
    )


@pytest.mark.parametrize(
    ("config_writer", "trainer_target", "save_strategy", "expected_immutable"),
    [
        (_write_dpo_config, "shaft.algorithms.dpo.ShaftDPOTrainer", "steps", True),
        (_write_grpo_config, "shaft.algorithms.grpo.ShaftGRPOTrainer", "steps", True),
        (_write_dpo_config, "shaft.algorithms.dpo.ShaftDPOTrainer", "no", False),
    ],
)
def test_rlhf_pipeline_materializes_model_identity_only_when_checkpointable(
    tmp_path: Path,
    config_writer,
    trainer_target: str,
    save_strategy: str,
    expected_immutable: bool,
) -> None:
    cfg = load_config(config_writer(tmp_path))
    cfg.train.save_strategy = save_strategy
    from shaft.pipeline import rlhf as rlhf_pipeline

    with patch(
        "shaft.pipeline.rlhf.resolve_model_plan",
        wraps=rlhf_pipeline.resolve_model_plan,
    ) as resolver:
        with patch("shaft.pipeline.rlhf.build_model_tokenizer_processor") as builder:
            builder.return_value = _build_fake_model_artifacts()
            with patch(trainer_target, _FakeTrainer):
                run_rlhf(cfg)

    assert resolver.call_args.kwargs["require_immutable_artifact"] is expected_immutable


def test_run_rlhf_initializes_seed_before_model_and_adapter_build(tmp_path: Path) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    cfg.experiment.seed = 149
    torch.manual_seed(cfg.experiment.seed + 1)

    def build_seeded_artifacts(*args, **kwargs):
        _ = args, kwargs
        assert torch.initial_seed() == cfg.experiment.seed
        return _build_fake_model_artifacts()

    with patch(
        "shaft.pipeline.rlhf.build_model_tokenizer_processor",
        side_effect=build_seeded_artifacts,
    ):
        with patch("shaft.algorithms.dpo.ShaftDPOTrainer", _FakeTrainer):
            _ = run_rlhf(cfg)


def test_run_rlhf_rejects_resume_contract_before_publish_or_model_load(
    tmp_path: Path,
) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    cfg.train.resume_from_checkpoint = str(tmp_path / "checkpoint-1")

    with patch(
        "shaft.pipeline.rlhf.resolve_resume_checkpoint_generation",
        return_value=_resolved_resume(cfg.train.resume_from_checkpoint),
    ):
        with patch("shaft.pipeline.rlhf.validate_resume_checkpoint"):
            with patch(
                "shaft.pipeline.rlhf.load_checkpoint_batching_metadata",
                return_value=SimpleNamespace(training_resume_contract=object()),
            ):
                with patch(
                    "shaft.pipeline.rlhf.build_training_resume_preflight_contract",
                    return_value=SimpleNamespace(fingerprint="preflight-v1"),
                ):
                    with patch(
                        "shaft.pipeline.rlhf.validate_batching_resume_contract",
                        side_effect=ValueError("batch contract drift"),
                    ):
                        with patch(
                            "shaft.pipeline.rlhf.publish_batching_run_metadata"
                        ) as publish_metadata:
                            with patch(
                                "shaft.pipeline.rlhf.build_model_tokenizer_processor"
                            ) as build_model:
                                with pytest.raises(ValueError, match="batch contract drift"):
                                    run_rlhf(cfg)

    publish_metadata.assert_not_called()
    build_model.assert_not_called()


def test_dpo_beta_resume_drift_fails_before_local_weight_hashing(
    tmp_path: Path,
) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    model_dir = tmp_path / "local-hf"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    cfg.model.model_type = "qwen3vl"
    cfg.model.model_name_or_path = str(model_dir)
    training_args = build_hf_training_args(cfg)
    batch_contract = build_batch_contract(config=cfg, training_args=training_args)
    dpo_args = build_trl_dpo_config(
        train_args=training_args,
        rlhf_config=cfg.rlhf.dpo,
    )
    checkpoint_contract = build_training_resume_contract(
        config=cfg,
        training_args=training_args,
        batch_contract_fingerprint=batch_contract.fingerprint,
        train_input_contract_fingerprint="stored-train-input",
        data_execution_fingerprint="stored-data-execution",
        model_plan_fingerprint="stored-model-plan",
        resolved_finetune_plan_fingerprint="stored-finetune-plan",
        resolved_optimizer_plan_fingerprint="stored-optimizer-plan",
        resolved_dpo_args=dpo_args,
    )
    cfg.rlhf.dpo.beta = 0.25
    cfg.train.resume_from_checkpoint = str(tmp_path / "checkpoint-1")

    def reject_drift(path, *, expected_training_resume_contract, **kwargs):
        _ = path, kwargs
        assert expected_training_resume_contract.fingerprint != checkpoint_contract.fingerprint
        raise ValueError("Training resume contract changed across exact resume")

    with (
        patch(
            "shaft.pipeline.rlhf.resolve_resume_checkpoint_generation",
            return_value=_resolved_resume(cfg.train.resume_from_checkpoint),
        ),
        patch("shaft.pipeline.rlhf.validate_resume_checkpoint"),
        patch(
            "shaft.pipeline.rlhf.load_checkpoint_batching_metadata",
            return_value=SimpleNamespace(
                training_resume_contract=checkpoint_contract
            ),
        ),
        patch(
            "shaft.pipeline.rlhf.validate_batching_resume_contract",
            side_effect=reject_drift,
        ),
        patch(
            "shaft.pipeline.rlhf.resolve_model_plan",
            side_effect=AssertionError("model identity resolved before cheap preflight"),
        ),
        patch(
            "shaft.model.artifact_identity._file_sha256",
            side_effect=AssertionError("weights hashed before cheap preflight"),
        ),
    ):
        with pytest.raises(ValueError, match="Training resume contract changed"):
            run_rlhf(cfg)


def test_run_rlhf_rejects_sample_execution_drift_before_publish_or_model_load(
    tmp_path: Path,
) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    cfg.train.resume_from_checkpoint = str(tmp_path / "checkpoint-1")

    class _FakeDataCenter:
        def __init__(self, data_config, *, seed, train_sample_budget):
            _ = data_config, seed, train_sample_budget

        def build_dataset_bundle(self, dataset_cls):
            assert dataset_cls is DPODataset
            return ShaftDatasetBundle(
                train_dataset=object(),
                eval_dataset=object(),
                train_sampler=SimpleNamespace(plan=object()),
                train_execution_fingerprint="weighted-ticket-execution-v2",
                train_stream_fingerprint="weighted-ticket-stream-v2",
                train_execution_contract_complete=True,
            )

    def validate_contract(
        path,
        *,
        expected_contract,
        expected_sample_execution_fingerprint=None,
        expected_training_resume_contract=None,
        require_train_input_contract_payload=False,
        require_training_resume_contract_payload=False,
    ):
        _ = path, expected_contract, expected_training_resume_contract
        if expected_sample_execution_fingerprint is None:
            assert require_train_input_contract_payload is True
            assert require_training_resume_contract_payload is True
            return
        assert require_train_input_contract_payload is False
        assert require_training_resume_contract_payload is False
        assert expected_sample_execution_fingerprint == "weighted-ticket-execution-v2"
        raise ValueError("sample execution drift")

    with patch(
        "shaft.pipeline.rlhf.resolve_resume_checkpoint_generation",
        return_value=_resolved_resume(cfg.train.resume_from_checkpoint),
    ):
        with patch("shaft.pipeline.rlhf.validate_resume_checkpoint"):
            with patch(
                "shaft.pipeline.rlhf.load_checkpoint_batching_metadata",
                return_value=SimpleNamespace(training_resume_contract=object()),
            ):
                with patch(
                    "shaft.pipeline.rlhf.build_training_resume_preflight_contract",
                    return_value=SimpleNamespace(fingerprint="preflight-v1"),
                ):
                    with patch(
                        "shaft.pipeline.rlhf.validate_batching_resume_contract",
                        side_effect=validate_contract,
                    ):
                        with patch("shaft.pipeline.rlhf.ShaftDataCenter", _FakeDataCenter):
                            with patch(
                                "shaft.pipeline.rlhf.publish_batching_run_metadata"
                            ) as publish_metadata:
                                with patch(
                                    "shaft.pipeline.rlhf.build_model_tokenizer_processor"
                                ) as build_model:
                                    with pytest.raises(
                                        ValueError,
                                        match="sample execution drift",
                                    ):
                                        run_rlhf(cfg)

    publish_metadata.assert_not_called()
    build_model.assert_not_called()


def test_run_dpo_rejects_invalid_epoch_sharding_before_model_load(
    tmp_path: Path,
) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
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
        with patch(
            "shaft.pipeline.rlhf.build_model_tokenizer_processor"
        ) as build_model:
            with pytest.raises(ValueError, match="unequal per-rank train step counts"):
                run_rlhf(cfg)

    assert len(sharding_calls) == 1
    assert sharding_calls[0]["require_equal_rank_batch_cardinality"] is True
    build_model.assert_not_called()


def test_run_rlhf_binds_grpo_grouped_geometry_before_resume(
    tmp_path: Path,
) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    cfg.train.resume_from_checkpoint = str(tmp_path / "checkpoint-1")
    checkpoint = Path(cfg.train.resume_from_checkpoint)
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 2}',
        encoding="utf-8",
    )
    base_fingerprint = "grpo-base-execution-v2"
    grouped_contract = ShaftGroupedSampleContract(
        mini_repeat_count=2,
        batch_size=1,
        iteration_count=1,
        steps_per_iteration=2,
    )

    class _FakeDataCenter:
        def __init__(self, data_config, *, seed, train_sample_budget):
            _ = data_config, seed, train_sample_budget

        def build_dataset_bundle(self, dataset_cls):
            assert dataset_cls is SFTDataset
            return ShaftDatasetBundle(
                train_dataset=object(),
                eval_dataset=object(),
                train_sampler=SimpleNamespace(plan=range(1)),
                train_execution_fingerprint=base_fingerprint,
                train_stream_fingerprint="grpo-grouped-stream-v1",
                train_execution_contract_complete=True,
            )

    def validate_contract(
        path,
        *,
        expected_contract,
        expected_sample_execution_fingerprint=None,
        expected_training_resume_contract=None,
        require_train_input_contract_payload=False,
        require_training_resume_contract_payload=False,
    ):
        _ = path, expected_contract, expected_training_resume_contract
        if expected_sample_execution_fingerprint is None:
            assert require_train_input_contract_payload is True
            assert require_training_resume_contract_payload is True
            return
        assert require_train_input_contract_payload is False
        assert require_training_resume_contract_payload is False
        assert expected_sample_execution_fingerprint == (
            grouped_contract.execution_fingerprint(base_fingerprint)
        )
        raise ValueError("GRPO grouped sample execution drift")

    with (
        patch(
            "shaft.pipeline.rlhf.resolve_resume_checkpoint_generation",
            return_value=_resolved_resume(cfg.train.resume_from_checkpoint),
        ),
        patch("shaft.pipeline.rlhf.validate_resume_checkpoint"),
        patch(
            "shaft.pipeline.rlhf.load_checkpoint_batching_metadata",
            return_value=SimpleNamespace(training_resume_contract=object()),
        ),
        patch(
            "shaft.pipeline.rlhf.build_training_resume_preflight_contract",
            return_value=SimpleNamespace(fingerprint="preflight-v1"),
        ),
        patch(
            "shaft.pipeline.rlhf.validate_batching_resume_contract",
            side_effect=validate_contract,
        ),
        patch("shaft.pipeline.rlhf.ShaftDataCenter", _FakeDataCenter),
        patch(
            "shaft.pipeline.rlhf.publish_batching_run_metadata"
        ) as publish_metadata,
        patch(
            "shaft.pipeline.rlhf.build_model_tokenizer_processor"
        ) as build_model,
    ):
        with pytest.raises(
            ValueError,
            match="GRPO grouped sample execution drift",
        ):
            run_rlhf(cfg)

    publish_metadata.assert_not_called()
    build_model.assert_not_called()


def test_run_rlhf_rejects_grpo_mid_generation_resume_before_model_load(
    tmp_path: Path,
) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 1}',
        encoding="utf-8",
    )
    cfg.train.resume_from_checkpoint = str(checkpoint)

    class _FakeDataCenter:
        def __init__(self, data_config, *, seed, train_sample_budget):
            _ = data_config, seed, train_sample_budget

        def build_dataset_bundle(self, dataset_cls):
            assert dataset_cls is SFTDataset
            return ShaftDatasetBundle(
                train_dataset=object(),
                eval_dataset=object(),
                train_sampler=SimpleNamespace(plan=range(1)),
                train_execution_fingerprint="grpo-mid-cycle-v1",
                train_stream_fingerprint="grpo-mid-cycle-stream-v1",
                train_execution_contract_complete=True,
            )

    with (
        patch(
            "shaft.pipeline.rlhf.resolve_resume_checkpoint_generation",
            return_value=_resolved_resume(checkpoint),
        ),
        patch("shaft.pipeline.rlhf.validate_resume_checkpoint"),
        patch(
            "shaft.pipeline.rlhf.load_checkpoint_batching_metadata",
            return_value=SimpleNamespace(training_resume_contract=object()),
        ),
        patch(
            "shaft.pipeline.rlhf.build_training_resume_preflight_contract",
            return_value=SimpleNamespace(fingerprint="preflight-v1"),
        ),
        patch("shaft.pipeline.rlhf.validate_batching_resume_contract"),
        patch("shaft.pipeline.rlhf.ShaftDataCenter", _FakeDataCenter),
        patch(
            "shaft.pipeline.rlhf.build_model_tokenizer_processor"
        ) as build_model,
    ):
        with pytest.raises(ValueError, match="generation-reuse cycle"):
            run_rlhf(cfg)

    build_model.assert_not_called()


@pytest.mark.parametrize("operation", ["save", "resume"])
def test_run_grpo_rejects_vllm_checkpointing_before_data_or_model_load(
    tmp_path: Path,
    operation: str,
) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    cfg.rlhf.grpo.vllm.enabled = True
    cfg.rlhf.grpo.use_vllm = True
    if operation == "save":
        cfg.train.save_strategy = "steps"
    else:
        cfg.train.save_strategy = "no"
        cfg.train.resume_from_checkpoint = str(tmp_path / "checkpoint-1")

    with patch(
        "shaft.pipeline.rlhf.validate_grpo_vllm_runtime_compatibility"
    ):
        with patch("shaft.pipeline.rlhf.ShaftDataCenter") as data_center:
            with patch(
                "shaft.pipeline.rlhf.build_model_tokenizer_processor"
            ) as build_model:
                with pytest.raises(ValueError, match="vLLM.*RNG state"):
                    run_rlhf(cfg)

    data_center.assert_not_called()
    build_model.assert_not_called()


def test_run_grpo_rejects_incompatible_vllm_before_model_plan_or_data(
    tmp_path: Path,
) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    cfg.rlhf.grpo.vllm.enabled = True
    cfg.rlhf.grpo.use_vllm = True

    def version(distribution: str) -> str:
        return {"trl": "0.29.1", "vllm": "0.19.1"}[distribution]

    with (
        patch(
            "shaft.algorithms.rlhf_utils.metadata.requires",
            return_value=['vllm<0.13.0,>=0.10.2; extra == "vllm"'],
        ),
        patch("shaft.algorithms.rlhf_utils.metadata.version", side_effect=version),
        patch("shaft.pipeline.rlhf.resolve_model_plan") as resolve_plan,
        patch("shaft.pipeline.rlhf.ShaftDataCenter") as data_center,
        patch(
            "shaft.pipeline.rlhf.build_model_tokenizer_processor"
        ) as build_model,
    ):
        with pytest.raises(
            ValueError,
            match=(
                r"required_vllm_spec=<0\.13\.0,>=0\.10\.2, "
                r"installed_vllm_version=0\.19\.1"
            ),
        ):
            run_rlhf(cfg)

    resolve_plan.assert_not_called()
    data_center.assert_not_called()
    build_model.assert_not_called()


def test_run_grpo_accepts_compatible_vllm_metadata(
    tmp_path: Path,
) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    cfg.rlhf.grpo.vllm.enabled = True
    cfg.rlhf.grpo.use_vllm = True

    def version(distribution: str) -> str:
        versions = {"trl": "0.29.1", "vllm": "0.12.1"}
        return versions.get(distribution, distribution_version(distribution))

    with (
        patch(
            "shaft.algorithms.rlhf_utils.metadata.requires",
            return_value=['vllm<0.13.0,>=0.10.2; extra == "vllm"'],
        ),
        patch("shaft.algorithms.rlhf_utils.metadata.version", side_effect=version),
        patch(
            "shaft.pipeline.rlhf.build_model_tokenizer_processor",
            return_value=_build_fake_model_artifacts(),
        ),
        patch("shaft.algorithms.grpo.ShaftGRPOTrainer", _FakeTrainer),
    ):
        metrics = run_rlhf(cfg)

    assert "train_loss" in metrics


def test_run_grpo_rejects_incomplete_prompt_group_before_model_load(
    tmp_path: Path,
) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    cfg.train.per_device_train_batch_size = 3

    class _FakeDataCenter:
        def __init__(self, data_config, *, seed, train_sample_budget):
            _ = data_config, seed, train_sample_budget

        def build_dataset_bundle(self, dataset_cls):
            assert dataset_cls is SFTDataset
            return ShaftDatasetBundle(
                train_dataset=object(),
                eval_dataset=object(),
                train_sampler=SimpleNamespace(plan=range(5)),
                train_execution_fingerprint="grpo-incomplete-group-v1",
                train_stream_fingerprint="grpo-incomplete-group-stream-v1",
                train_execution_contract_complete=True,
            )

    with patch("shaft.pipeline.rlhf.ShaftDataCenter", _FakeDataCenter):
        with patch(
            "shaft.pipeline.rlhf.build_model_tokenizer_processor"
        ) as build_model:
            with pytest.raises(ValueError, match="complete grouped batches"):
                run_rlhf(cfg)

    build_model.assert_not_called()


def test_run_rlhf_rank_nonzero_skips_run_level_file_ops(tmp_path: Path) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    cfg.train.save_final_model = True
    cfg.train.save_final_state = True

    with patch("shaft.algorithms.grpo.ShaftGRPOTrainer", _FakeTrainer):
        with patch("shaft.pipeline.rlhf.is_rank_zero", return_value=False):
            with patch("shaft.pipeline.execution.is_rank_zero", return_value=False):
                with patch("shaft.pipeline.rlhf.ensure_hf_export_layout") as mocked_ensure:
                    with patch("shaft.pipeline.rlhf.prune_root_output_layout") as mocked_prune:
                        metrics = run_rlhf(cfg)

    assert "train_loss" in metrics
    mocked_ensure.assert_not_called()
    mocked_prune.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("ensure", "synthetic RLHF export validation failure"),
        ("prune", "synthetic RLHF output prune failure"),
    ],
)
def test_run_rlhf_propagates_local_finalization_file_failure(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    cfg.train.save_final_model = mode == "ensure"
    target = (
        "shaft.pipeline.rlhf.ensure_hf_export_layout"
        if mode == "ensure"
        else "shaft.pipeline.rlhf.prune_root_output_layout"
    )

    with patch("shaft.algorithms.grpo.ShaftGRPOTrainer", _FakeTrainer):
        with patch(target, side_effect=OSError(message)):
            with pytest.raises(OSError, match=message):
                run_rlhf(cfg)


def test_run_rlhf_uses_data_center_for_dpo(tmp_path: Path) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    fake_train_dataset = object()
    fake_eval_dataset = object()
    fake_train_sampler = SimpleNamespace(plan=object())
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
                train_execution_fingerprint="dpo-execution-v2",
                train_stream_fingerprint="dpo-stream-v2",
                train_execution_contract_complete=True,
            )

    with patch("shaft.pipeline.rlhf.ShaftDataCenter", _FakeDataCenter):
        with patch("shaft.pipeline.rlhf.build_model_tokenizer_processor") as mocked_builder:
            mocked_builder.return_value = _build_fake_model_artifacts()
            with patch("shaft.algorithms.dpo.ShaftDPOTrainer", _FakeTrainer):
                _ = run_rlhf(cfg)

    assert captured["data_config"] is cfg.data
    assert captured["seed"] == cfg.experiment.seed
    assert captured["train_sample_budget"] == 1
    assert captured["dataset_cls"] is DPODataset
    assert _FakeTrainer.last_kwargs["train_dataset"] is fake_train_dataset
    assert _FakeTrainer.last_kwargs["train_sampler"] is fake_train_sampler
    assert _FakeTrainer.last_kwargs["eval_dataset"] is None
    assert _FakeTrainer.last_kwargs["model_adapter"] is mocked_builder.return_value.model_adapter
    assert (
        _FakeTrainer.last_kwargs["finetune_plan"]
        is mocked_builder.return_value.finetune_plan
    )
    metadata = load_batching_run_metadata(cfg.experiment.output_dir)
    assert metadata.grouping == "none"
    assert metadata.cardinality == "fixed"
    assert metadata.packing == "none"
    assert metadata.layout == "padded"
    assert metadata.sample_execution_fingerprint == "dpo-execution-v2"
    assert any(
        isinstance(callback, ShaftBatchingMetadataCallback)
        for callback in _FakeTrainer.last_kwargs["callbacks"]
    )


def test_dpo_uses_distinct_train_and_eval_pixel_budget_collators(tmp_path: Path) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    cfg.data.min_pixels = 100
    cfg.data.max_pixels = 1000
    cfg.eval.enabled = True
    cfg.eval.min_pixels = 200
    cfg.eval.max_pixels = 2000
    cfg.eval.datasets = {
        "dpo_ds": EvalDatasetPolicyConfig(min_pixels=300, max_pixels=3000)
    }
    fake_train_dataset = object()
    fake_eval_dataset = object()
    fake_eval_datasets_by_name = {"dpo_ds": fake_eval_dataset}

    class _FakeDataCenter:
        def __init__(self, data_config, *, seed, train_sample_budget):
            _ = data_config, seed, train_sample_budget

        def build_dataset_bundle(self, dataset_cls):
            assert dataset_cls is DPODataset
            return ShaftDatasetBundle(
                train_dataset=fake_train_dataset,
                eval_dataset=fake_eval_dataset,
                eval_datasets_by_name=fake_eval_datasets_by_name,
                train_sampler=SimpleNamespace(plan=object()),
                train_execution_fingerprint="dpo-eval-execution-v2",
                train_stream_fingerprint="dpo-eval-stream-v2",
                train_execution_contract_complete=True,
            )

    with patch("shaft.pipeline.rlhf.ShaftDataCenter", _FakeDataCenter):
        with patch("shaft.pipeline.rlhf.build_model_tokenizer_processor") as builder:
            builder.return_value = _build_fake_model_artifacts()
            with patch("shaft.algorithms.dpo.ShaftDPOTrainer", _FakeTrainer):
                run_rlhf(cfg)

    train_collator = _FakeTrainer.last_kwargs["data_collator"]
    eval_collator = _FakeTrainer.last_kwargs["eval_data_collator"]
    assert train_collator is not eval_collator
    assert (train_collator.min_pixels, train_collator.max_pixels) == (100, 1000)
    assert (eval_collator.min_pixels, eval_collator.max_pixels) == (200, 2000)
    assert eval_collator._resolve_pixel_budget(["dpo_ds"]) == (300, 3000)
    assert _FakeTrainer.last_kwargs["eval_dataset"] is fake_eval_datasets_by_name


def test_run_rlhf_uses_sft_dataset_for_grpo(tmp_path: Path) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    fake_train_dataset = object()
    fake_eval_dataset = object()
    fake_train_sampler = SimpleNamespace(plan=range(1))
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
                train_execution_fingerprint="grpo-execution-v2",
                train_stream_fingerprint="grpo-stream-v2",
                train_execution_contract_complete=True,
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
    assert _FakeTrainer.last_kwargs["sample_plan"] is fake_train_sampler.plan
    assert "finetune_mode" not in _FakeTrainer.last_kwargs
    assert "data_collator" not in _FakeTrainer.last_kwargs
    assert _FakeTrainer.last_kwargs["model_adapter"] is mocked_builder.return_value.model_adapter
    assert (
        _FakeTrainer.last_kwargs["finetune_plan"]
        is mocked_builder.return_value.finetune_plan
    )
    grouped_contract = _FakeTrainer.last_kwargs["grouped_sample_contract"]
    assert grouped_contract == ShaftGroupedSampleContract(
        mini_repeat_count=2,
        batch_size=1,
        iteration_count=1,
        steps_per_iteration=2,
    )
    metadata = load_batching_run_metadata(cfg.experiment.output_dir)
    assert metadata.sample_execution_fingerprint == grouped_contract.execution_fingerprint(
        "grpo-execution-v2"
    )


def test_run_grpo_uses_grouped_unique_prompt_budget_for_step_duration(
    tmp_path: Path,
) -> None:
    cfg = load_config(_write_grpo_config(tmp_path, sample_count=3))
    cfg.train.gradient_accumulation_steps = 7
    cfg.rlhf.grpo.num_generations = 3
    captured: dict[str, object] = {}

    class _FakeDataCenter:
        def __init__(self, data_config, *, seed, train_sample_budget):
            _ = data_config, seed
            captured["train_sample_budget"] = train_sample_budget

        def build_dataset_bundle(self, dataset_cls):
            assert dataset_cls is SFTDataset
            return ShaftDatasetBundle(
                train_dataset=object(),
                eval_dataset=object(),
                train_sampler=SimpleNamespace(plan=range(3)),
                train_execution_fingerprint="grpo-step-horizon-v1",
                train_stream_fingerprint="grpo-step-horizon-stream-v1",
                train_execution_contract_complete=True,
            )

    with patch("shaft.pipeline.rlhf.ShaftDataCenter", _FakeDataCenter):
        with patch("shaft.pipeline.rlhf.build_model_tokenizer_processor") as builder:
            builder.return_value = _build_fake_model_artifacts()
            with patch("shaft.algorithms.grpo.ShaftGRPOTrainer", _FakeTrainer):
                _ = run_rlhf(cfg)

    grouped_contract = _FakeTrainer.last_kwargs["grouped_sample_contract"]
    assert grouped_contract == ShaftGroupedSampleContract(
        mini_repeat_count=3,
        batch_size=3,
        iteration_count=1,
        steps_per_iteration=9,
    )
    assert captured["train_sample_budget"] == 3


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
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
  datasets:
    - dataset_name: grpo_ds
      source_type: jsonl_sft
      train_path: {train_jsonl}
      val_path: {val_jsonl}
  num_workers: 0
  persistent_workers: false
  pin_memory: false
train:
  duration:
    unit: steps
    value: 1
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
        def __init__(self, data_config, *, seed, train_sample_budget):
            _ = data_config, seed, train_sample_budget

        def build_dataset_bundle(self, dataset_cls):
            assert dataset_cls is SFTDataset
            return ShaftDatasetBundle(
                train_dataset=fake_train_dataset,
                eval_dataset=object(),
                eval_datasets_by_name=fake_eval_datasets_by_name,
                train_sampler=SimpleNamespace(plan=range(1)),
                train_execution_fingerprint="grpo-eval-execution-v2",
                train_stream_fingerprint="grpo-eval-stream-v2",
                train_execution_contract_complete=True,
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
    progress_callback = next(
        callback
        for callback in _FakeTrainer.last_kwargs["callbacks"]
        if isinstance(callback, ShaftProgressCallback)
    )
    assert (
        _FakeTrainer.last_kwargs["online_eval_runner"].progress_manager
        is progress_callback.progress_manager
    )
