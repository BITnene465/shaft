from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from transformers.integrations.deepspeed import deepspeed_config

from shaft.algorithms.rlhf_utils import build_trl_dpo_config
from shaft.config import load_config
from shaft.pipeline import run_rlhf, run_sft
from shaft.pipeline.execution import finalize_training_outputs
from shaft.pipeline.training_args import build_hf_training_args
from shaft.training.checkpointing import ShaftCheckpointProtocol
from tests.support.pipeline import FakePipelineTrainer
from tests.support.pipeline import build_fake_model_artifacts
from tests.support.pipeline import write_sft_pipeline_config
from tests.support.rlhf import write_dpo_config


pytestmark = pytest.mark.component


@pytest.fixture(autouse=True)
def _reset_fake_pipeline_trainer_state():
    FakePipelineTrainer.last_kwargs = None
    yield
    FakePipelineTrainer.last_kwargs = None


def _deepspeed_config(stage: int = 3) -> dict[str, object]:
    return {
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "zero_optimization": {"stage": stage},
    }


def _assert_zero3_is_active() -> None:
    active_config = deepspeed_config()
    assert active_config is not None
    assert active_config["zero_optimization"]["stage"] == 3


def test_sft_pipeline_initializes_deepspeed_before_model_loading(tmp_path: Path) -> None:
    config = write_sft_pipeline_config(tmp_path)
    config.model.model_type = "smoke_vlm"
    config.model.model_name_or_path = "models/Smoke-VLM"
    config.data.media_snapshot_id = "pipeline-distributed-contract-v1"
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config = _deepspeed_config()

    def _fake_build_model(
        runtime_config,
        *,
        init_from_checkpoint=None,
        sequence_execution_contract=None,
        resolved_model_plan=None,
        local_phase_runner=None,
    ):
        assert runtime_config is config
        assert init_from_checkpoint is None
        assert sequence_execution_contract is not None
        assert resolved_model_plan is not None
        assert local_phase_runner is not None
        _assert_zero3_is_active()
        return build_fake_model_artifacts()

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor", _fake_build_model):
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", FakePipelineTrainer):
            metrics = run_sft(config)

    assert "train_loss" in metrics
    assert FakePipelineTrainer.last_kwargs["args"].deepspeed == config.train.distributed.deepspeed.config
    assert (
        FakePipelineTrainer.last_kwargs["shaft_checkpoint_protocol"]
        is ShaftCheckpointProtocol.BACKEND_NATIVE
    )


def test_rlhf_pipeline_initializes_deepspeed_before_model_loading(tmp_path: Path) -> None:
    config = load_config(write_dpo_config(tmp_path))
    config.data.media_snapshot_id = "pipeline-distributed-contract-v1"
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config = _deepspeed_config()

    def _fake_build_model(
        runtime_config,
        *,
        init_from_checkpoint=None,
        sequence_execution_contract=None,
        resolved_model_plan=None,
        local_phase_runner=None,
    ):
        assert runtime_config is config
        assert init_from_checkpoint is None
        assert resolved_model_plan is not None
        assert sequence_execution_contract is not None
        assert local_phase_runner is not None
        _assert_zero3_is_active()
        return build_fake_model_artifacts()

    with patch("shaft.pipeline.rlhf.build_model_tokenizer_processor", _fake_build_model):
        with patch("shaft.algorithms.dpo.ShaftDPOTrainer", FakePipelineTrainer):
            metrics = run_rlhf(config)

    assert "train_loss" in metrics
    assert FakePipelineTrainer.last_kwargs["args"].deepspeed == config.train.distributed.deepspeed.config
    assert (
        FakePipelineTrainer.last_kwargs["shaft_checkpoint_protocol"]
        is ShaftCheckpointProtocol.BACKEND_NATIVE
    )


def test_training_args_expose_active_deepspeed_contract(tmp_path: Path) -> None:
    config = write_sft_pipeline_config(tmp_path)
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config = _deepspeed_config(stage=2)

    args = build_hf_training_args(config)

    assert args.deepspeed == config.train.distributed.deepspeed.config
    assert getattr(args, "hf_deepspeed_config", None) is not None
    assert deepspeed_config()["zero_optimization"]["stage"] == 2


def test_non_deepspeed_training_args_clear_global_deepspeed_state(tmp_path: Path) -> None:
    deepspeed_dir = tmp_path / "deepspeed"
    deepspeed_dir.mkdir()
    deepspeed_runtime = write_sft_pipeline_config(deepspeed_dir)
    deepspeed_runtime.train.distributed.strategy = "deepspeed"
    deepspeed_runtime.train.distributed.deepspeed.config = _deepspeed_config(stage=2)
    _ = build_hf_training_args(deepspeed_runtime)
    assert deepspeed_config()["zero_optimization"]["stage"] == 2

    ddp_dir = tmp_path / "ddp"
    ddp_dir.mkdir()
    ddp_runtime = write_sft_pipeline_config(ddp_dir)
    ddp_args = build_hf_training_args(ddp_runtime)

    assert ddp_args.deepspeed is None
    assert deepspeed_config() is None


def test_dpo_trl_config_preserves_deepspeed_contract(tmp_path: Path) -> None:
    config = load_config(write_dpo_config(tmp_path))
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config = _deepspeed_config(stage=2)

    train_args = build_hf_training_args(config)
    dpo_args = build_trl_dpo_config(train_args=train_args, rlhf_config=config.rlhf.dpo)

    assert dpo_args.deepspeed == config.train.distributed.deepspeed.config
    assert getattr(dpo_args, "hf_deepspeed_config", None) is not None


def test_training_output_finalization_separates_model_save_from_local_file_ops(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class _Trainer:
        def save_model(self, *, output_dir: str) -> None:
            events.append(f"save_model:{output_dir}")

        def save_state(self) -> None:
            events.append("save_state")

    best_export_dir = tmp_path / "best"
    finalize_training_outputs(
        trainer=_Trainer(),
        best_export_dir=best_export_dir,
        save_final_state=True,
        validate_export=lambda _path: events.append("validate_export"),
        prune_output=lambda: events.append("prune_output"),
    )

    assert events == [
        f"save_model:{best_export_dir.resolve()}",
        "validate_export",
        "save_state",
        "prune_output",
    ]


def test_training_output_finalization_uses_one_normalized_export_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_paths: list[str] = []
    validated_paths: list[Path] = []

    class _Trainer:
        def save_model(self, *, output_dir: str) -> None:
            saved_paths.append(output_dir)

    monkeypatch.chdir(tmp_path)
    finalize_training_outputs(
        trainer=_Trainer(),
        best_export_dir="nested/../best",
        save_final_state=False,
        validate_export=validated_paths.append,
        prune_output=lambda: None,
    )

    expected = (tmp_path / "best").resolve()
    assert saved_paths == [str(expected)]
    assert validated_paths == [expected]


def test_training_output_finalization_preserves_local_exception_type() -> None:
    class _Trainer:
        def save_state(self) -> None:
            raise OSError("synthetic local final-state failure")

    with pytest.raises(OSError, match="synthetic local final-state failure"):
        finalize_training_outputs(
            trainer=_Trainer(),
            best_export_dir=None,
            save_final_state=True,
            validate_export=None,
            prune_output=lambda: None,
        )
