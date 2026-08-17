from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest
from safetensors.torch import load_file
import torch

from shaft.config import load_config
from shaft.model import build_model_tokenizer_processor
from shaft.opd import ShaftOPDTrainer
from shaft.opd.loss import OPDTeacherDistribution
from shaft.opd.remote_teacher import (
    OPDTeacherIdentity,
    PROTOCOL_VERSION,
    decode_teacher_score_request,
    encode_teacher_distribution,
)
from shaft.opd.input_abi import build_opd_input_abi
from shaft.opd.telemetry import OPD_TELEMETRY_FILENAME
from shaft.pipeline import run_opd
from shaft.training import load_batching_run_metadata
from shaft.training.checkpointing import validate_training_checkpoint_commit
from tests.support.opd import write_opd_config


@pytest.mark.parametrize("finetune_mode", ["full", "lora"])
def test_run_opd_updates_only_student_and_publishes_contract(
    tmp_path: Path,
    monkeypatch,
    finetune_mode: str,
) -> None:
    config = load_config(
        write_opd_config(
            tmp_path,
            train_steps=2,
            save_final_model=True,
            finetune_mode=finetune_mode,
        )
    )
    captured: dict[str, object] = {}
    original_train = ShaftOPDTrainer.train

    def _capture_train(self, *args, **kwargs):
        teacher_model = self.execution_runtime.teacher_provider.model
        student_before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        teacher_before = {
            name: parameter.detach().clone()
            for name, parameter in teacher_model.named_parameters()
        }
        result = original_train(self, *args, **kwargs)
        captured["student_changed"] = any(
            not torch.equal(student_before[name], parameter.detach())
            for name, parameter in self.model.named_parameters()
        )
        captured["teacher_unchanged"] = all(
            torch.equal(teacher_before[name], parameter.detach())
            for name, parameter in teacher_model.named_parameters()
        )
        captured["teacher_frozen"] = all(
            not parameter.requires_grad for parameter in teacher_model.parameters()
        )
        return result

    monkeypatch.setattr(ShaftOPDTrainer, "train", _capture_train)
    metrics = run_opd(config)

    assert "train_loss" in metrics
    assert captured == {
        "student_changed": True,
        "teacher_unchanged": True,
        "teacher_frozen": True,
    }
    metadata = load_batching_run_metadata(config.experiment.output_dir)
    assert metadata.sample_execution_fingerprint
    assert metadata.training_resume_contract is not None
    assert metadata.training_resume_contract.algorithm == "opd"
    objective = dict(metadata.training_resume_contract.objective)
    assert objective["teacher_model_plan_fingerprint"]
    assert objective["teacher_student_input_abi_fingerprint"]
    export_dir = Path(config.experiment.output_dir) / "best"
    if finetune_mode == "full":
        assert (export_dir / "config.json").is_file()
        assert (export_dir / "model.safetensors").is_file()
    else:
        assert (export_dir / "adapter_config.json").is_file()
        adapter_state = load_file(str(export_dir / "adapter_model.safetensors"))
        assert adapter_state
        assert any(bool(value.ne(0).any().item()) for value in adapter_state.values())
    assert (export_dir / "smoke_processor.json").is_file()


@pytest.mark.parametrize("finetune_mode", ["full", "lora"])
def test_run_opd_sampled_rollout_exact_resume_matches_uninterrupted(
    tmp_path: Path,
    finetune_mode: str,
) -> None:
    config_path = write_opd_config(
        tmp_path,
        train_steps=2,
        save_steps=1,
        do_sample=True,
        train_size=4,
        gradient_accumulation_steps=2,
        finetune_mode=finetune_mode,
    )
    uninterrupted = load_config(config_path)
    uninterrupted.train.efficiency.enabled = True
    run_opd(uninterrupted)

    checkpoint_one = Path(uninterrupted.experiment.output_dir) / "checkpoint-1"
    expected_final = Path(uninterrupted.experiment.output_dir) / "checkpoint-2"
    validate_training_checkpoint_commit(checkpoint_one)
    validate_training_checkpoint_commit(expected_final)

    resumed = load_config(config_path)
    resumed.train.efficiency.enabled = True
    resumed.experiment.output_dir = str(tmp_path / "outputs_opd_resumed")
    resumed.train.resume_from_checkpoint = str(checkpoint_one)
    run_opd(resumed)

    actual_final = Path(resumed.experiment.output_dir) / "checkpoint-2"
    validate_training_checkpoint_commit(actual_final)
    _assert_checkpoint_state_equal(expected_final, actual_final)
    telemetry = json.loads(
        (Path(resumed.experiment.output_dir) / OPD_TELEMETRY_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert [frame["global_step"] for frame in telemetry["rank_frames"][0]] == [1, 2]


def test_run_opd_publishes_phase_telemetry_and_checkpoint_snapshot(tmp_path: Path) -> None:
    config = load_config(
        write_opd_config(
            tmp_path,
            train_steps=1,
            save_steps=1,
        )
    )
    config.train.efficiency.enabled = True
    config.train.efficiency.persist = True
    metrics = run_opd(config)

    assert metrics["opd_efficiency/completion_tokens_per_second"] > 0
    summary = Path(config.experiment.output_dir) / OPD_TELEMETRY_FILENAME
    assert summary.is_file()
    checkpoint = Path(config.experiment.output_dir) / "checkpoint-1"
    assert (checkpoint / "shaft_opd_telemetry_rank0.json").is_file()
    validate_training_checkpoint_commit(checkpoint)


def test_run_opd_with_external_teacher_protocol_updates_student(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(write_opd_config(tmp_path, train_steps=1))
    teacher_artifacts = build_model_tokenizer_processor(config)
    with torch.no_grad():
        next(teacher_artifacts.model.parameters()).add_(0.25)
    teacher_artifacts.model.eval()
    input_abi = build_opd_input_abi(teacher_artifacts)
    artifact_fingerprint = "a" * 64
    identity = OPDTeacherIdentity(
        protocol_version=PROTOCOL_VERSION,
        artifact_fingerprint=artifact_fingerprint,
        model_type=teacher_artifacts.model_adapter.model_type,
        input_abi=input_abi,
    )

    class InProcessTransport:
        def __init__(self, remote_config):
            _ = remote_config

        def get_identity(self):
            return identity

        def score(self, payload, *, idempotency_key):
            assert idempotency_key
            request = decode_teacher_score_request(payload, max_bytes=len(payload))
            with torch.no_grad():
                logits = teacher_artifacts.model(**request.model_inputs).logits
            flattened = logits[:, :-1, :][request.causal_position_mask]
            distribution: OPDTeacherDistribution = (
                request.objective_plan.build_teacher_distribution(flattened)
            )
            return encode_teacher_distribution(distribution)

    monkeypatch.setattr(
        "shaft.opd.remote_teacher.UrllibOPDTeacherHTTPTransport",
        InProcessTransport,
    )
    config.opd.teacher.provider = "http"
    config.opd.teacher.model_name_or_path = ""
    config.opd.teacher.remote.endpoint = "http://teacher.invalid"
    config.opd.teacher.remote.artifact_fingerprint = artifact_fingerprint
    config.train.efficiency.enabled = True
    metrics = run_opd(config)

    assert metrics["train_loss"] > 0
    assert metrics["opd_efficiency/completion_tokens_per_second"] > 0


def _assert_checkpoint_state_equal(expected: Path, actual: Path) -> None:
    model_filename = (
        "model.safetensors"
        if (expected / "model.safetensors").is_file()
        else "adapter_model.safetensors"
    )
    expected_model = load_file(str(expected / model_filename))
    actual_model = load_file(str(actual / model_filename))
    assert expected_model.keys() == actual_model.keys()
    for name in expected_model:
        assert torch.equal(expected_model[name], actual_model[name]), name

    for filename in ("optimizer.pt", "scheduler.pt"):
        _assert_nested_equal(
            torch.load(expected / filename, map_location="cpu", weights_only=True),
            torch.load(actual / filename, map_location="cpu", weights_only=True),
        )
    expected_rng = sorted(expected.glob("rng_state*.pth"))
    actual_rng = sorted(actual.glob("rng_state*.pth"))
    assert [path.name for path in expected_rng] == [path.name for path in actual_rng]
    for expected_path, actual_path in zip(expected_rng, actual_rng, strict=True):
        _assert_nested_equal(
            torch.load(expected_path, map_location="cpu", weights_only=False),
            torch.load(actual_path, map_location="cpu", weights_only=False),
        )


def _assert_nested_equal(expected, actual) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(expected, actual)
        return
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert np.array_equal(expected, actual)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_equal(expected_item, actual_item)
        return
    assert expected == actual
