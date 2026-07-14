from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file

from shaft.config import load_config
from shaft.model.smoke_vlm import SmokeProcessor
from shaft.observability import PROGRESS_SNAPSHOT_FILENAME
from shaft.pipeline import run_sft
from shaft.training.batch_planning import BATCH_PLANNING_CALLBACK_NAME
from tests.support.configs import write_sft_smoke_config


pytestmark = pytest.mark.smoke


def _run_mode(
    tmp_path: Path, mode: str, *, online_eval: bool = False
) -> tuple[Path, dict[str, float]]:
    cfg_path = write_sft_smoke_config(tmp_path, finetune_mode=mode, online_eval=online_eval)
    cfg = load_config(cfg_path)
    metrics = run_sft(cfg)
    assert "train_loss" in metrics
    assert "epoch" in metrics
    return cfg_path, metrics


def test_smoke_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processor_batch_sizes: list[int] = []
    original_call = SmokeProcessor.__call__

    def _counting_call(self, *args, **kwargs):
        processor_batch_sizes.append(len(kwargs["text"]))
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(SmokeProcessor, "__call__", _counting_call)
    _run_mode(tmp_path, "full")
    assert processor_batch_sizes == [1, 1]


def test_smoke_lora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "lora")


def test_smoke_dora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "dora")


def test_smoke_qlora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "qlora")


def test_cli_non_tty_progress_is_sparse_and_persists_final_state(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=2,
        val_size=1,
        train_steps=1,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "sft",
            "--config",
            str(cfg_path),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, output
    assert "\r" not in output
    assert "\x1b[" not in output
    progress_lines = [line for line in output.splitlines() if "progress " in line]
    assert any("progress data started" in line for line in progress_lines)
    assert any("progress model succeeded" in line for line in progress_lines)
    assert any("progress train succeeded" in line for line in progress_lines)
    assert len(progress_lines) <= 12

    config = load_config(cfg_path)
    snapshot = json.loads(
        (Path(config.experiment.output_dir) / PROGRESS_SNAPSHOT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert snapshot["status"] == "succeeded"
    assert snapshot["active_task_id"] is None
    assert snapshot["tasks"]["train"]["current"] == 1
    assert snapshot["tasks"]["train"]["total"] == 1


def test_cli_forced_interactive_progress_uses_compact_single_line_contract(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=4,
        val_size=1,
        train_steps=2,
    )
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8")
        + "\nprogress:\n"
        + "  enabled: true\n"
        + "  display: interactive\n"
        + "  width: 72\n"
        + "  refresh_interval: 0.01\n"
        + "  leave_completed: false\n"
        + "  persist: true\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "sft",
            "--config",
            str(cfg_path),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr
    decoded = output.decode("utf-8", errors="replace")

    assert completed.returncode == 0, decoded
    assert b"\rtrain" in output
    assert "█" in decoded or "▏" in decoded
    assert "s/step" in decoded
    assert "lr " in decoded
    assert "[----------]" not in decoded
    assert "progress train" not in decoded


def test_bounded_cost_grouping_keeps_fixed_per_device_microbatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=6,
        val_size=1,
        bounded_cost_grouping=True,
        bounded_max_tokens_per_microbatch=2048,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
    )
    train_path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    prompt_lengths = [300, 1, 1, 1, 1, 10]
    for row, prompt_length in zip(rows, prompt_lengths, strict=True):
        row["user_prompt"] = "x" * prompt_length
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    config.eval.enabled = False

    processor_batch_sizes: list[int] = []
    original_call = SmokeProcessor.__call__

    def _counting_call(self, *args, **kwargs):
        processor_batch_sizes.append(len(kwargs["text"]))
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(SmokeProcessor, "__call__", _counting_call)

    metrics = run_sft(config)

    assert metrics["train_loss"] >= 0
    train_batch_sizes = processor_batch_sizes[-2:]
    assert sum(train_batch_sizes) == 6
    assert train_batch_sizes == [3, 3]


def test_token_budget_uses_real_batch_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=3,
        val_size=1,
        bounded_cost_grouping=True,
        bounded_cardinality="token_budget",
        bounded_max_tokens_per_microbatch=450,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
    )
    train_path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    for row, prompt_length in zip(rows, [300, 1, 1], strict=True):
        row["user_prompt"] = "x" * prompt_length
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    config.eval.enabled = False

    processor_batch_sizes: list[int] = []
    original_call = SmokeProcessor.__call__

    def _counting_call(self, *args, **kwargs):
        processor_batch_sizes.append(len(kwargs["text"]))
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(SmokeProcessor, "__call__", _counting_call)

    metrics = run_sft(config)

    assert metrics["train_loss"] >= 0
    assert processor_batch_sizes[-2:] == [1, 2]
    assert metrics["train_samples_per_second"] == pytest.approx(
        round(3 / metrics["train_runtime"], 3)
    )
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "[train-batch] local_packs=1..2 global_packs=1..2 "
        "optimizer_packs=2..4 per_device_train_batch_size=2"
        in message
        for message in messages
    )
    assert not any(
        "Instantaneous batch size per device" in message
        or "Total train batch size (w. parallel, distributed & accumulation)" in message
        for message in messages
    )


def test_bounded_cost_smoke_supports_persistent_workers(tmp_path: Path) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=6,
        val_size=1,
        bounded_cost_grouping=True,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
    )
    config = load_config(cfg_path)
    config.eval.enabled = False
    config.data.num_workers = 2
    config.data.prefetch_factor = 2
    config.data.persistent_workers = True

    metrics = run_sft(config)

    assert metrics["train_loss"] >= 0
    assert metrics["train_steps_per_second"] > 0


def test_smoke_online_eval_canary(tmp_path: Path) -> None:
    cfg_path, _ = _run_mode(tmp_path, "full", online_eval=True)
    cfg = load_config(cfg_path)
    trainer_state_path = Path(cfg.experiment.output_dir) / "checkpoint-1" / "trainer_state.json"
    assert trainer_state_path.exists()
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    assert float(trainer_state["best_metric"]) == 1.0
    assert str(trainer_state["best_model_checkpoint"]).endswith("checkpoint-1")


def test_bounded_exact_resume_and_contract_guard(tmp_path: Path) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=12,
        val_size=1,
        bounded_cost_grouping=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
    )
    initial = load_config(cfg_path)
    initial.eval.enabled = False
    initial.train.duration.value = 2
    initial.train.save_strategy = "steps"
    initial.train.save_steps = 1
    initial.train.save_total_limit = 2
    initial.train.save_final_state = True

    metrics = run_sft(initial)
    checkpoint_one = Path(initial.experiment.output_dir) / "checkpoint-1"
    uninterrupted_checkpoint = Path(initial.experiment.output_dir) / "checkpoint-2"

    assert metrics["train_loss"] >= 0
    assert _bounded_callback_payload(checkpoint_one)

    resumed = load_config(cfg_path)
    resumed.eval.enabled = False
    resumed.train.duration.value = 2
    resumed.experiment.output_dir = str(tmp_path / "bounded-resumed")
    resumed.train.save_strategy = "steps"
    resumed.train.save_steps = 1
    resumed.train.save_total_limit = 2
    resumed.train.save_final_state = True
    resumed.train.resume_from_checkpoint = str(checkpoint_one)

    resumed_metrics = run_sft(resumed)
    resumed_checkpoint = Path(resumed.experiment.output_dir) / "checkpoint-2"

    assert resumed_metrics["train_loss"] >= 0
    _assert_checkpoint_training_state_equal(
        uninterrupted_checkpoint,
        resumed_checkpoint,
    )
    assert _bounded_callback_payload(
        uninterrupted_checkpoint
    ) == _bounded_callback_payload(resumed_checkpoint)

    changed_contract = load_config(cfg_path)
    changed_contract.eval.enabled = False
    changed_contract.train.duration.value = 2
    changed_contract.data.batching.buffer_size = 16
    changed_contract.train.resume_from_checkpoint = str(checkpoint_one)
    with pytest.raises(ValueError, match="changed fields.*buffer_size"):
        run_sft(changed_contract)

    changed_duration = load_config(cfg_path)
    changed_duration.eval.enabled = False
    changed_duration.train.duration.value = 3
    changed_duration.train.resume_from_checkpoint = str(checkpoint_one)
    with pytest.raises(ValueError, match="training contract changed"):
        run_sft(changed_duration)


def test_token_budget_exact_resume_preserves_variable_draw_cursor(
    tmp_path: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=12,
        val_size=1,
        bounded_cost_grouping=True,
        bounded_cardinality="token_budget",
        bounded_max_tokens_per_microbatch=450,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        train_steps=2,
        save_steps=1,
    )
    train_path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["user_prompt"] = "x" * 300
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    initial = load_config(cfg_path)
    initial.eval.enabled = False
    initial.train.save_total_limit = 2
    initial.train.save_final_state = True
    run_sft(initial)
    checkpoint_one = Path(initial.experiment.output_dir) / "checkpoint-1"
    uninterrupted = Path(initial.experiment.output_dir) / "checkpoint-2"

    resumed = load_config(cfg_path)
    resumed.eval.enabled = False
    resumed.experiment.output_dir = str(tmp_path / "token-budget-resumed")
    resumed.train.save_total_limit = 2
    resumed.train.save_final_state = True
    resumed.train.resume_from_checkpoint = str(checkpoint_one)
    run_sft(resumed)
    resumed_checkpoint = Path(resumed.experiment.output_dir) / "checkpoint-2"

    _assert_checkpoint_training_state_equal(uninterrupted, resumed_checkpoint)
    uninterrupted_callback = _bounded_callback_payload(uninterrupted)
    resumed_callback = _bounded_callback_payload(resumed_checkpoint)
    assert uninterrupted_callback == resumed_callback
    assert uninterrupted_callback["args"]["spec"]["cardinality"] == "token_budget"


def test_bounded_resume_rejects_changed_source_execution_contract(tmp_path: Path) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=4,
        val_size=1,
        bounded_cost_grouping=True,
        gradient_accumulation_steps=1,
        train_steps=2,
        save_steps=1,
    )
    initial = load_config(cfg_path)
    initial.eval.enabled = False
    initial.train.save_total_limit = 2
    run_sft(initial)
    checkpoint = Path(initial.experiment.output_dir) / "checkpoint-1"

    train_path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["target_text"] = "changed-supervision-" * 20
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    resumed = load_config(cfg_path)
    resumed.eval.enabled = False
    resumed.train.resume_from_checkpoint = str(checkpoint)

    with pytest.raises(ValueError, match="sample execution changed"):
        run_sft(resumed)


def _assert_checkpoint_training_state_equal(expected: Path, actual: Path) -> None:
    expected_model = load_file(str(expected / "model.safetensors"))
    actual_model = load_file(str(actual / "model.safetensors"))
    assert expected_model.keys() == actual_model.keys()
    for name in expected_model:
        assert torch.equal(expected_model[name], actual_model[name]), name

    expected_optimizer = torch.load(
        expected / "optimizer.pt",
        map_location="cpu",
        weights_only=True,
    )
    actual_optimizer = torch.load(
        actual / "optimizer.pt",
        map_location="cpu",
        weights_only=True,
    )
    _assert_nested_state_equal(expected_optimizer, actual_optimizer)

    expected_scheduler = torch.load(
        expected / "scheduler.pt",
        map_location="cpu",
        weights_only=True,
    )
    actual_scheduler = torch.load(
        actual / "scheduler.pt",
        map_location="cpu",
        weights_only=True,
    )
    _assert_nested_state_equal(expected_scheduler, actual_scheduler)


def _bounded_callback_payload(checkpoint: Path) -> dict:
    trainer_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    return trainer_state["stateful_callbacks"][BATCH_PLANNING_CALLBACK_NAME]


def _assert_nested_state_equal(expected, actual) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(expected, actual)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_state_equal(expected[key], actual[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual)
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_state_equal(expected_item, actual_item)
        return
    assert expected == actual
