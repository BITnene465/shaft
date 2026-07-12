from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import load_file
import torch

from shaft.observability import PROGRESS_SNAPSHOT_FILENAME
from shaft.training.batch_planning import (
    BATCHING_RUN_METADATA_FILENAME,
    BOUNDED_BATCHING_CALLBACK_NAME,
    checkpoint_has_bounded_batching_state,
)
from shaft.training.checkpointing import resolve_resume_checkpoint
from tests.support.configs import write_sft_smoke_config


pytestmark = pytest.mark.smoke


def _run_torchrun(
    repo_root: Path,
    config_path: Path,
    *extra_args: str,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            "scripts/train.py",
            "sft",
            "--config",
            str(config_path),
            *extra_args,
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _assert_torchrun_succeeded(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.returncode == 0:
        return
    output = f"{completed.stdout}\n{completed.stderr}"
    if "Operation not permitted" in output and "RendezvousConnectionError" in output:
        pytest.skip("torchrun rendezvous is blocked by current sandbox/network policy.")
    raise AssertionError(
        f"torchrun failed (code={completed.returncode}).\n{output}"
    )


def _run_bounded_fault(
    repo_root: Path,
    config_path: Path,
    mode: str,
    *,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            "tests/support/distributed_bounded_fault.py",
            str(config_path),
            mode,
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _run_progress_console(
    repo_root: Path,
    sync_dir: Path,
    *,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            "tests/support/distributed_progress_console.py",
            str(sync_dir),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def test_torchrun_train_eval_smoke(tmp_path: Path, repo_root: Path) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=4,
        val_size=2,
        distributed=True,
    )
    completed = _run_torchrun(repo_root, config_path, "--max-steps", "1")
    _assert_torchrun_succeeded(completed)

    output = f"{completed.stdout}\n{completed.stderr}"
    assert output.count("progress train started") == 1
    assert "\r" not in output
    progress = json.loads(
        (tmp_path / "outputs" / PROGRESS_SNAPSHOT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert progress["status"] == "succeeded"
    assert progress["tasks"]["train"]["current"] == 1


def test_torchrun_interactive_progress_keeps_one_rank_zero_console_line(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    completed = _run_progress_console(repo_root, tmp_path / "sync")
    _assert_torchrun_succeeded(completed)

    output = f"{completed.stdout}\n{completed.stderr}"
    assert output.count("rank-zero-warning") == 1
    assert "rank-one-warning-must-be-hidden" not in output
    assert "0/10k 0%" in output
    assert "1/10k 0.01%" in output
    warning_end = output.index("rank-zero-warning") + len("rank-zero-warning")
    assert output[warning_end:].lstrip().startswith("train")


def test_torchrun_bounded_variable_batch_contract(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=20,
        val_size=2,
        distributed=True,
        bounded_cost_aware=True,
        bounded_max_samples_per_microbatch=4,
        bounded_max_padded_tokens=512,
        gradient_accumulation_steps=2,
        train_steps=2,
        save_steps=1,
    )
    train_path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["user_prompt"] = "x" * 300
    rows[7]["user_prompt"] = "y" * 260
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    completed = _run_torchrun(repo_root, config_path)
    _assert_torchrun_succeeded(completed)

    output = f"{completed.stdout}\n{completed.stderr}"
    assert "[bounded-batch]" in output
    assert "[bounded-batch-planned-summary]" in output
    metadata = json.loads(
        (tmp_path / "outputs" / BATCHING_RUN_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["strategy"] == "bounded_cost_aware"
    assert metadata["buffer_size"] == 8
    assert _bounded_callback_payload(tmp_path / "outputs" / "checkpoint-1")
    assert not list((tmp_path / "outputs").glob("*cost_plan*"))


def test_torchrun_bounded_exact_resume_with_persistent_workers(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        output_name="uninterrupted",
        train_size=36,
        val_size=3,
        distributed=True,
        bounded_cost_aware=True,
        bounded_max_samples_per_microbatch=3,
        bounded_max_padded_tokens=512,
        gradient_accumulation_steps=2,
        train_steps=3,
        save_steps=1,
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        .replace("  num_workers: 0", "  num_workers: 2", 1)
        .replace("  persistent_workers: false", "  persistent_workers: true", 1)
        .replace("  save_total_limit: 2", "  save_total_limit: 3", 1),
        encoding="utf-8",
    )
    train_path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    for index in range(0, 36, 6):
        rows[index]["user_prompt"] = "x" * 300
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    uninterrupted = _run_torchrun(repo_root, config_path)
    _assert_torchrun_succeeded(uninterrupted)
    checkpoint_one = tmp_path / "uninterrupted" / "checkpoint-1"
    expected_final = tmp_path / "uninterrupted" / "checkpoint-3"
    assert _bounded_callback_payload(checkpoint_one)

    resumed_output = tmp_path / "resumed"
    resume_config = tmp_path / "resume.yaml"
    resume_config.write_text(
        config_path.read_text(encoding="utf-8").replace(
            f"output_dir: {tmp_path / 'uninterrupted'}",
            f"output_dir: {resumed_output}",
            1,
        ),
        encoding="utf-8",
    )
    resumed = _run_torchrun(
        repo_root,
        resume_config,
        "--resume-from",
        str(checkpoint_one),
    )
    _assert_torchrun_succeeded(resumed)

    _assert_exact_checkpoint(expected_final, resumed_output / "checkpoint-3")


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        (
            "constructor_failure",
            "synthetic rank-local provider construction failure",
        ),
        ("provider_failure", "synthetic rank-local media failure"),
        ("cost_drift", "first-buffer costs or plans differ"),
    ],
)
def test_torchrun_bounded_startup_faults_fail_all_ranks_without_hanging(
    tmp_path: Path,
    repo_root: Path,
    mode: str,
    message: str,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        distributed=True,
        bounded_cost_aware=True,
        train_size=12,
        val_size=1,
        train_steps=1,
    )

    completed = _run_bounded_fault(repo_root, config_path, mode)
    output = f"{completed.stdout}\n{completed.stderr}"

    assert completed.returncode != 0
    assert message in output


def test_torchrun_bounded_checkpoint_state_failure_preserves_last_complete_resume(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        distributed=True,
        bounded_cost_aware=True,
        train_size=20,
        val_size=1,
        train_steps=2,
        save_steps=1,
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  save_total_limit: 2",
            "  save_total_limit: 1",
            1,
        ),
        encoding="utf-8",
    )

    completed = _run_bounded_fault(
        repo_root,
        config_path,
        "checkpoint_write_failure",
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    run_dir = tmp_path / "outputs"
    checkpoint_one = run_dir / "checkpoint-1"

    assert completed.returncode != 0
    assert "synthetic bounded trainer-state write failure" in output
    assert checkpoint_has_bounded_batching_state(checkpoint_one)
    assert resolve_resume_checkpoint(
        run_dir,
        require_bounded_state=True,
    ) == str(checkpoint_one)


def test_torchrun_global_weighted_loss_matches_single_process_reference(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    result_path = tmp_path / "distributed-loss-result.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            "tests/support/distributed_loss_probe.py",
            str(result_path),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    _assert_torchrun_succeeded(completed)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["rank_batch_sizes"] == [[1, 1], [2, 1]]
    assert result["global_denominator"] == pytest.approx(12.0)
    assert result["reference_loss"] > 0
    assert result["max_parameter_error"] < 1e-7


def _assert_exact_checkpoint(expected: Path, actual: Path) -> None:
    expected_model = load_file(str(expected / "model.safetensors"))
    actual_model = load_file(str(actual / "model.safetensors"))
    assert expected_model.keys() == actual_model.keys()
    for name in expected_model:
        assert torch.equal(expected_model[name], actual_model[name]), name
    for state_filename in ("optimizer.pt", "scheduler.pt"):
        _assert_nested_state_equal(
            torch.load(expected / state_filename, map_location="cpu", weights_only=True),
            torch.load(actual / state_filename, map_location="cpu", weights_only=True),
        )
    expected_rng = sorted(expected.glob("rng_state*.pth"))
    actual_rng = sorted(actual.glob("rng_state*.pth"))
    assert [path.name for path in expected_rng] == [path.name for path in actual_rng]
    for expected_path, actual_path in zip(expected_rng, actual_rng, strict=True):
        _assert_nested_state_equal(
            torch.load(expected_path, map_location="cpu", weights_only=False),
            torch.load(actual_path, map_location="cpu", weights_only=False),
        )
    expected_bounded = _bounded_callback_payload(expected)
    actual_bounded = _bounded_callback_payload(actual)
    assert expected_bounded == actual_bounded


def _bounded_callback_payload(checkpoint: Path) -> dict:
    trainer_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    return trainer_state["stateful_callbacks"][BOUNDED_BATCHING_CALLBACK_NAME]


def _assert_nested_state_equal(expected, actual) -> None:
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
            _assert_nested_state_equal(expected[key], actual[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual)
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_state_equal(expected_item, actual_item)
        return
    assert expected == actual
