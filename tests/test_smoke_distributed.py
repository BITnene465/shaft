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

from shaft.observability import (
    PROGRESS_SNAPSHOT_FILENAME,
    TRAINING_EFFICIENCY_FILENAME,
)
from shaft.training.batch_planning import (
    BATCH_PLANNING_CALLBACK_NAME,
    BATCHING_RUN_METADATA_FILENAME,
    checkpoint_has_batch_planning_state,
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
    raise AssertionError(f"torchrun failed (code={completed.returncode}).\n{output}")


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


def _run_efficiency_snapshot_fault(
    repo_root: Path,
    output_dir: Path,
    mode: str,
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
            "tests/support/distributed_efficiency_snapshot_fault.py",
            str(output_dir),
            mode,
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
        (tmp_path / "outputs" / PROGRESS_SNAPSHOT_FILENAME).read_text(encoding="utf-8")
    )
    assert progress["status"] == "succeeded"
    assert progress["tasks"]["train"]["current"] == 1
    efficiency = json.loads(
        (tmp_path / "outputs" / TRAINING_EFFICIENCY_FILENAME).read_text(encoding="utf-8")
    )
    assert efficiency["world_size"] == 2
    assert efficiency["final_global_step"] == 1
    assert efficiency["aggregate"]["optimizer_steps"] == 1
    assert efficiency["aggregate"]["useful_tokens"] > 0
    assert (
        efficiency["aggregate"]["materialized_tokens"] >= efficiency["aggregate"]["useful_tokens"]
    )


@pytest.mark.parametrize("mode", ["missing", "corrupt"])
def test_efficiency_resume_discards_asymmetric_rank_snapshots_without_hanging(
    tmp_path: Path,
    repo_root: Path,
    mode: str,
) -> None:
    output_dir = tmp_path / mode
    completed = _run_efficiency_snapshot_fault(repo_root, output_dir, mode)
    _assert_torchrun_succeeded(completed)

    result = json.loads((output_dir / "fault_result.json").read_text(encoding="utf-8"))
    assert result["complete_history"] is False
    assert result["initial_global_step"] == 1
    assert result["final_global_step"] == 2
    assert result["aggregate"]["optimizer_steps"] == 1
    assert result["aggregate"]["useful_tokens"] == 9


def test_efficiency_aggregation_rejects_rank_divergent_optimizer_updates(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    output_dir = tmp_path / "update-mismatch"
    completed = _run_efficiency_snapshot_fault(
        repo_root,
        output_dir,
        "update_mismatch",
    )
    _assert_torchrun_succeeded(completed)

    assert (output_dir / "mismatch_rejected.txt").read_text(encoding="utf-8") == "ok\n"


def test_efficiency_aggregation_rejects_rank_divergent_cuda_timing_coverage(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    output_dir = tmp_path / "timing-mismatch"
    completed = _run_efficiency_snapshot_fault(
        repo_root,
        output_dir,
        "timing_mismatch",
    )
    _assert_torchrun_succeeded(completed)

    assert (output_dir / "timing_mismatch_rejected.txt").read_text(encoding="utf-8") == "ok\n"


def test_efficiency_peak_memory_uses_max_across_ranks(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    output_dir = tmp_path / "peak-memory-max"
    completed = _run_efficiency_snapshot_fault(
        repo_root,
        output_dir,
        "peak_memory_max",
    )
    _assert_torchrun_succeeded(completed)

    result = json.loads((output_dir / "peak_memory_max.json").read_text(encoding="utf-8"))
    assert result["peak_device_memory_allocated_bytes"] == 3 * 1024**3
    assert result["peak_device_memory_reserved_bytes"] == 4 * 1024**3


def test_efficiency_peak_memory_is_unavailable_if_any_rank_is_unavailable(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    output_dir = tmp_path / "peak-memory-unavailable"
    completed = _run_efficiency_snapshot_fault(
        repo_root,
        output_dir,
        "peak_memory_unavailable",
    )
    _assert_torchrun_succeeded(completed)

    result = json.loads((output_dir / "peak_memory_unavailable.json").read_text(encoding="utf-8"))
    assert result["peak_device_memory_allocated_bytes"] is None
    assert result["peak_device_memory_reserved_bytes"] is None


def test_efficiency_monitor_rejects_rank_divergent_contracts_without_hanging(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    output_dir = tmp_path / "contract-mismatch"
    completed = _run_efficiency_snapshot_fault(
        repo_root,
        output_dir,
        "contract_mismatch",
    )
    _assert_torchrun_succeeded(completed)

    assert (output_dir / "contract_mismatch_rejected.txt").read_text(encoding="utf-8") == "ok\n"


@pytest.mark.parametrize(
    "mode",
    [
        "revoke_fail",
        "snapshot_write_fail",
        "manifest_write_fail",
        "transaction_commit_fail",
    ],
)
def test_efficiency_snapshot_commit_converges_rank_local_io_failures(
    tmp_path: Path,
    repo_root: Path,
    mode: str,
) -> None:
    output_dir = tmp_path / mode
    completed = _run_efficiency_snapshot_fault(repo_root, output_dir, mode)
    _assert_torchrun_succeeded(completed)

    assert (output_dir / f"{mode}_rejected.txt").read_text(encoding="utf-8") == "ok\n"


def test_efficiency_summary_write_converges_rank_zero_io_failure(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    output_dir = tmp_path / "summary-write-fail"
    completed = _run_efficiency_snapshot_fault(
        repo_root,
        output_dir,
        "summary_write_fail",
    )
    _assert_torchrun_succeeded(completed)

    assert (output_dir / "summary_write_fail_rejected.txt").read_text(encoding="utf-8") == "ok\n"


def test_torchrun_interactive_progress_keeps_one_rank_zero_console_line(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    completed = _run_progress_console(repo_root, tmp_path / "sync")
    _assert_torchrun_succeeded(completed)

    output = f"{completed.stdout}\n{completed.stderr}"
    assert output.count("rank-zero-warning") == 1
    assert "rank-one-warning-must-be-hidden" not in output
    frames = [line.rstrip() for line in output.splitlines() if line.startswith("train ")]
    assert any("0%" in frame and "0/10k" in frame for frame in frames)
    assert any("0.01%" in frame and "1/10k" in frame for frame in frames)
    assert any("━" in frame or "╸" in frame for frame in frames)
    assert "▏" not in output and "·" not in output
    warning_end = output.index("rank-zero-warning") + len("rank-zero-warning")
    assert output[warning_end:].lstrip().startswith("train")


@pytest.mark.parametrize("per_device_batch_size", [1, 2])
def test_torchrun_bounded_fixed_batch_contract(
    tmp_path: Path,
    repo_root: Path,
    per_device_batch_size: int,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=20,
        val_size=2,
        distributed=True,
        bounded_cost_grouping=True,
        bounded_max_tokens_per_microbatch=1024,
        per_device_train_batch_size=per_device_batch_size,
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
    assert "[batch-contract]" in output
    assert "[batch-plan]" in output
    assert "[batch-plan-summary]" in output
    metadata = json.loads(
        (tmp_path / "outputs" / BATCHING_RUN_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert metadata["grouping"] == "bounded_cost"
    assert metadata["cardinality"] == "fixed"
    assert metadata["packing"] == "none"
    assert metadata["layout"] == "padded"
    assert metadata["per_device_train_batch_size"] == per_device_batch_size
    assert metadata["global_pack_count"] == 2 * per_device_batch_size
    assert metadata["optimizer_pack_count"] == 4 * per_device_batch_size
    assert metadata["buffer_size"] == 8
    assert _planning_callback_payload(tmp_path / "outputs" / "checkpoint-1")
    assert not list((tmp_path / "outputs").glob("*cost_plan*"))


def test_torchrun_bounded_token_budget_uses_variable_local_batches(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=20,
        val_size=2,
        distributed=True,
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

    completed = _run_torchrun(repo_root, config_path)
    _assert_torchrun_succeeded(completed)

    output = f"{completed.stdout}\n{completed.stderr}"
    assert "cardinality=token_budget" in output
    assert "first_logical_segments=3" in output
    assert "first_physical_packs=3" in output
    metadata = json.loads(
        (tmp_path / "outputs" / BATCHING_RUN_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert metadata["cardinality"] == "token_budget"
    assert metadata["local_pack_count_range"] == [1, 2]
    assert metadata["global_pack_count_range"] == [2, 4]
    assert metadata["optimizer_pack_count_range"] == [4, 8]
    assert metadata["global_pack_count"] is None
    checkpoint = tmp_path / "outputs" / "checkpoint-1"
    callback = _planning_callback_payload(checkpoint)
    state = callback["attributes"]["planning_state"]
    assert state["global_microstep"] == 2
    assert 4 <= state["emitted_samples"] <= 8


@pytest.mark.parametrize(
    ("cardinality", "batch_cap", "token_cap", "mixing"),
    [
        ("fixed", 1, 512, "concat"),
        ("token_budget", 2, 450, "concat"),
        ("fixed", 1, 512, "weighted"),
    ],
)
def test_torchrun_exact_resume(
    tmp_path: Path,
    repo_root: Path,
    cardinality: str,
    batch_cap: int,
    token_cap: int,
    mixing: str,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        output_name="uninterrupted",
        train_size=36,
        val_size=3,
        distributed=True,
        bounded_cost_grouping=True,
        bounded_cardinality=cardinality,
        bounded_max_tokens_per_microbatch=token_cap,
        per_device_train_batch_size=batch_cap,
        gradient_accumulation_steps=2,
        train_steps=3,
        save_steps=1,
    )
    if mixing == "weighted":
        config_path.write_text(
            config_path.read_text(encoding="utf-8")
            .replace("    mixing: concat", "    mixing: weighted", 1)
            .replace("    shuffle: false", "    shuffle: true", 1),
            encoding="utf-8",
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
    uninterrupted_logs = f"{uninterrupted.stdout}\n{uninterrupted.stderr}"
    assert "Num Epochs =" not in uninterrupted_logs
    checkpoint_one = tmp_path / "uninterrupted" / "checkpoint-1"
    expected_final = tmp_path / "uninterrupted" / "checkpoint-3"
    assert _planning_callback_payload(checkpoint_one)["args"]["spec"]["cardinality"] == cardinality

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
    resumed_logs = f"{resumed.stdout}\n{resumed.stderr}"
    assert "[train-resume] global_step=1/3" in resumed_logs
    assert "Resuming training from checkpoint with epoch" not in resumed_logs

    _assert_exact_checkpoint(expected_final, resumed_output / "checkpoint-3")
    uninterrupted_efficiency = json.loads(
        (tmp_path / "uninterrupted" / TRAINING_EFFICIENCY_FILENAME).read_text(encoding="utf-8")
    )
    resumed_efficiency = json.loads(
        (resumed_output / TRAINING_EFFICIENCY_FILENAME).read_text(encoding="utf-8")
    )
    assert resumed_efficiency["complete_history"] is True
    assert resumed_efficiency["aggregate"]["optimizer_steps"] == 3
    assert (
        resumed_efficiency["aggregate"]["useful_tokens"]
        == uninterrupted_efficiency["aggregate"]["useful_tokens"]
    )


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
        bounded_cost_grouping=True,
        train_size=12,
        val_size=1,
        train_steps=1,
    )

    completed = _run_bounded_fault(repo_root, config_path, mode)
    output = f"{completed.stdout}\n{completed.stderr}"

    assert completed.returncode != 0
    assert message.lower() in output.lower()


@pytest.mark.parametrize(
    ("fault_mode", "message"),
    [
        (
            "checkpoint_write_failure",
            "synthetic bounded trainer-state write failure",
        ),
        (
            "checkpoint_peer_rng_failure",
            "synthetic peer-rank RNG-state write failure",
        ),
    ],
)
def test_torchrun_bounded_checkpoint_failure_preserves_last_complete_resume(
    tmp_path: Path,
    repo_root: Path,
    fault_mode: str,
    message: str,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        distributed=True,
        bounded_cost_grouping=True,
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
        fault_mode,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    run_dir = tmp_path / "outputs"
    checkpoint_one = run_dir / "checkpoint-1"

    assert completed.returncode != 0
    assert message in output
    assert checkpoint_has_batch_planning_state(checkpoint_one)
    assert resolve_resume_checkpoint(
        run_dir,
        require_planning_state=True,
    ) == str(checkpoint_one)


def test_torchrun_bounded_checkpoint_rewrite_revokes_stale_completion(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        distributed=True,
        bounded_cost_grouping=True,
        train_size=20,
        val_size=1,
        train_steps=2,
        save_steps=1,
    )
    first_run = _run_torchrun(repo_root, config_path)
    _assert_torchrun_succeeded(first_run)

    run_dir = tmp_path / "outputs"
    checkpoint_one = run_dir / "checkpoint-1"
    checkpoint_two = run_dir / "checkpoint-2"
    assert checkpoint_has_batch_planning_state(checkpoint_one)
    assert checkpoint_has_batch_planning_state(checkpoint_two)

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "train:\n",
            f"train:\n  resume_from_checkpoint: {checkpoint_one}\n",
            1,
        ),
        encoding="utf-8",
    )
    failed_rewrite = _run_bounded_fault(
        repo_root,
        config_path,
        "checkpoint_peer_rng_failure",
    )
    output = f"{failed_rewrite.stdout}\n{failed_rewrite.stderr}"

    assert failed_rewrite.returncode != 0
    assert "synthetic peer-rank RNG-state write failure" in output
    assert not checkpoint_has_batch_planning_state(checkpoint_two)
    assert checkpoint_has_batch_planning_state(checkpoint_one)
    assert resolve_resume_checkpoint(
        run_dir,
        require_planning_state=True,
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
    expected_planning = _planning_callback_payload(expected)
    actual_planning = _planning_callback_payload(actual)
    assert expected_planning == actual_planning


def _planning_callback_payload(checkpoint: Path) -> dict:
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    return trainer_state["stateful_callbacks"][BATCH_PLANNING_CALLBACK_NAME]


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
