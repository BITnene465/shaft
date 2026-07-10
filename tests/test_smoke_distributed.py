from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from shaft.data.cost_plan import COST_PLAN_REFERENCE_FILENAME
from shaft.training.batch_planning import BATCH_PLANNING_SIGNATURE_FILENAME
from tests.support.configs import write_sft_smoke_config


pytestmark = pytest.mark.smoke


def test_torchrun_train_eval_smoke(tmp_path: Path, repo_root: Path) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=4,
        val_size=2,
        distributed=True,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "scripts/train.py",
        "sft",
        "--config",
        str(cfg_path),
        "--max-steps",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr or ""
        if "Operation not permitted" in stderr and "RendezvousConnectionError" in stderr:
            pytest.skip("torchrun rendezvous is blocked by current sandbox/network policy.")
        raise AssertionError(
            f"torchrun smoke failed (code={completed.returncode}).\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def test_torchrun_cost_aware_trainer_dataloader_contract(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=8,
        val_size=2,
        distributed=True,
        cost_aware=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "scripts/train.py",
        "sft",
        "--config",
        str(cfg_path),
        "--max-steps",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr or ""
        if "Operation not permitted" in stderr and "RendezvousConnectionError" in stderr:
            pytest.skip("torchrun rendezvous is blocked by current sandbox/network policy.")
        raise AssertionError(
            f"cost-aware torchrun smoke failed (code={completed.returncode}).\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert "[batch-plan-summary]" in combined_output
    assert "[cost-plan-cache]" in combined_output
    assert "samples=8" in combined_output
    assert (tmp_path / "outputs" / BATCH_PLANNING_SIGNATURE_FILENAME).is_file()
    assert (tmp_path / "outputs" / COST_PLAN_REFERENCE_FILENAME).is_file()


def test_torchrun_cost_plan_rank_zero_failure_reaches_all_ranks(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=8,
        val_size=2,
        distributed=True,
        cost_aware=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
    )
    (tmp_path / "image.png").unlink()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "scripts/train.py",
        "sft",
        "--config",
        str(cfg_path),
        "--max-steps",
        "1",
    ]

    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    stderr = completed.stderr or ""
    if "Operation not permitted" in stderr and "RendezvousConnectionError" in stderr:
        pytest.skip("torchrun rendezvous is blocked by current sandbox/network policy.")
    assert completed.returncode != 0
    combined_output = f"{completed.stdout}\n{stderr}"
    assert "Rank-zero CostPlan materialization failed" in combined_output
    assert "image.png" in combined_output
    assert not (tmp_path / "outputs" / COST_PLAN_REFERENCE_FILENAME).exists()


def test_torchrun_cost_plan_nonzero_load_failure_reaches_all_ranks(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=8,
        val_size=2,
        distributed=True,
        cost_aware=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "tests/support/distributed_cost_plan_load_fault.py",
        str(cfg_path),
    ]

    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    stderr = completed.stderr or ""
    if "Operation not permitted" in stderr and "RendezvousConnectionError" in stderr:
        pytest.skip("torchrun rendezvous is blocked by current sandbox/network policy.")
    assert completed.returncode != 0
    combined_output = f"{completed.stdout}\n{stderr}"
    assert "injected nonzero-rank CostPlan mmap failure" in combined_output
    assert "Shared CostPlan is not readable on every rank" in combined_output
    assert not (tmp_path / "outputs" / COST_PLAN_REFERENCE_FILENAME).exists()


def test_torchrun_global_weighted_loss_matches_single_process_reference(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    result_path = tmp_path / "distributed-loss-result.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "tests/support/distributed_loss_probe.py",
        str(result_path),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr or ""
        if "Operation not permitted" in stderr and "RendezvousConnectionError" in stderr:
            pytest.skip("torchrun rendezvous is blocked by current sandbox/network policy.")
        raise AssertionError(
            f"distributed loss probe failed (code={completed.returncode}).\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["global_denominator"] == pytest.approx(10.5)
    assert result["reference_loss"] > 0
    assert result["max_parameter_error"] < 1e-7
