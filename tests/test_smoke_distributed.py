from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
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


def test_torchrun_dynamic_cost_aware_variable_batch_contract(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=12,
        val_size=3,
        distributed=True,
        dynamic_cost_aware=True,
        dynamic_target_samples=12,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
    )
    train_path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["user_prompt"] = "x" * 300
    rows[6]["user_prompt"] = "y" * 280
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
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
            f"dynamic cost-aware torchrun smoke failed (code={completed.returncode}).\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert "[dynamic-batch-plan-summary]" in combined_output
    assert "selected_samples=12" in combined_output
    assert "local_batch_min=1" in combined_output
    assert "local_batch_max=5" in combined_output
    assert (tmp_path / "outputs" / BATCH_PLANNING_SIGNATURE_FILENAME).is_file()
    assert (tmp_path / "outputs" / COST_PLAN_REFERENCE_FILENAME).is_file()


def test_torchrun_dynamic_cost_aware_exact_resume_on_cpu(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        output_name="uninterrupted",
        train_size=36,
        val_size=3,
        distributed=True,
        dynamic_cost_aware=True,
        dynamic_target_samples=12,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
        train_steps=3,
        save_steps=1,
    )
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8")
        .replace("  num_workers: 0", "  num_workers: 2", 1)
        .replace("  persistent_workers: false", "  persistent_workers: true", 1)
        .replace("  save_total_limit: 2", "  save_total_limit: 3", 1),
        encoding="utf-8",
    )
    train_path = tmp_path / "train.jsonl"
    rows = [
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
    ]
    for index in range(0, 36, 6):
        rows[index]["user_prompt"] = "x" * 300
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"

    def run_torchrun(config_path: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
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
            timeout=300,
            check=False,
        )

    uninterrupted = run_torchrun(cfg_path)
    if uninterrupted.returncode != 0:
        stderr = uninterrupted.stderr or ""
        if "Operation not permitted" in stderr and "RendezvousConnectionError" in stderr:
            pytest.skip("torchrun rendezvous is blocked by current sandbox/network policy.")
        raise AssertionError(
            "uninterrupted dynamic torchrun failed "
            f"(code={uninterrupted.returncode}).\nSTDOUT:\n{uninterrupted.stdout}\n"
            f"STDERR:\n{uninterrupted.stderr}"
        )

    checkpoint_one = tmp_path / "uninterrupted" / "checkpoint-1"
    uninterrupted_checkpoint_three = tmp_path / "uninterrupted" / "checkpoint-3"
    resumed_output = tmp_path / "resumed-from-one"
    resume_cfg = tmp_path / "sft_full_resume_from_one.yaml"
    resume_cfg.write_text(
        cfg_path.read_text(encoding="utf-8").replace(
            f"output_dir: {tmp_path / 'uninterrupted'}",
            f"output_dir: {resumed_output}",
            1,
        ),
        encoding="utf-8",
    )
    resumed = run_torchrun(
        resume_cfg,
        "--resume-from",
        str(checkpoint_one),
    )
    if resumed.returncode != 0:
        raise AssertionError(
            f"resumed dynamic torchrun failed (code={resumed.returncode}).\n"
            f"STDOUT:\n{resumed.stdout}\nSTDERR:\n{resumed.stderr}"
        )

    def assert_exact_checkpoint(actual_checkpoint: Path) -> None:
        expected_model = load_file(
            str(uninterrupted_checkpoint_three / "model.safetensors")
        )
        actual_model = load_file(str(actual_checkpoint / "model.safetensors"))
        assert expected_model.keys() == actual_model.keys()
        for name in expected_model:
            assert torch.equal(expected_model[name], actual_model[name]), name
        for state_filename in ("optimizer.pt", "scheduler.pt"):
            expected_state = torch.load(
                uninterrupted_checkpoint_three / state_filename,
                map_location="cpu",
                weights_only=True,
            )
            actual_state = torch.load(
                actual_checkpoint / state_filename,
                map_location="cpu",
                weights_only=True,
            )
            _assert_nested_state_equal(expected_state, actual_state)
        expected_rng_paths = sorted(
            uninterrupted_checkpoint_three.glob("rng_state*.pth")
        )
        actual_rng_paths = sorted(actual_checkpoint.glob("rng_state*.pth"))
        assert [path.name for path in expected_rng_paths] == [
            path.name for path in actual_rng_paths
        ]
        for expected_path, actual_path in zip(
            expected_rng_paths,
            actual_rng_paths,
            strict=True,
        ):
            _assert_nested_state_equal(
                torch.load(expected_path, map_location="cpu", weights_only=False),
                torch.load(actual_path, map_location="cpu", weights_only=False),
            )
        resumed_state = json.loads(
            (actual_checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
        assert int(resumed_state["global_step"]) == 3

    assert_exact_checkpoint(resumed_output / "checkpoint-3")

    resumed_from_two_output = tmp_path / "resumed-from-two"
    resume_from_two_cfg = tmp_path / "sft_full_resume_from_two.yaml"
    resume_from_two_cfg.write_text(
        cfg_path.read_text(encoding="utf-8").replace(
            f"output_dir: {tmp_path / 'uninterrupted'}",
            f"output_dir: {resumed_from_two_output}",
            1,
        ),
        encoding="utf-8",
    )
    resumed_from_two = run_torchrun(
        resume_from_two_cfg,
        "--resume-from",
        str(resumed_output / "checkpoint-2"),
    )
    if resumed_from_two.returncode != 0:
        raise AssertionError(
            "second-generation resumed dynamic torchrun failed "
            f"(code={resumed_from_two.returncode}).\n"
            f"STDOUT:\n{resumed_from_two.stdout}\nSTDERR:\n{resumed_from_two.stderr}"
        )
    assert not (resumed_from_two_output / "checkpoint-1").exists()
    assert not (resumed_from_two_output / "checkpoint-2").exists()
    assert_exact_checkpoint(resumed_from_two_output / "checkpoint-3")


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
    assert result["rank_batch_sizes"] == [[1, 1], [2, 1]]
    assert result["global_denominator"] == pytest.approx(12.0)
    assert result["reference_loss"] > 0
    assert result["max_parameter_error"] < 1e-7


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
