from __future__ import annotations

import errno
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
from pathlib import Path
import time

import numpy as np
import pytest
from safetensors.torch import load_file
import torch
import yaml

from shaft.config import load_config
from shaft.observability import (
    PROGRESS_SNAPSHOT_FILENAME,
    TRAINING_EFFICIENCY_FILENAME,
)
from shaft.training.batch_planning import (
    BATCH_PLANNING_CALLBACK_NAME,
    BATCHING_RUN_METADATA_FILENAME,
    checkpoint_has_batch_planning_state,
)
from shaft.training.checkpointing import (
    ShaftCheckpointProtocol,
    resolve_resume_checkpoint,
    validate_training_checkpoint_commit,
)
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


def _reserve_loopback_port() -> int:
    for _ in range(128):
        port = 20_000 + secrets.randbelow(40_000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind(("127.0.0.1", port))
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    continue
                raise
        return port
    raise RuntimeError("Could not reserve an unused high loopback port after 128 attempts.")


def _logs_contain(log_paths: list[Path], markers: tuple[str, ...]) -> bool:
    for log_path in log_paths:
        try:
            content = log_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        if any(marker in content for marker in markers):
            return True
    return False


def _terminate_process_groups(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    terminate_deadline = time.monotonic() + 2.0
    for process in processes:
        remaining = max(0.0, terminate_deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
    kill_deadline = time.monotonic() + 2.0
    unreaped: list[int] = []
    for process in processes:
        remaining = max(0.0, kill_deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            unreaped.append(process.pid)
    if unreaped:
        raise RuntimeError(f"Failed to reap two-node torchrun agents after SIGKILL: {unreaped}.")


def _run_two_node_torchrun(
    repo_root: Path,
    config_path: Path,
    topology_path: Path,
    *,
    processes_per_node: int = 1,
    timeout: float = 300,
) -> subprocess.CompletedProcess[str]:
    if (
        not isinstance(processes_per_node, int)
        or isinstance(processes_per_node, bool)
        or processes_per_node < 1
    ):
        raise ValueError("processes_per_node must be a positive integer.")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    last_result: subprocess.CompletedProcess[str] | None = None
    overall_deadline = time.monotonic() + float(timeout)
    port_conflict_markers = ("Address already in use", "EADDRINUSE")
    for attempt in range(3):
        if time.monotonic() >= overall_deadline:
            raise subprocess.TimeoutExpired("two-node torchrun", timeout)
        port = _reserve_loopback_port()
        log_paths = [
            topology_path.parent / f"multinode-agent-{node_rank}-attempt-{attempt}.log"
            for node_rank in range(2)
        ]
        processes: list[subprocess.Popen[str]] = []
        handles = []
        timed_out = False
        try:
            for node_rank, log_path in enumerate(log_paths):
                handle = log_path.open("w", encoding="utf-8")
                handles.append(handle)
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "torch.distributed.run",
                            "--nnodes=2",
                            f"--nproc-per-node={processes_per_node}",
                            f"--node-rank={node_rank}",
                            "--master-addr=127.0.0.1",
                            f"--master-port={port}",
                            "tests/support/distributed_multinode_train.py",
                            str(config_path),
                            str(topology_path),
                        ],
                        cwd=repo_root,
                        env=env,
                        text=True,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
            while any(process.poll() is None for process in processes):
                if _logs_contain(log_paths, port_conflict_markers):
                    _terminate_process_groups(processes)
                    break
                if any(
                    process.poll() not in {None, 0}
                    for process in processes
                ):
                    _terminate_process_groups(processes)
                    break
                if time.monotonic() >= overall_deadline:
                    timed_out = True
                    _terminate_process_groups(processes)
                    break
                time.sleep(0.05)
        finally:
            try:
                _terminate_process_groups(processes)
            finally:
                for handle in handles:
                    handle.close()

        output = "\n".join(
            f"===== logical node {node_rank} =====\n{path.read_text(encoding='utf-8')}"
            for node_rank, path in enumerate(log_paths)
        )
        if timed_out:
            raise subprocess.TimeoutExpired(
                "two-node torchrun",
                timeout,
                output=output,
            )
        if any(process.returncode is None for process in processes):
            raise RuntimeError("Two-node torchrun launcher returned before every agent was reaped.")
        return_codes = [int(process.returncode) for process in processes]
        return_code = next((code for code in return_codes if code != 0), 0)
        last_result = subprocess.CompletedProcess(
            args=["two-node-torchrun"],
            returncode=return_code,
            stdout=output,
            stderr="",
        )
        if return_code == 0 or not any(marker in output for marker in port_conflict_markers):
            return last_result
    assert last_result is not None
    return last_result


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


def _run_training_contract_drift(
    repo_root: Path,
    output_dir: Path,
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
            "tests/support/distributed_training_contract_drift.py",
            str(output_dir),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _run_resume_generation_drift(
    repo_root: Path,
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
            "tests/support/distributed_resume_generation_drift.py",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _run_interceptor_fault(
    repo_root: Path,
    output_dir: Path,
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
            "tests/support/distributed_interceptor_fault.py",
            str(output_dir),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _run_neutral_hook_fault(
    repo_root: Path,
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
            "tests/support/distributed_neutral_hook_fault.py",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _run_rlhf_trainer_prepare_fault(
    repo_root: Path,
    output_dir: Path,
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
            "tests/support/distributed_rlhf_trainer_prepare_fault.py",
            str(output_dir),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _run_finalization_fault(
    repo_root: Path,
    output_dir: Path,
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
            "tests/support/distributed_finalization_fault.py",
            str(output_dir),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _run_pipeline_finalization_fault(
    repo_root: Path,
    config_path: Path,
    mode: str,
    *,
    timeout: int = 120,
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
            "tests/support/distributed_pipeline_finalization_fault.py",
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


def test_two_node_launcher_exhausts_port_conflicts_without_false_success(
    tmp_path: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        distributed=True,
        train_size=4,
        val_size=2,
        train_steps=1,
    )
    topology_path = tmp_path / "exhausted-port-topology.json"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied_listener:
        occupied_listener.bind(("127.0.0.1", 0))
        occupied_listener.listen(8)
        occupied_port = int(occupied_listener.getsockname()[1])
        reserve_calls = 0

        def reserve_occupied_port() -> int:
            nonlocal reserve_calls
            reserve_calls += 1
            return occupied_port

        monkeypatch.setattr(
            sys.modules[__name__],
            "_reserve_loopback_port",
            reserve_occupied_port,
        )
        completed = _run_two_node_torchrun(
            repo_root,
            config_path,
            topology_path,
            timeout=30,
        )

    assert reserve_calls == 3
    assert completed.returncode != 0
    assert any(
        marker in completed.stdout
        for marker in ("Address already in use", "EADDRINUSE")
    )
    assert not topology_path.exists()


def test_two_node_launcher_execution_deadline_reaps_agents(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        distributed=True,
        train_size=4,
        val_size=2,
        train_steps=1,
    )
    topology_path = tmp_path / "deadline-topology.json"
    started_at = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        _run_two_node_torchrun(
            repo_root,
            config_path,
            topology_path,
            timeout=0.25,
        )

    assert time.monotonic() - started_at < 8.0
    assert "logical node" in str(exc_info.value.output)
    assert not topology_path.exists()


def test_two_logical_nodes_two_local_workers_train_save_and_exact_resume(
    tmp_path: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        distributed=True,
        bounded_cost_grouping=True,
        output_name="uninterrupted",
        train_size=20,
        val_size=2,
        train_steps=2,
        save_steps=1,
        save_total_limit=2,
    )
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["train"]["full_determinism"] = True
    config_payload["train"]["distributed"] = {
        "strategy": "ddp",
        "ddp": {"static_graph": True},
    }
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    topology_path = tmp_path / "two-node-fresh-topology.json"

    original_reserve_loopback_port = _reserve_loopback_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied_listener:
        occupied_listener.bind(("127.0.0.1", 0))
        occupied_listener.listen()
        occupied_port = int(occupied_listener.getsockname()[1])
        reserve_calls = 0

        def reserve_with_one_collision() -> int:
            nonlocal reserve_calls
            reserve_calls += 1
            if reserve_calls == 1:
                return occupied_port
            return original_reserve_loopback_port()

        monkeypatch.setattr(
            sys.modules[__name__],
            "_reserve_loopback_port",
            reserve_with_one_collision,
        )
        completed = _run_two_node_torchrun(
            repo_root,
            config_path,
            topology_path,
            processes_per_node=2,
        )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_reserve_loopback_port",
        original_reserve_loopback_port,
    )

    _assert_torchrun_succeeded(completed)
    assert reserve_calls == 2
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    assert topology == [
        {
            "group_rank": 0,
            "local_rank": 0,
            "local_world_size": 2,
            "rank": 0,
            "world_size": 4,
        },
        {
            "group_rank": 0,
            "local_rank": 1,
            "local_world_size": 2,
            "rank": 1,
            "world_size": 4,
        },
        {
            "group_rank": 1,
            "local_rank": 0,
            "local_world_size": 2,
            "rank": 2,
            "world_size": 4,
        },
        {
            "group_rank": 1,
            "local_rank": 1,
            "local_world_size": 2,
            "rank": 3,
            "world_size": 4,
        },
    ]
    uninterrupted_checkpoint = tmp_path / "uninterrupted" / "checkpoint-2"
    commit = validate_training_checkpoint_commit(uninterrupted_checkpoint)
    assert commit["global_step"] == 2
    assert sorted(
        path.name for path in uninterrupted_checkpoint.glob("rng_state*.pth")
    ) == [
        "rng_state_0.pth",
        "rng_state_1.pth",
        "rng_state_2.pth",
        "rng_state_3.pth",
    ]

    resume_output = tmp_path / "resumed"
    resume_config = tmp_path / "two-node-resume.yaml"
    resume_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resume_payload["experiment"]["output_dir"] = str(resume_output)
    resume_payload["train"]["resume_from_checkpoint"] = str(
        tmp_path / "uninterrupted" / "checkpoint-1"
    )
    resume_config.write_text(
        yaml.safe_dump(resume_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    resume_topology_path = tmp_path / "two-node-resume-topology.json"

    resumed = _run_two_node_torchrun(
        repo_root,
        resume_config,
        resume_topology_path,
        processes_per_node=2,
    )

    _assert_torchrun_succeeded(resumed)
    assert json.loads(resume_topology_path.read_text(encoding="utf-8")) == topology
    resumed_checkpoint = resume_output / "checkpoint-2"
    resumed_commit = validate_training_checkpoint_commit(resumed_checkpoint)
    assert resumed_commit["global_step"] == 2
    _assert_exact_checkpoint(
        uninterrupted_checkpoint,
        resumed_checkpoint,
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


def test_training_contract_convergence_rejects_rank_drift_without_hanging(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    output_dir = tmp_path / "training-contract-drift"
    completed = _run_training_contract_drift(repo_root, output_dir)
    _assert_torchrun_succeeded(completed)

    assert (output_dir / "contract_drift_rejected.txt").read_text(encoding="utf-8") == "ok\n"
    assert (output_dir / "builder_failure_rejected.txt").read_text(encoding="utf-8") == "ok\n"
    assert (output_dir / "pipeline_setup_failure_rejected.txt").read_text(
        encoding="utf-8"
    ) == "ok\n"
    assert (output_dir / "collective_owner_boundaries_verified.txt").read_text(
        encoding="utf-8"
    ) == "ok\n"
    assert (output_dir / "model_build_phase_failures_converged.txt").read_text(
        encoding="utf-8"
    ) == "ok\n"


def test_resume_generation_drift_is_rejected_before_training(
    repo_root: Path,
) -> None:
    completed = _run_resume_generation_drift(repo_root)
    _assert_torchrun_succeeded(completed)

    assert "same-step resume generation drift rejected" in (
        f"{completed.stdout}\n{completed.stderr}"
    )


def test_pipeline_before_interceptor_failure_converges_before_body(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    output_dir = tmp_path / "interceptor-fault"
    completed = _run_interceptor_fault(repo_root, output_dir)
    _assert_torchrun_succeeded(completed)

    assert (output_dir / "before_interceptor_failure_rejected.txt").read_text(
        encoding="utf-8"
    ) == "ok\n"
    assert (output_dir / "same_name_schedule_drift_rejected.txt").read_text(
        encoding="utf-8"
    ) == "ok\n"


def test_rlhf_trainer_prepare_failures_converge_before_constructor(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    output_dir = tmp_path / "rlhf-trainer-prepare-fault"
    completed = _run_rlhf_trainer_prepare_fault(repo_root, output_dir)
    _assert_torchrun_succeeded(completed)

    assert (output_dir / "rlhf_trainer_boundaries_verified.txt").read_text(
        encoding="utf-8"
    ) == "ok\n"


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
        schedule_mixing=mixing,
        schedule_shuffle=(mixing == "weighted"),
        secondary_train_size=(12 if mixing == "weighted" else 0),
        secondary_weight=3.0,
        num_workers=2,
        persistent_workers=True,
        per_device_train_batch_size=batch_cap,
        gradient_accumulation_steps=2,
        train_steps=3,
        save_steps=1,
        save_total_limit=3,
    )
    parsed_config = load_config(config_path)
    assert parsed_config.data.schedule.mixing == mixing
    assert parsed_config.data.schedule.shuffle is (mixing == "weighted")
    assert len(parsed_config.data.datasets) == (2 if mixing == "weighted" else 1)
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
    run_metadata = json.loads(
        (tmp_path / "uninterrupted" / BATCHING_RUN_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    source_weights = dict(run_metadata["source_weights"])
    assert set(source_weights) == (
        {"smoke_ds", "smoke_secondary"} if mixing == "weighted" else {"smoke_ds"}
    )
    checkpoint_one = tmp_path / "uninterrupted" / "checkpoint-1"
    expected_final = tmp_path / "uninterrupted" / "checkpoint-3"
    assert _planning_callback_payload(checkpoint_one)["args"]["spec"]["cardinality"] == cardinality

    resumed_output = tmp_path / "resumed"
    resume_config = tmp_path / "resume.yaml"
    resume_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resume_payload["experiment"]["output_dir"] = str(resumed_output)
    resume_config.write_text(
        yaml.safe_dump(resume_payload, sort_keys=False, allow_unicode=True),
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
        ("cost_drift", "batch-planning-startup contract differs across ranks"),
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
        (
            "checkpoint_peer_on_save_failure",
            "synthetic peer-rank on_save callback failure",
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
    if fault_mode == "checkpoint_peer_on_save_failure":
        assert not list(run_dir.glob("unexpected_post_failure_collective_rank*.txt"))
    assert checkpoint_has_batch_planning_state(checkpoint_one)
    assert resolve_resume_checkpoint(
        run_dir,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        require_planning_state=True,
    ) == str(checkpoint_one)


def test_torchrun_checkpoint_rejects_rank_local_callback_before_collective(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        distributed=True,
        bounded_cost_grouping=True,
        train_size=12,
        val_size=1,
        train_steps=1,
        save_steps=1,
    )

    completed = _run_bounded_fault(
        repo_root,
        config_path,
        "checkpoint_rank_local_callback_schedule",
        timeout=60,
    )
    output = f"{completed.stdout}\n{completed.stderr}"

    assert completed.returncode != 0
    assert "identical ordered on_save callback schedules" in output
    assert not list((tmp_path / "outputs").glob("unexpected_rank_local_collective_rank*.txt"))


def test_torchrun_neutral_hook_failure_isolated_before_peer_collectives(
    repo_root: Path,
) -> None:
    completed = _run_neutral_hook_fault(repo_root)
    _assert_torchrun_succeeded(completed)

    output = f"{completed.stdout}\n{completed.stderr}"
    assert "neutral hook distributed isolation ok" in output
    assert "synthetic rank-local neutral before_step failure" in output
    assert "synthetic rank-local neutral after_step failure" in output
    assert "synthetic rank-local neutral on_save failure" in output


def test_torchrun_training_finalization_faults_converge_without_hanging(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    completed = _run_finalization_fault(repo_root, tmp_path / "finalization")
    _assert_torchrun_succeeded(completed)

    output = f"{completed.stdout}\n{completed.stderr}"
    assert "distributed training finalization convergence ok" in output
    assert "final export path drift rejected before side effects" in output


@pytest.mark.parametrize("mode", ["ensure", "prune"])
def test_torchrun_pipeline_rank_zero_finalization_fault_converges(
    tmp_path: Path,
    repo_root: Path,
    mode: str,
) -> None:
    config_path = write_sft_smoke_config(
        tmp_path,
        distributed=True,
        train_size=4,
        val_size=1,
        train_steps=1,
    )
    completed = _run_pipeline_finalization_fault(repo_root, config_path, mode)
    _assert_torchrun_succeeded(completed)

    output = f"{completed.stdout}\n{completed.stderr}"
    assert f"pipeline finalization {mode} convergence ok" in output


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
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
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
    _assert_resume_discrete_state_equal(expected, actual)


def _assert_resume_discrete_state_equal(expected: Path, actual: Path) -> None:
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
