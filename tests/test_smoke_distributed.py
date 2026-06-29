from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
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
