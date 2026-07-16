from __future__ import annotations

from pathlib import Path

import pytest
import torch

from shaft.training.topology import validate_training_topology
from tests.support.pipeline import write_sft_pipeline_config as _write_config


pytestmark = pytest.mark.component


def test_training_topology_rejects_single_process_data_parallel(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.train.use_cpu = False
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    try:
        validate_training_topology(config)
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("validate_training_topology should reject single-process multi-GPU training")

    assert "torch.nn.DataParallel" in message
    assert "CUDA_VISIBLE_DEVICES=1" in message


def test_training_topology_allows_distributed_launch(monkeypatch, tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.use_cpu = False
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    validate_training_topology(config)
