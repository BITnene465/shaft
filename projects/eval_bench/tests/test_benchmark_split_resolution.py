from __future__ import annotations

from pathlib import Path

import pytest

from eval_bench.benchmark import resolve_benchmark_split_name, resolve_benchmark_split_path


def test_suite_split_resolution_requires_declared_top_level_split(tmp_path: Path) -> None:
    payload = {
        "split": "suite",
        "manifest_path": str(tmp_path / "splits" / "grounding_arrow.txt"),
        "split_manifests": {
            "grounding_arrow": str(tmp_path / "splits" / "grounding_arrow.txt"),
        },
    }

    with pytest.raises(FileNotFoundError, match="benchmark split 'suite' is not available"):
        resolve_benchmark_split_path(payload, split="suite")


def test_suite_split_name_resolution_rejects_missing_default_split(tmp_path: Path) -> None:
    payload = {
        "split": "suite",
        "manifest_path": str(tmp_path / "splits" / "grounding_arrow.txt"),
        "split_manifests": {
            "grounding_arrow": str(tmp_path / "splits" / "grounding_arrow.txt"),
        },
    }

    with pytest.raises(FileNotFoundError, match="benchmark default split 'suite' is not available"):
        resolve_benchmark_split_name(payload, task="detection", prompt_id="custom")


def test_historical_arrow_keypoint_prompt_uses_point_arrow_crop_split(tmp_path: Path) -> None:
    payload = {
        "split": "suite",
        "manifest_path": str(tmp_path / "splits" / "suite.txt"),
        "split_manifests": {
            "suite": str(tmp_path / "splits" / "suite.txt"),
            "point_arrow": str(tmp_path / "splits" / "point_arrow.txt"),
        },
    }

    split = resolve_benchmark_split_name(
        payload,
        task="detection",
        prompt_id="arrow_keypoint.v2.4.main",
        target_labels=["arrow"],
    )

    assert split == "point_arrow"
