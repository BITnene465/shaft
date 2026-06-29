from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.artifacts import BenchmarkArtifacts, CampaignArtifacts, SuiteArtifacts
from eval_bench.schema import (
    BenchmarkManifest,
    CampaignManifest,
    EvalSuiteManifest,
    SuiteTaskSplit,
)


def test_official_suite_manifest_requires_official_benchmark_type(tmp_path: Path) -> None:
    benchmark = BenchmarkManifest(
        benchmark_id="banana_v2_4_val",
        benchmark_type="official",
        tasks=["detection"],
        root=str(tmp_path / "benchmarks" / "banana_v2_4_val" / "data"),
        split="suite",
        manifest_path=str(tmp_path / "benchmarks" / "banana_v2_4_val" / "splits" / "suite.txt"),
        sample_count=400,
    )
    BenchmarkArtifacts(tmp_path, benchmark.benchmark_id).write_manifest(benchmark)

    suite = EvalSuiteManifest(
        suite_id="banana_v2_4_val",
        version="v2.4",
        benchmark_id="banana_v2_4_val",
        benchmark_type="official",
        official=True,
        metric_profile="detection_iou_v1",
        sample_universe={"sample_count": 400},
        task_splits=[_suite_task_split(tmp_path)],
    )
    suite_path = SuiteArtifacts(tmp_path, suite.suite_id).write_manifest(suite)
    assert json.loads(suite_path.read_text(encoding="utf-8"))["official"] is True

    with pytest.raises(ValueError, match="official suites"):
        EvalSuiteManifest(
            suite_id="bad_suite",
            version="v1",
            benchmark_type="temporary",
            official=True,
            metric_profile="detection_iou_v1",
            sample_universe={},
            task_splits=suite.task_splits,
        ).validate()


def test_benchmark_manifest_rejects_unsupported_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported benchmark_type"):
        BenchmarkManifest(
            benchmark_id="not_a_benchmark",
            benchmark_type="prelabel",  # type: ignore[arg-type]
            tasks=["detection"],
            root=str(tmp_path / "benchmarks" / "not_a_benchmark" / "data"),
            split="val",
            manifest_path=str(tmp_path / "benchmarks" / "not_a_benchmark" / "splits" / "val.txt"),
            sample_count=0,
        ).validate()


def test_suite_manifest_rejects_duplicate_task_split(tmp_path: Path) -> None:
    split = _suite_task_split(tmp_path)
    with pytest.raises(ValueError, match="duplicate task split"):
        EvalSuiteManifest(
            suite_id="duplicate_suite",
            version="v1",
            benchmark_type="official",
            official=True,
            metric_profile="detection_iou_v1",
            sample_universe={"sample_count": 400},
            task_splits=[split, split],
        ).validate()


def test_campaign_manifest_writes_suite_reference(tmp_path: Path) -> None:
    campaign = CampaignManifest(
        campaign_id="model_a__banana_v2_4_val",
        suite_id="banana_v2_4_val",
        model_id="model-a",
        checkpoint="outputs/model-a/ckpt-100",
        prompt_set=["grounding_layout.main"],
        pixel_budget=2_000_000,
        decoding_config={"max_tokens": 2048, "temperature": 0.0},
        run_ids=["layout_run"],
        aggregate_report={"f1_iou50": 0.8},
    )
    campaign_path = CampaignArtifacts(tmp_path, campaign.campaign_id).write_manifest(campaign)
    assert json.loads(campaign_path.read_text(encoding="utf-8"))["suite_id"] == campaign.suite_id


def _suite_task_split(tmp_path: Path) -> SuiteTaskSplit:
    return SuiteTaskSplit(
        split="grounding_layout",
        benchmark_id="banana_v2_4_val",
        manifest_path=str(tmp_path / "benchmarks" / "banana_v2_4_val" / "splits" / "layout.txt"),
        sample_count=200,
        tasks=["detection"],
        layers=["layout"],
        target_labels=["icon"],
    )
