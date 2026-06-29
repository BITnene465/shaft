from __future__ import annotations

import json
from pathlib import Path

from eval_bench.artifacts import RunArtifacts
from eval_bench.comparison import (
    comparison_sample_detail_payload,
    compare_report_payloads,
    compare_runs,
)
from eval_bench.evaluator import evaluate_run
from eval_bench.schema import (
    BenchmarkRef,
    EvalRunManifest,
    EvalSpec,
    ModelRef,
    PredictionDocument,
    PredictionInstance,
    PromptRef,
)
from projects.eval_bench.tests.support.evaluator import write_eval_run
from projects.eval_bench.tests.support.files import write_json


def test_comparison_sample_detail_uses_side_specific_indices(tmp_path: Path) -> None:
    bench_dir = tmp_path / "benchmarks" / "bench1"
    (bench_dir / "splits").mkdir(parents=True)
    (bench_dir / "splits" / "baseline.txt").write_text(
        "part1/json/a.json\npart1/json/b.json\n",
        encoding="utf-8",
    )
    (bench_dir / "splits" / "candidate.txt").write_text(
        "part1/json/b.json\npart1/json/a.json\n",
        encoding="utf-8",
    )
    for stem, label in (("a", "arrow"), ("b", "shape")):
        write_json(
            bench_dir / "data" / "part1" / "json" / f"{stem}.json",
            {
                "image_path": f"part1/images/{stem}.png",
                "instances": [{"label": label, "bbox": [0, 0, 100, 100]}],
            },
        )
    for run_id, split in (("baseline", "baseline"), ("candidate", "candidate")):
        RunArtifacts(tmp_path, run_id).write_manifest(
            EvalRunManifest(
                run_id=run_id,
                model=ModelRef(model_id="model-a", path="outputs/model-a/best"),
                benchmark=BenchmarkRef(
                    benchmark_id="bench1",
                    root=str(bench_dir / "data"),
                    split=split,
                    tasks=["detection"],
                    manifest_path=str(bench_dir / "splits" / f"{split}.txt"),
                ),
                spec=EvalSpec(
                    spec_id="detection.default",
                    task="detection",
                    prompt=PromptRef(prompt_id="grounding_shape.latest"),
                    target_labels=["shape"],
                ),
            )
        )

    detail = comparison_sample_detail_payload(
        store_root=tmp_path,
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        sample_index=0,
        baseline_sample_index=1,
        candidate_sample_index=0,
    )

    assert detail["baseline_index"] == 1
    assert detail["candidate_index"] == 0
    assert detail["baseline"]["sample"]["image"] == "part1/images/b.png"
    assert detail["candidate"]["sample"]["image"] == "part1/images/b.png"


def test_compare_runs_reports_improvements_and_regressions(tmp_path: Path) -> None:
    baseline = write_eval_run(tmp_path, task="detection")
    candidate = RunArtifacts(tmp_path, "run_candidate")
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    candidate.write_manifest(
        EvalRunManifest(
            run_id="run_candidate",
            model=ModelRef(model_id="model-b", path="outputs/model-b/best"),
            benchmark=BenchmarkRef(
                benchmark_id="bench1",
                root=str(tmp_path / "benchmarks" / "bench1" / "data"),
                split="val",
                tasks=["detection", "keypoint"],
                manifest_path=str(split_path),
            ),
            spec=EvalSpec(
                spec_id="detection.default",
                task="detection",
                prompt=PromptRef(prompt_id="detection.prompt"),
            ),
        )
    )
    write_json(
        tmp_path / "benchmarks" / "bench1" / "data" / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 100, 100]},
                {"label": "shape", "bbox": [200, 200, 260, 260]},
            ],
        },
    )
    baseline.write_prediction(
        PredictionDocument(
            image="part1/images/a.png",
            instances=[PredictionInstance(label="icon", bbox=[0, 0, 100, 100])],
            metadata={"producer": "test"},
        ),
        task="detection",
    )
    candidate.write_prediction(
        PredictionDocument(
            image="part1/images/a.png",
            instances=[
                PredictionInstance(label="icon", bbox=[0, 0, 100, 100]),
                PredictionInstance(label="shape", bbox=[200, 200, 260, 260]),
            ],
            metadata={"producer": "test"},
        ),
        task="detection",
    )

    evaluate_run(store_root=tmp_path, run_id="run_detection")
    evaluate_run(store_root=tmp_path, run_id="run_candidate")
    comparison_path = compare_runs(
        store_root=tmp_path,
        baseline_run_id="run_detection",
        candidate_run_id="run_candidate",
    )
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    assert comparison["comparison_id"] == "run_detection__vs__run_candidate"
    assert comparison["benchmark_id"] == "bench1"
    assert comparison["benchmark_split"] == "val"
    assert comparison["delta"]["matched_count"] == 1
    assert comparison["summary"]["improved_samples"] == 1
    assert comparison["summary"]["improved_labels"] == 1
    assert comparison["top_improvements"][0]["sample_index"] == 0
    assert comparison["top_improvements"][0]["candidate_index"] == 0
    assert comparison["top_improvements"][0]["delta"]["false_negative_count"] == -1
    assert "shape" in comparison["top_improvements"][0]["labels"]
    shape_delta = next(item for item in comparison["labels"] if item["label"] == "shape")
    assert shape_delta["delta"]["matched_count"] == 1
    assert shape_delta["delta"]["false_negative_count"] == -1


def test_compare_reports_warns_when_target_label_scope_differs() -> None:
    comparison = compare_report_payloads(
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        baseline={
            "benchmark_id": "bench1",
            "benchmark_split": "val",
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": [],
            "samples": [],
            "labels": [],
        },
        candidate={
            "benchmark_id": "bench1",
            "benchmark_split": "val",
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["arrow"],
            "samples": [],
            "labels": [],
        },
    )

    assert "baseline and candidate target labels differ" in comparison["warnings"]


def test_compare_reports_warns_when_benchmark_scope_differs() -> None:
    comparison = compare_report_payloads(
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        baseline={
            "benchmark_id": "bench-a",
            "benchmark_split": "grounding_arrow",
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["arrow"],
            "samples": [],
            "labels": [],
        },
        candidate={
            "benchmark_id": "bench-b",
            "benchmark_split": "grounding_layout",
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["arrow"],
            "samples": [],
            "labels": [],
        },
    )

    assert "baseline and candidate benchmarks differ" in comparison["warnings"]
    assert "baseline and candidate benchmark splits differ" in comparison["warnings"]


def test_compare_reports_scores_keypoint_distance_as_lower_better() -> None:
    baseline_sample = {
        "index": 0,
        "json_path": "part1/json/a.json",
        "image": "part1/images/a.png",
        "matched_count": 1,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "mean_iou": 1.0,
        "keypoint_pair_count": 1,
        "mean_keypoint_distance": 10.0,
        "labels": {
            "arrow": {
                "matched_count": 1,
                "false_positive_count": 0,
                "false_negative_count": 0,
                "mean_iou": 1.0,
                "keypoint_pair_count": 1,
                "mean_keypoint_distance": 10.0,
            }
        },
    }
    candidate_sample = {
        **baseline_sample,
        "mean_keypoint_distance": 4.0,
        "labels": {
            "arrow": {
                "matched_count": 1,
                "false_positive_count": 0,
                "false_negative_count": 0,
                "mean_iou": 1.0,
                "keypoint_pair_count": 1,
                "mean_keypoint_distance": 4.0,
            }
        },
    }
    comparison = compare_report_payloads(
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        baseline={
            "task": "keypoint",
            "metric_profile": "keypoint_endpoint_v1",
            "matched_count": 1,
            "keypoint_pair_count": 1,
            "mean_keypoint_distance": 10.0,
            "samples": [baseline_sample],
            "labels": [
                {
                    "label": "arrow",
                    "gt_count": 1,
                    "pred_count": 1,
                    "matched_count": 1,
                    "keypoint_pair_count": 1,
                    "mean_keypoint_distance": 10.0,
                }
            ],
        },
        candidate={
            "task": "keypoint",
            "metric_profile": "keypoint_endpoint_v1",
            "matched_count": 1,
            "keypoint_pair_count": 1,
            "mean_keypoint_distance": 4.0,
            "samples": [candidate_sample],
            "labels": [
                {
                    "label": "arrow",
                    "gt_count": 1,
                    "pred_count": 1,
                    "matched_count": 1,
                    "keypoint_pair_count": 1,
                    "mean_keypoint_distance": 4.0,
                }
            ],
        },
    )

    assert comparison["delta"]["mean_keypoint_distance"] == -6.0
    assert comparison["summary"]["improved_samples"] == 1
    assert comparison["summary"]["regressed_samples"] == 0
    assert comparison["top_improvements"][0]["delta"]["mean_keypoint_distance"] == -6.0
    assert comparison["top_improvements"][0]["delta_score"] > 0.0
    assert comparison["labels"][0]["delta"]["mean_keypoint_distance"] == -6.0
    assert comparison["labels"][0]["delta_score"] > 0.0
