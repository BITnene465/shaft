from __future__ import annotations

import json
from pathlib import Path

from eval_bench.artifacts import RunArtifacts
from eval_bench.comparison import compare_runs
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
from eval_bench.store import EvalBenchStore


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run(tmp_path: Path, *, task: str) -> RunArtifacts:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")
    artifacts = RunArtifacts(tmp_path, f"run_{task}")
    manifest = EvalRunManifest(
        run_id=f"run_{task}",
        model=ModelRef(model_id="model-a", path="outputs/model-a/best"),
        benchmark=BenchmarkRef(
            benchmark_id="bench1",
            root=str(tmp_path / "benchmarks" / "bench1" / "data"),
            split="val",
            tasks=["detection", "keypoint"],
            manifest_path=str(split_path),
        ),
        spec=EvalSpec(
            spec_id=f"{task}.default",
            task=task,  # type: ignore[arg-type]
            prompt=PromptRef(prompt_id=f"{task}.prompt"),
        ),
    )
    artifacts.write_manifest(manifest)
    return artifacts


def test_evaluate_run_writes_detection_metrics(tmp_path: Path) -> None:
    artifacts = _write_run(tmp_path, task="detection")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "data" / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [{"label": "icon", "bbox": [0, 0, 100, 100]}],
        },
    )
    artifacts.write_prediction(
        PredictionDocument(
            image="part1/images/a.png",
            instances=[
                PredictionInstance(label="icon", bbox=[0, 0, 100, 100]),
                PredictionInstance(label="icon", bbox=[200, 200, 260, 260]),
            ],
            metadata={"producer": "test"},
        ),
        task="detection",
    )

    report_path = evaluate_run(store_root=tmp_path, run_id="run_detection")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads((report_path.parent / "summary.json").read_text(encoding="utf-8"))

    assert report["sample_count"] == 1
    assert summary["sample_count"] == 1
    assert "samples" not in summary
    assert report["gt_instance_count"] == 1
    assert report["pred_instance_count"] == 2
    assert report["matched_count"] == 1
    assert report["precision_iou50"] == 0.5
    assert report["recall_iou50"] == 1.0
    assert report["labels"][0]["label"] == "icon"
    assert report["labels"][0]["mean_iou"] == 1.0
    assert report["samples"][0]["matched_count"] == 1
    assert report["samples"][0]["false_positive_count"] == 1
    assert report["samples"][0]["false_negative_count"] == 0
    assert report["samples"][0]["matches"][0]["gt_index"] == 0
    assert report["samples"][0]["matches"][0]["pred_index"] == 0
    assert report["samples"][0]["false_positives"][0]["label"] == "icon"
    run_summary = EvalBenchStore(tmp_path).runs()[0]
    assert run_summary.report_path == str(report_path)
    assert run_summary.precision_iou50 == 0.5
    assert run_summary.recall_iou50 == 1.0
    sample_detail = EvalBenchStore(tmp_path).run_sample_detail("run_detection", sample_index=0)
    assert sample_detail.diagnostics is not None
    assert sample_detail.diagnostics["false_positive_count"] == 1
    assert sample_detail.sample.diagnostics is not None
    assert sample_detail.sample.diagnostics["matched_count"] == 1


def test_evaluate_run_respects_target_labels(tmp_path: Path) -> None:
    artifacts = _write_run(tmp_path, task="detection")
    run_payload = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    run_payload["spec"]["target_labels"] = ["icon"]
    artifacts.manifest_path.write_text(json.dumps(run_payload), encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "data" / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 100, 100]},
                {"label": "arrow", "bbox": [200, 200, 260, 260]},
            ],
        },
    )
    artifacts.write_prediction(
        PredictionDocument(
            image="part1/images/a.png",
            instances=[
                PredictionInstance(label="icon", bbox=[0, 0, 100, 100]),
                PredictionInstance(label="arrow", bbox=[200, 200, 260, 260]),
            ],
            metadata={"producer": "test"},
        ),
        task="detection",
    )

    report_path = evaluate_run(store_root=tmp_path, run_id="run_detection")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads((report_path.parent / "summary.json").read_text(encoding="utf-8"))

    assert report["target_labels"] == ["icon"]
    assert summary["target_labels"] == ["icon"]
    assert report["gt_instance_count"] == 1
    assert report["pred_instance_count"] == 1
    assert [item["label"] for item in report["labels"]] == ["icon"]
    assert "arrow" not in report["samples"][0]["labels"]


def test_evaluate_run_records_keypoint_distance(tmp_path: Path) -> None:
    artifacts = _write_run(tmp_path, task="keypoint")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "data" / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [
                {
                    "label": "arrow",
                    "bbox": [0, 0, 100, 20],
                    "linestrip": [[0, 10], [100, 10]],
                }
            ],
        },
    )
    artifacts.write_prediction(
        PredictionDocument(
            image="part1/images/a.png",
            instances=[
                PredictionInstance(
                    label="arrow",
                    bbox=[0, 0, 100, 20],
                    keypoints=[[0, 10], [90, 10]],
                )
            ],
            metadata={"producer": "test"},
        ),
        task="keypoint",
    )

    report_path = evaluate_run(store_root=tmp_path, run_id="run_keypoint")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["labels"][0]["keypoint_pair_count"] == 1
    assert report["labels"][0]["mean_keypoint_distance"] == 5.0
    assert report["samples"][0]["matches"][0]["keypoint_distance"] == 5.0


def test_compare_runs_reports_improvements_and_regressions(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, task="detection")
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
    _write_json(
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
