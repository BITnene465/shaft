from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.artifacts import RunArtifacts
from eval_bench.comparison import comparison_sample_detail_payload, compare_report_payloads, compare_runs
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
    raw_output = artifacts.raw_outputs_dir / "part1" / "txt" / "a.txt"
    raw_output.parent.mkdir(parents=True)
    raw_output.write_text('[{"label":"icon","bbox_2d":[0,0,1000,1000]}]', encoding="utf-8")

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
    assert report["empty_prediction_rate"] == 0.0
    assert report["pred_gt_ratio"] == 2.0
    assert report["output_char_length"]["count"] == 1
    assert report["output_char_length"]["min"] == len(
        '[{"label":"icon","bbox_2d":[0,0,1000,1000]}]'
    )
    assert report["output_token_length"]["count"] == 1
    assert report["dense_sample_buckets"] == [
        {
            "bucket": "1",
            "sample_count": 1,
            "gt_instance_count": 1,
            "pred_instance_count": 2,
            "matched_count": 1,
            "empty_prediction_count": 0,
            "precision_iou50": 0.5,
            "recall_iou50": 1.0,
            "empty_prediction_rate": 0.0,
        }
    ]
    assert summary["empty_prediction_rate"] == 0.0
    assert summary["pred_gt_ratio"] == 2.0
    assert summary["dense_sample_buckets"][0]["bucket"] == "1"
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
    assert run_summary.f1_iou50 == pytest.approx(2 / 3)
    assert run_summary.precision_iou50 == 0.5
    assert run_summary.recall_iou50 == 1.0
    sample_detail = EvalBenchStore(tmp_path).run_sample_detail("run_detection", sample_index=0)
    assert sample_detail.diagnostics is not None
    assert sample_detail.diagnostics["false_positive_count"] == 1
    assert sample_detail.sample.diagnostics is not None
    assert sample_detail.sample.diagnostics["matched_count"] == 1


def test_evaluate_run_uses_named_benchmark_split(tmp_path: Path) -> None:
    bench_dir = tmp_path / "benchmarks" / "bench1"
    (bench_dir / "splits").mkdir(parents=True)
    (bench_dir / "splits" / "grounding_arrow.txt").write_text(
        "part1/json/arrow.json\n",
        encoding="utf-8",
    )
    (bench_dir / "splits" / "grounding_shape.txt").write_text(
        "part1/json/shape.json\n",
        encoding="utf-8",
    )
    _write_json(
        bench_dir / "data" / "part1" / "json" / "arrow.json",
        {
            "image_path": "part1/images/arrow.png",
            "instances": [{"label": "arrow", "bbox": [0, 0, 100, 100]}],
        },
    )
    _write_json(
        bench_dir / "data" / "part1" / "json" / "shape.json",
        {
            "image_path": "part1/images/shape.png",
            "instances": [{"label": "shape", "bbox": [0, 0, 100, 100]}],
        },
    )
    artifacts = RunArtifacts(tmp_path, "run_named_split")
    artifacts.write_manifest(
        EvalRunManifest(
            run_id="run_named_split",
            model=ModelRef(model_id="model-a", path="outputs/model-a/best"),
            benchmark=BenchmarkRef(
                benchmark_id="bench1",
                root=str(bench_dir / "data"),
                split="grounding_shape",
                tasks=["detection"],
                manifest_path=str(bench_dir / "splits" / "grounding_arrow.txt"),
            ),
            spec=EvalSpec(
                spec_id="detection.default",
                task="detection",
                prompt=PromptRef(prompt_id="grounding_shape.latest"),
                target_labels=["shape"],
            ),
        )
    )
    run_payload = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    run_payload["benchmark"]["split_manifests"] = {
        "grounding_arrow": str(bench_dir / "splits" / "grounding_arrow.txt"),
        "grounding_shape": str(bench_dir / "splits" / "grounding_shape.txt"),
    }
    artifacts.manifest_path.write_text(json.dumps(run_payload), encoding="utf-8")
    artifacts.write_prediction(
        PredictionDocument(
            image="part1/images/shape.png",
            instances=[PredictionInstance(label="shape", bbox=[0, 0, 100, 100])],
            metadata={"producer": "test"},
        ),
        task="detection",
    )

    report_path = evaluate_run(store_root=tmp_path, run_id="run_named_split")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["sample_count"] == 1
    assert report["samples"][0]["json_path"] == "part1/json/shape.json"
    assert report["matched_count"] == 1


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
        _write_json(
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
    assert report["target_labels_source"] == "explicit"
    assert summary["target_labels"] == ["icon"]
    assert summary["target_labels_source"] == "explicit"
    assert summary["metric_profile"] == "detection_iou_v1"
    assert report["gt_instance_count"] == 1
    assert report["pred_instance_count"] == 1
    assert [item["label"] for item in report["labels"]] == ["icon"]
    assert "arrow" not in report["samples"][0]["labels"]
    store = EvalBenchStore(tmp_path)
    sample_page = store.run_sample_page("run_detection")
    assert sample_page.labels == ["icon"]
    assert sample_page.samples[0].gt_instance_count == 1
    assert sample_page.samples[0].pred_instance_count == 1
    assert sample_page.samples[0].labels == ["icon"]
    sample_detail = store.run_sample_detail("run_detection", sample_index=0)
    assert [item["label"] for item in sample_detail.gt_instances] == ["icon"]
    assert [item["label"] for item in sample_detail.pred_instances] == ["icon"]


def test_evaluate_run_rejects_keypoint_label_subtasks(tmp_path: Path) -> None:
    artifacts = _write_run(tmp_path, task="keypoint")
    run_payload = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    run_payload["spec"]["target_labels"] = ["icon"]
    artifacts.manifest_path.write_text(json.dumps(run_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="keypoint target_labels only support arrow"):
        evaluate_run(store_root=tmp_path, run_id="run_keypoint")


def test_evaluate_run_infers_layout_target_labels_from_prompt_id(tmp_path: Path) -> None:
    artifacts = _write_run(tmp_path, task="detection")
    run_payload = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    run_payload["spec"]["prompt"]["prompt_id"] = "grounding_layout.latest"
    run_payload["spec"].pop("target_labels", None)
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

    assert report["target_labels"] == ["icon", "image", "shape"]
    assert report["target_labels_source"] == "suite_default"
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
    assert report["keypoint_pair_count"] == 1
    assert report["mean_keypoint_distance"] == 5.0
    assert report["samples"][0]["matches"][0]["keypoint_distance"] == 5.0


def test_keypoint_profile_matches_by_endpoint_distance_not_bbox_iou(tmp_path: Path) -> None:
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
                    keypoints=[[90, 10], [0, 10]],
                )
            ],
            metadata={"producer": "test"},
        ),
        task="keypoint",
    )

    report_path = evaluate_run(store_root=tmp_path, run_id="run_keypoint")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["metric_profile"] == "keypoint_endpoint_v1"
    assert report["matched_count"] == 0
    assert report["precision_iou50"] == 0.0
    assert report["recall_iou50"] == 0.0
    assert report["keypoint_pair_count"] == 0
    assert report["samples"][0]["false_negative_count"] == 1
    assert report["samples"][0]["false_positive_count"] == 1


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


def test_compare_reports_warns_when_target_label_scope_differs() -> None:
    comparison = compare_report_payloads(
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        baseline={
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": [],
            "samples": [],
            "labels": [],
        },
        candidate={
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["arrow"],
            "samples": [],
            "labels": [],
        },
    )

    assert "baseline and candidate target labels differ" in comparison["warnings"]


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
