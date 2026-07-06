from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.artifacts import RunArtifacts
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
from support.evaluator import write_eval_run
from support.files import write_json


def test_evaluate_run_writes_detection_metrics(tmp_path: Path) -> None:
    artifacts = write_eval_run(tmp_path, task="detection")
    write_json(
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
    assert report["benchmark_id"] == "bench1"
    assert report["benchmark_split"] == "val"
    assert summary["sample_count"] == 1
    assert summary["benchmark_id"] == "bench1"
    assert summary["benchmark_split"] == "val"
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
    write_json(
        bench_dir / "data" / "part1" / "json" / "arrow.json",
        {
            "image_path": "part1/images/arrow.png",
            "instances": [{"label": "arrow", "bbox": [0, 0, 100, 100]}],
        },
    )
    write_json(
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


def test_evaluate_run_respects_target_labels(tmp_path: Path) -> None:
    artifacts = write_eval_run(tmp_path, task="detection", target_labels=["icon"])
    write_json(
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


def test_evaluate_run_infers_layout_target_labels_from_prompt_id(tmp_path: Path) -> None:
    artifacts = write_eval_run(
        tmp_path,
        task="detection",
        prompt_id="grounding_layout.latest",
    )
    write_json(
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
    artifacts = write_eval_run(tmp_path, task="keypoint")
    write_json(
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
    artifacts = write_eval_run(tmp_path, task="keypoint")
    write_json(
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
