from __future__ import annotations

from pathlib import Path

from .files import write_json


def write_basic_run(base_dir: Path, *, run_id: str = "run1", status: str = "succeeded") -> None:
    data_root = base_dir / "benchmarks" / "bench1" / "data"
    write_json(
        base_dir / "runs" / run_id / "run.json",
        {
            "run_id": run_id,
            "status": status,
            "created_at": "2026-05-09T00:10:00Z",
            "model": {"model_id": "model-a", "path": "outputs/model-a/best"},
            "benchmark": {
                "benchmark_id": "bench1",
                "root": str(data_root),
                "split": "val",
                "tasks": ["detection"],
            },
            "spec": {"task": "detection"},
        },
    )


def write_sample_store(base_dir: Path) -> None:
    split_path = base_dir / "benchmarks" / "bench1" / "splits" / "val.txt"
    data_root = base_dir / "benchmarks" / "bench1" / "data"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\npart1/json/b.json\n", encoding="utf-8")
    write_json(
        base_dir / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["layout", "arrow"],
            "split": "val",
            "sample_count": 2,
            "root": str(data_root),
            "manifest_path": str(split_path),
            "labels": ["arrow", "icon"],
        },
    )
    write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "image_width": 100,
            "image_height": 100,
            "instances": [
                {"label": "icon", "bbox": [0, 0, 40, 40]},
                {"label": "arrow", "bbox": [50, 50, 90, 90]},
            ],
        },
    )
    write_json(
        data_root / "part1" / "json" / "b.json",
        {
            "image_path": "part1/images/b.png",
            "instances": [{"label": "icon", "bbox": [0, 0, 30, 30]}],
        },
    )
    run_dir = base_dir / "runs" / "run_arrow"
    write_json(
        run_dir / "run.json",
        {
            "run_id": "run_arrow",
            "status": "succeeded",
            "created_at": "2026-05-09T00:10:00Z",
            "model": {"model_id": "model-a", "path": "outputs/model-a/best"},
            "benchmark": {
                "benchmark_id": "bench1",
                "root": str(data_root),
                "split": "val",
                "tasks": ["detection"],
                "manifest_path": str(split_path),
            },
            "spec": {
                "task": "detection",
                "metric_profile": "detection_iou_v1",
                "target_labels": ["arrow"],
                "metadata": {"target_labels_source": "explicit"},
                "prompt": {"prompt_id": "grounding_arrow.v2.4.main"},
            },
        },
    )
    write_json(
        run_dir / "predictions" / "part1" / "json" / "a.json",
        {
            "image": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 40, 40]},
                {"label": "arrow", "bbox": [52, 52, 88, 88]},
            ],
        },
    )
    write_json(
        run_dir / "reports" / "summary.json",
        {
            "run_id": "run_arrow",
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["arrow"],
            "target_labels_source": "explicit",
            "prediction_file_count": 1,
            "precision_iou50": 1.0,
            "recall_iou50": 1.0,
            "mean_iou": 0.81,
            "labels": ["arrow"],
        },
    )
    write_json(
        run_dir / "reports" / "metrics.json",
        {
            "run_id": "run_arrow",
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["arrow"],
            "target_labels_source": "explicit",
            "sample_count": 2,
            "prediction_file_count": 1,
            "precision_iou50": 1.0,
            "recall_iou50": 1.0,
            "mean_iou": 0.81,
            "labels": [
                {
                    "label": "arrow",
                    "gt_count": 1,
                    "pred_count": 1,
                    "matched_count": 1,
                    "false_positive_count": 0,
                    "false_negative_count": 0,
                    "precision_iou50": 1.0,
                    "recall_iou50": 1.0,
                    "mean_iou": 0.81,
                }
            ],
            "samples": [
                {
                    "index": 0,
                    "image": "part1/images/a.png",
                    "gt_instance_count": 1,
                    "pred_instance_count": 1,
                    "matched_count": 1,
                    "false_negative_count": 0,
                    "false_positive_count": 0,
                    "mean_iou": 0.81,
                    "labels": {
                        "arrow": {
                            "gt_count": 1,
                            "pred_count": 1,
                            "matched_count": 1,
                            "false_negative_count": 0,
                            "false_positive_count": 0,
                            "mean_iou": 0.81,
                        }
                    },
                    "matches": [{"label": "arrow", "gt_index": 0, "pred_index": 0, "iou": 0.81}],
                    "false_negatives": [],
                    "false_positives": [],
                },
                {
                    "index": 1,
                    "image": "part1/images/b.png",
                    "gt_instance_count": 0,
                    "pred_instance_count": 0,
                    "matched_count": 0,
                    "false_negative_count": 0,
                    "false_positive_count": 0,
                    "mean_iou": 0.0,
                    "labels": {},
                    "matches": [],
                    "false_negatives": [],
                    "false_positives": [],
                },
            ],
        },
    )
