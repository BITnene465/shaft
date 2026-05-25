from __future__ import annotations

import json
from pathlib import Path

from eval_bench.cli import (
    _build_parser,
    _cmd_create_job,
    _cmd_get_run_note,
    _cmd_init_run,
    _cmd_import_predictions,
    _cmd_list_benchmarks,
    _cmd_list_benchmark_samples,
    _cmd_list_comparisons,
    _cmd_list_job_templates,
    _cmd_list_jobs,
    _cmd_list_prompt_templates,
    _cmd_list_run_samples,
    _cmd_list_runs,
    _cmd_list_services,
    _cmd_preflight_job,
    _cmd_rank_board,
    _cmd_register_service,
    _cmd_delete_prompt_template,
    _cmd_set_run_note,
    _cmd_show_benchmark_sample,
    _cmd_show_run,
    _cmd_show_run_report,
    _cmd_show_run_sample,
    _cmd_upsert_prompt_template,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_sample_store(tmp_path: Path) -> None:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\npart1/json/b.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
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
    _write_json(
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
    _write_json(
        data_root / "part1" / "json" / "b.json",
        {
            "image_path": "part1/images/b.png",
            "instances": [{"label": "icon", "bbox": [0, 0, 30, 30]}],
        },
    )
    run_dir = tmp_path / "runs" / "run_arrow"
    _write_json(
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
                "prompt": {"prompt_id": "grounding_arrow.latest"},
            },
        },
    )
    _write_json(
        run_dir / "predictions" / "part1" / "json" / "a.json",
        {
            "image": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 40, 40]},
                {"label": "arrow", "bbox": [52, 52, 88, 88]},
            ],
        },
    )
    _write_json(
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
    _write_json(
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


def test_init_run_cli_accepts_target_label_subset(tmp_path: Path) -> None:
    args = _build_parser().parse_args(
        [
            "init-run",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--task",
            "detection",
            "--model-id",
            "model-a",
            "--model-path",
            "outputs/model-a/best",
            "--benchmark-id",
            "bench1",
            "--benchmark-root",
            str(tmp_path / "benchmarks" / "bench1" / "data"),
            "--split",
            "val",
            "--spec-id",
            "layout.icons",
            "--prompt-id",
            "grounding_layout.latest",
            "--target-label",
            "icon",
            "--target-label",
            "image",
        ]
    )

    _cmd_init_run(args)

    payload = json.loads((tmp_path / "runs" / "run1" / "run.json").read_text(encoding="utf-8"))
    assert payload["spec"]["target_labels"] == ["icon", "image"]
    assert payload["spec"]["metadata"]["target_labels_source"] == "explicit"


def test_init_run_cli_infers_target_labels_from_prompt_policy(tmp_path: Path) -> None:
    args = _build_parser().parse_args(
        [
            "init-run",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--task",
            "detection",
            "--model-id",
            "model-a",
            "--model-path",
            "outputs/model-a/best",
            "--benchmark-id",
            "bench1",
            "--benchmark-root",
            str(tmp_path / "benchmarks" / "bench1" / "data"),
            "--split",
            "val",
            "--spec-id",
            "layout.default",
            "--prompt-id",
            "grounding_layout.latest",
        ]
    )

    _cmd_init_run(args)

    payload = json.loads((tmp_path / "runs" / "run1" / "run.json").read_text(encoding="utf-8"))
    assert payload["spec"]["target_labels"] == ["icon", "image", "shape"]
    assert payload["spec"]["metadata"]["target_labels_source"] == "legacy_prompt_id"


def test_cli_gets_and_sets_run_note(tmp_path: Path, capsys) -> None:
    init_args = _build_parser().parse_args(
        [
            "init-run",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--task",
            "detection",
            "--model-id",
            "model-a",
            "--model-path",
            "outputs/model-a/best",
            "--benchmark-id",
            "bench1",
            "--benchmark-root",
            str(tmp_path / "benchmarks" / "bench1" / "data"),
            "--split",
            "val",
            "--spec-id",
            "layout.icons",
            "--prompt-id",
            "grounding_layout.latest",
        ]
    )
    _cmd_init_run(init_args)
    capsys.readouterr()

    note_file = tmp_path / "note.md"
    note_file.write_text("repro: ckpt epoch_3\nidea: prompt v2", encoding="utf-8")
    set_args = _build_parser().parse_args(
        [
            "set-run-note",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--note-file",
            str(note_file),
        ]
    )
    _cmd_set_run_note(set_args)
    set_payload = json.loads(capsys.readouterr().out)

    assert set_payload["note"] == "repro: ckpt epoch_3\nidea: prompt v2"
    assert set_payload["max_length"] == 20_000

    get_args = _build_parser().parse_args(
        ["get-run-note", "--output-root", str(tmp_path), "--run-id", "run1"]
    )
    _cmd_get_run_note(get_args)
    get_payload = json.loads(capsys.readouterr().out)
    assert get_payload["note"] == set_payload["note"]
    assert get_payload["path"].endswith("runs/run1/note.json")
    assert get_payload["max_length"] == 20_000


def test_cli_import_predictions_accepts_target_label_subset(tmp_path: Path, capsys) -> None:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "split": "val",
            "sample_count": 1,
            "root": str(data_root),
            "manifest_path": str(split_path),
        },
    )
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 100, 100]},
                {"label": "arrow", "bbox": [200, 200, 260, 260]},
            ],
        },
    )
    prediction_root = tmp_path / "predictions"
    _write_json(
        prediction_root / "part1" / "json" / "a.json",
        {
            "image": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 100, 100]},
                {"label": "arrow", "bbox": [200, 200, 260, 260]},
            ],
        },
    )
    args = _build_parser().parse_args(
        [
            "import-predictions",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "imported_arrow",
            "--benchmark-id",
            "bench1",
            "--prediction-root",
            str(prediction_root),
            "--task",
            "detection",
            "--model-id",
            "external-model",
            "--prompt-id",
            "grounding_layout.latest",
            "--target-label",
            "arrow",
        ]
    )

    _cmd_import_predictions(args)
    payload = json.loads(capsys.readouterr().out)
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))

    assert payload["run_id"] == "imported_arrow"
    assert report["target_labels"] == ["arrow"]
    assert report["target_labels_source"] == "explicit"
    assert [item["label"] for item in report["labels"]] == ["arrow"]


def test_cli_prints_filtered_rank_board(tmp_path: Path, capsys) -> None:
    for run_id, label, precision in (
        ("run_a", "icon", 0.9),
        ("run_b", "arrow", 0.5),
    ):
        run_dir = tmp_path / "runs" / run_id
        (run_dir / "reports").mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "succeeded",
                    "created_at": "2026-05-09T00:10:00Z",
                    "model": {"model_id": run_id, "path": "outputs/model/best"},
                    "benchmark": {
                        "benchmark_id": "bench1",
                        "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
                        "split": "val",
                        "tasks": ["detection"],
                    },
                    "spec": {
                        "task": "detection",
                        "metric_profile": "detection_iou_v1",
                        "target_labels": [label],
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "reports" / "summary.json").write_text(
            json.dumps(
                {
                    "precision_iou50": precision,
                    "recall_iou50": precision,
                    "mean_iou": precision,
                    "prediction_file_count": 1,
                }
            ),
            encoding="utf-8",
        )

    args = _build_parser().parse_args(
        [
            "rank-board",
            "--output-root",
            str(tmp_path),
            "--label",
            "icon",
            "--metric-profile",
            "detection_iou_v1",
            "--min-score",
            "0.7",
            "--sort-by",
            "run_id",
            "--sort-order",
            "desc",
        ]
    )
    _cmd_rank_board(args)
    payload = json.loads(capsys.readouterr().out)

    assert payload["total"] == 1
    assert payload["sort_by"] == "run_id"
    assert payload["sort_order"] == "desc"
    assert payload["filters"]["min_score"] == "0.7"
    assert payload["facets"]["metric_profiles"] == [{"value": "detection_iou_v1", "count": 1}]
    assert payload["entries"][0]["run_id"] == "run_a"
    assert payload["entries"][0]["rank"] == 1


def test_cli_lists_benchmarks_runs_and_comparisons_with_agent_filters(
    tmp_path: Path,
    capsys,
) -> None:
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "labels": ["arrow", "icon"],
            "layers": ["layout"],
            "split": "val",
            "sample_count": 2,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"),
            "created_at": "2026-05-09T00:00:00Z",
        },
    )
    _write_json(
        tmp_path / "benchmarks" / "bench2" / "benchmark.json",
        {
            "benchmark_id": "bench2",
            "tasks": ["keypoint"],
            "layers": ["arrow"],
            "split": "val",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "bench2" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench2" / "splits" / "val.txt"),
        },
    )
    _write_json(
        tmp_path / "runs" / "run_a" / "run.json",
        {
            "run_id": "run_a",
            "status": "succeeded",
            "created_at": "2026-05-09T00:10:00Z",
            "model": {"model_id": "model-a", "path": "outputs/model-a/best"},
            "benchmark": {
                "benchmark_id": "bench1",
                "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
                "split": "val",
                "tasks": ["detection"],
            },
            "spec": {
                "task": "detection",
                "metric_profile": "detection_iou_v1",
                "target_labels": ["arrow"],
                "prompt": {"prompt_id": "grounding_arrow.latest"},
            },
        },
    )
    _write_json(
        tmp_path / "runs" / "run_b" / "run.json",
        {
            "run_id": "run_b",
            "status": "failed",
            "model": {"model_id": "model-b", "path": "outputs/model-b/best"},
            "benchmark": {
                "benchmark_id": "bench2",
                "root": str(tmp_path / "benchmarks" / "bench2" / "data"),
                "split": "val",
                "tasks": ["keypoint"],
            },
            "spec": {
                "task": "keypoint",
                "metric_profile": "keypoint_endpoint_v1",
                "target_labels": ["arrow"],
                "prompt": {"prompt_id": "keypoint_arrow.latest"},
            },
        },
    )
    _write_json(
        tmp_path / "exports" / "comparisons" / "run_base__vs__run_a.json",
        {
            "comparison_id": "run_base__vs__run_a",
            "baseline_run_id": "run_base",
            "candidate_run_id": "run_a",
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["arrow"],
            "sample_count": 2,
            "created_at": "2026-05-09T00:30:00Z",
            "delta": {"precision_iou50": 0.2},
            "summary": {"improved_samples": 1},
        },
    )

    benchmark_args = _build_parser().parse_args(
        [
            "list-benchmarks",
            "--output-root",
            str(tmp_path),
            "--task",
            "detection",
            "--layer",
            "layout",
            "--split",
            "val",
            "--query",
            "bench1",
        ]
    )
    _cmd_list_benchmarks(benchmark_args)
    benchmarks = json.loads(capsys.readouterr().out)
    assert benchmarks["total"] == 1
    assert benchmarks["filters"]["task"] == "detection"
    assert benchmarks["filters"]["split"] == "val"
    assert benchmarks["benchmarks"][0]["benchmark_id"] == "bench1"
    assert benchmarks["benchmarks"][0]["labels"] == ["arrow", "icon"]

    run_args = _build_parser().parse_args(
        [
            "list-runs",
            "--output-root",
            str(tmp_path),
            "--task",
            "detection",
            "--benchmark-id",
            "bench1",
            "--label",
            "arrow",
            "--model-id",
            "model-a",
            "--metric-profile",
            "detection_iou_v1",
            "--query",
            "grounding",
        ]
    )
    _cmd_list_runs(run_args)
    runs = json.loads(capsys.readouterr().out)
    assert runs["total"] == 1
    assert runs["filters"]["label"] == "arrow"
    assert runs["runs"][0]["run_id"] == "run_a"
    assert runs["runs"][0]["target_labels"] == ["arrow"]

    comparison_args = _build_parser().parse_args(
        [
            "list-comparisons",
            "--output-root",
            str(tmp_path),
            "--task",
            "detection",
            "--baseline-run-id",
            "run_base",
            "--label",
            "arrow",
            "--query",
            "run_a",
        ]
    )
    _cmd_list_comparisons(comparison_args)
    comparisons = json.loads(capsys.readouterr().out)
    assert comparisons["total"] == 1
    assert comparisons["filters"]["baseline_run_id"] == "run_base"
    assert comparisons["comparisons"][0]["comparison_id"] == "run_base__vs__run_a"
    assert comparisons["comparisons"][0]["metric_profile"] == "detection_iou_v1"


def test_cli_manages_job_and_prompt_templates_for_agents(tmp_path: Path, capsys) -> None:
    job_template_args = _build_parser().parse_args(["list-job-templates", "--query", "keypoint"])
    _cmd_list_job_templates(job_template_args)
    job_templates = json.loads(capsys.readouterr().out)
    assert job_templates["total"] == 1
    assert "keypoint_eval_job" in job_templates["templates"]
    assert job_templates["templates"]["keypoint_eval_job"]["manifest"]["eval"]["task"] == "keypoint"

    list_args = _build_parser().parse_args(
        ["list-prompt-templates", "--output-root", str(tmp_path), "--task", "detection"]
    )
    _cmd_list_prompt_templates(list_args)
    prompt_templates = json.loads(capsys.readouterr().out)
    assert prompt_templates["total"] >= 1
    assert "grounding_arrow.latest" in prompt_templates["by_id"]
    assert prompt_templates["by_id"]["grounding_arrow.latest"]["task"] == "detection"

    custom_payload = {
        "prompt_id": "custom.arrow.v1",
        "label": "Custom Arrow",
        "task": "detection",
        "system_prompt": "You inspect visual structures.",
        "user_prompt": "Detect arrows only.",
        "parser": "raw_data_detection_v1",
        "metric_profile": "detection_iou_v1",
        "metadata": {"target_labels": ["arrow"], "source": "agent_cli_test"},
    }
    upsert_args = _build_parser().parse_args(
        [
            "upsert-prompt-template",
            "--output-root",
            str(tmp_path),
            "--payload-json",
            json.dumps(custom_payload),
        ]
    )
    _cmd_upsert_prompt_template(upsert_args)
    upserted = json.loads(capsys.readouterr().out)
    assert upserted["prompt_id"] == "custom.arrow.v1"
    assert upserted["metadata"]["target_labels"] == ["arrow"]

    custom_list_args = _build_parser().parse_args(
        [
            "list-prompt-templates",
            "--output-root",
            str(tmp_path),
            "--query",
            "agent_cli_test",
        ]
    )
    _cmd_list_prompt_templates(custom_list_args)
    custom_list = json.loads(capsys.readouterr().out)
    assert custom_list["total"] == 1
    assert custom_list["templates"][0]["prompt_id"] == "custom.arrow.v1"

    delete_args = _build_parser().parse_args(
        [
            "delete-prompt-template",
            "--output-root",
            str(tmp_path),
            "--prompt-id",
            "custom.arrow.v1",
        ]
    )
    _cmd_delete_prompt_template(delete_args)
    deleted = json.loads(capsys.readouterr().out)
    assert deleted == {"prompt_id": "custom.arrow.v1", "deleted": True}


def test_cli_reads_run_reports_and_scoped_samples_for_agents(tmp_path: Path, capsys) -> None:
    _write_sample_store(tmp_path)

    show_args = _build_parser().parse_args(
        ["show-run", "--output-root", str(tmp_path), "--run-id", "run_arrow"]
    )
    _cmd_show_run(show_args)
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["run"]["run_id"] == "run_arrow"
    assert run_payload["run"]["target_labels"] == ["arrow"]
    assert run_payload["run"]["precision_iou50"] == 1.0

    report_args = _build_parser().parse_args(
        ["show-run-report", "--output-root", str(tmp_path), "--run-id", "run_arrow", "--summary"]
    )
    _cmd_show_run_report(report_args)
    report = json.loads(capsys.readouterr().out)
    assert report["target_labels_source"] == "explicit"
    assert report["labels"] == ["arrow"]

    samples_args = _build_parser().parse_args(
        [
            "list-run-samples",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run_arrow",
            "--label",
            "arrow",
        ]
    )
    _cmd_list_run_samples(samples_args)
    samples = json.loads(capsys.readouterr().out)
    assert samples["labels"] == ["arrow"]
    assert samples["total"] == 1
    assert samples["samples"][0]["labels"] == ["arrow"]
    assert samples["samples"][0]["gt_instance_count"] == 1
    assert samples["samples"][0]["diagnostics"]["matched_count"] == 1

    detail_args = _build_parser().parse_args(
        [
            "show-run-sample",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run_arrow",
            "--sample-index",
            "0",
        ]
    )
    _cmd_show_run_sample(detail_args)
    detail = json.loads(capsys.readouterr().out)
    assert [item["label"] for item in detail["gt_instances"]] == ["arrow"]
    assert [item["label"] for item in detail["pred_instances"]] == ["arrow"]
    assert [item["label"] for item in detail["raw_payload"]["instances"]] == ["arrow"]
    assert [item["label"] for item in detail["prediction_payload"]["instances"]] == ["arrow"]


def test_cli_reads_benchmark_samples_for_agents(tmp_path: Path, capsys) -> None:
    _write_sample_store(tmp_path)

    samples_args = _build_parser().parse_args(
        [
            "list-benchmark-samples",
            "--output-root",
            str(tmp_path),
            "--benchmark-id",
            "bench1",
            "--label",
            "arrow",
        ]
    )
    _cmd_list_benchmark_samples(samples_args)
    samples = json.loads(capsys.readouterr().out)
    assert samples["benchmark_id"] == "bench1"
    assert samples["labels"] == ["arrow", "icon"]
    assert samples["total"] == 1
    assert samples["samples"][0]["labels"] == ["arrow", "icon"]

    detail_args = _build_parser().parse_args(
        [
            "show-benchmark-sample",
            "--output-root",
            str(tmp_path),
            "--benchmark-id",
            "bench1",
            "--sample-index",
            "0",
        ]
    )
    _cmd_show_benchmark_sample(detail_args)
    detail = json.loads(capsys.readouterr().out)
    assert detail["sample"]["instance_count"] == 2
    assert [item["label"] for item in detail["gt_instances"]] == ["icon", "arrow"]


def test_cli_preflights_and_creates_manifest_first_job(tmp_path: Path, capsys) -> None:
    model_path = tmp_path / "models" / "model-a"
    _write_json(model_path / "config.json", {"num_attention_heads": 4})
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "split": "val",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"),
        },
    )
    payload_path = tmp_path / "job.json"
    _write_json(
        payload_path,
        {
            "manifest": {
                "kind": "eval_job",
                "runtime": {
                    "mode": "ephemeral",
                    "engine": "vllm_openai",
                    "env": {"CUDA_VISIBLE_DEVICES": "0"},
                    "args": {
                        "model": str(model_path),
                        "served-model-name": "model-a",
                        "host": "127.0.0.1",
                        "tensor-parallel-size": 1,
                        "trust-remote-code": True,
                    },
                },
                "eval": {
                    "model_id": "model-a",
                    "benchmark_id": "bench1",
                    "task": "detection",
                    "prompt_id": "grounding_arrow.latest",
                    "target_labels": ["arrow"],
                },
            }
        },
    )

    preflight_args = _build_parser().parse_args(
        [
            "preflight-job",
            "--output-root",
            str(tmp_path),
            "--payload-file",
            str(payload_path),
        ]
    )
    _cmd_preflight_job(preflight_args)
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["ok"] is True
    assert preflight["kind"] == "eval_job"
    assert preflight["resolved_payload"]["prompt_text"]
    assert preflight["resolved_payload"]["target_labels"] == ["arrow"]
    assert preflight["runtime_command"][0]

    create_args = _build_parser().parse_args(
        [
            "create-job",
            "--output-root",
            str(tmp_path),
            "--payload-file",
            str(payload_path),
        ]
    )
    _cmd_create_job(create_args)
    job = json.loads(capsys.readouterr().out)
    assert job["kind"] == "eval"
    assert job["status"] == "queued"
    assert job["payload"]["benchmark_id"] == "bench1"
    assert job["payload"]["target_labels"] == ["arrow"]
    assert job["payload"]["manifest"]["kind"] == "eval_job"

    list_args = _build_parser().parse_args(
        [
            "list-jobs",
            "--output-root",
            str(tmp_path),
            "--kind",
            "eval",
            "--status",
            "queued",
            "--query",
            "grounding_arrow",
        ]
    )
    _cmd_list_jobs(list_args)
    jobs = json.loads(capsys.readouterr().out)
    assert jobs["total"] == 1
    assert jobs["filters"]["kind"] == "eval"
    assert jobs["jobs"][0]["job_id"] == job["job_id"]


def test_cli_lists_services_with_agent_filters(tmp_path: Path, capsys) -> None:
    register_args = _build_parser().parse_args(
        [
            "register-service",
            "--output-root",
            str(tmp_path),
            "--kind",
            "external_vllm",
            "--service-id",
            "external-qwen3vl",
            "--endpoint",
            "http://127.0.0.1:8000/v1",
            "--served-model-name",
            "qwen3vl-best",
        ]
    )
    _cmd_register_service(register_args)
    capsys.readouterr()

    list_args = _build_parser().parse_args(
        [
            "list-services",
            "--output-root",
            str(tmp_path),
            "--kind",
            "external_vllm",
            "--status",
            "registered",
            "--query",
            "qwen3vl",
        ]
    )
    _cmd_list_services(list_args)
    payload = json.loads(capsys.readouterr().out)

    assert payload["total"] == 1
    assert payload["filters"]["kind"] == "external_vllm"
    assert payload["services"][0]["service_id"] == "external-qwen3vl"
