from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.prediction_import import import_predictions_for_benchmark
from eval_bench.store import EvalBenchStore


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_benchmark(tmp_path: Path) -> None:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "test.txt"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\npart1/json/b.json\n", encoding="utf-8")
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection", "keypoint"],
            "layers": ["layout"],
            "split": "test",
            "sample_count": 2,
            "root": str(data_root),
            "manifest_path": str(split_path),
        },
    )
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [{"label": "icon", "bbox": [0, 0, 100, 100]}],
        },
    )
    _write_json(
        data_root / "part1" / "json" / "b.json",
        {
            "image_path": "part1/images/b.png",
            "instances": [{"label": "shape", "bbox": [20, 20, 80, 80]}],
        },
    )


def test_import_predictions_creates_run_and_evaluates_against_benchmark(tmp_path: Path) -> None:
    _write_benchmark(tmp_path)
    prediction_root = tmp_path / "external_predictions"
    _write_json(
        prediction_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [{"label": "icon", "bbox_2d": [0, 0, 100, 100]}],
        },
    )

    result = import_predictions_for_benchmark(
        store_root=tmp_path,
        run_id="imported_run",
        benchmark_id="bench1",
        prediction_root=prediction_root,
        task="detection",
        model_id="external-model",
    )

    assert result.imported_predictions == 1
    assert result.missing_predictions == ["part1/json/b.json"]
    assert result.report_path is not None
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["sample_count"] == 2
    assert report["prediction_file_count"] == 1
    assert report["matched_count"] == 1
    assert report["missing_predictions"] == ["part1/json/b.json"]
    assert report["recall_iou50"] == 0.5
    run = EvalBenchStore(tmp_path).runs()[0]
    assert run.run_id == "imported_run"
    assert run.model_id == "external-model"
    assert run.inference["backend"] == "imported"
    detail = EvalBenchStore(tmp_path).run_sample_detail("imported_run", sample_index=0)
    assert detail.diagnostics is not None
    assert detail.diagnostics["matched_count"] == 1


def test_import_predictions_applies_prompt_target_labels(tmp_path: Path) -> None:
    _write_benchmark(tmp_path)
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
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
    prediction_root = tmp_path / "external_predictions"
    _write_json(
        prediction_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox_2d": [0, 0, 100, 100]},
                {"label": "arrow", "bbox_2d": [200, 200, 260, 260]},
            ],
        },
    )

    result = import_predictions_for_benchmark(
        store_root=tmp_path,
        run_id="layout_import",
        benchmark_id="bench1",
        prediction_root=prediction_root,
        task="detection",
        model_id="external-model",
        prompt_id="grounding_layout.latest",
    )

    assert result.report_path is not None
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["target_labels"] == ["icon", "image", "shape"]
    assert report["target_labels_source"] == "legacy_prompt_id"
    assert [item["label"] for item in report["labels"]] == ["icon", "shape"]
    assert "arrow" not in report["samples"][0]["labels"]


def test_import_predictions_explicit_target_labels_override_prompt_policy(tmp_path: Path) -> None:
    _write_benchmark(tmp_path)
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
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
    prediction_root = tmp_path / "external_predictions"
    _write_json(
        prediction_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox_2d": [0, 0, 100, 100]},
                {"label": "arrow", "bbox_2d": [200, 200, 260, 260]},
            ],
        },
    )

    result = import_predictions_for_benchmark(
        store_root=tmp_path,
        run_id="arrow_import",
        benchmark_id="bench1",
        prediction_root=prediction_root,
        task="detection",
        model_id="external-model",
        prompt_id="grounding_layout.latest",
        target_labels=["arrow"],
    )

    assert result.report_path is not None
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["target_labels"] == ["arrow"]
    assert report["target_labels_source"] == "explicit"
    assert [item["label"] for item in report["labels"]] == ["arrow"]
    assert "icon" not in report["samples"][0]["labels"]


def test_import_predictions_rejects_keypoint_label_subtasks(tmp_path: Path) -> None:
    _write_benchmark(tmp_path)
    prediction_root = tmp_path / "external_predictions"
    prediction_root.mkdir()

    with pytest.raises(ValueError, match="keypoint target_labels only support arrow"):
        import_predictions_for_benchmark(
            store_root=tmp_path,
            run_id="bad_keypoint_import",
            benchmark_id="bench1",
            prediction_root=prediction_root,
            task="keypoint",
            model_id="external-model",
            target_labels=["icon"],
        )


def test_import_predictions_rejects_unknown_target_labels(tmp_path: Path) -> None:
    _write_benchmark(tmp_path)
    manifest_path = tmp_path / "benchmarks" / "bench1" / "benchmark.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labels"] = ["arrow", "icon"]
    _write_json(manifest_path, manifest)
    prediction_root = tmp_path / "external_predictions"
    prediction_root.mkdir()

    with pytest.raises(
        ValueError,
        match="target_labels not found in benchmark label index: arrwo",
    ):
        import_predictions_for_benchmark(
            store_root=tmp_path,
            run_id="bad_label_import",
            benchmark_id="bench1",
            prediction_root=prediction_root,
            task="detection",
            model_id="external-model",
            target_labels=["arrwo"],
        )
    assert not (tmp_path / "runs" / "bad_label_import").exists()


def test_import_predictions_supports_flat_basename_lookup(tmp_path: Path) -> None:
    _write_benchmark(tmp_path)
    prediction_root = tmp_path / "external_predictions"
    _write_json(
        prediction_root / "a.json",
        {
            "instances": [{"label": "icon", "bbox": [0, 0, 100, 100]}],
        },
    )

    result = import_predictions_for_benchmark(
        store_root=tmp_path,
        run_id="flat_import",
        benchmark_id="bench1",
        prediction_root=prediction_root,
        task="detection",
        model_id="external-model",
    )

    prediction = json.loads(
        (tmp_path / "runs" / "flat_import" / "predictions" / "part1" / "json" / "a.json")
        .read_text(encoding="utf-8")
    )
    assert prediction["image"] == "part1/images/a.png"
    assert result.imported_predictions == 1


def test_import_predictions_strict_mode_fails_on_missing_files(tmp_path: Path) -> None:
    _write_benchmark(tmp_path)
    prediction_root = tmp_path / "external_predictions"
    prediction_root.mkdir()

    with pytest.raises(FileNotFoundError, match="missing 2 prediction files"):
        import_predictions_for_benchmark(
            store_root=tmp_path,
            run_id="strict_import",
            benchmark_id="bench1",
            prediction_root=prediction_root,
            task="detection",
            model_id="external-model",
            strict=True,
        )
