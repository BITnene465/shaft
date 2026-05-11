from __future__ import annotations

import json
import time
from pathlib import Path

from eval_bench.evaluator import evaluate_run
from eval_bench.store import EvalBenchStore


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dashboard_state_scan_stays_manifest_level(tmp_path: Path) -> None:
    for index in range(160):
        run_id = f"run_{index:04d}"
        _write_json(
            tmp_path / "runs" / run_id / "run.json",
            {
                "run_id": run_id,
                "status": "succeeded",
                "created_at": "2026-05-09T00:00:00Z",
                "model": {"model_id": "model-a", "path": "outputs/model-a/best"},
                "benchmark": {
                    "benchmark_id": "multitask_val_v1",
                    "root": "eval_bench_store/benchmarks/multitask_val_v1/data",
                    "split": "val",
                    "tasks": ["detection", "keypoint"],
                },
                "spec": {"task": "detection"},
            },
        )

    store = EvalBenchStore(tmp_path)
    start = time.perf_counter()
    state = store.state()
    elapsed = time.perf_counter() - start

    assert state.run_count == 160
    assert elapsed < 1.0


def test_evaluator_sample_diagnostics_stays_linear(tmp_path: Path) -> None:
    sample_count = 200
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(
        "".join(f"part1/json/sample_{index:04d}.json\n" for index in range(sample_count)),
        encoding="utf-8",
    )
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    for index in range(sample_count):
        _write_json(
            data_root / "part1" / "json" / f"sample_{index:04d}.json",
            {
                "image_path": f"part1/images/sample_{index:04d}.png",
                "instances": [
                    {"label": "icon", "bbox": [0, 0, 10, 10]},
                    {"label": "shape", "bbox": [20, 20, 40, 40]},
                    {"label": "arrow", "bbox": [50, 50, 80, 60]},
                ],
            },
        )
        _write_json(
            tmp_path / "runs" / "run_perf" / "predictions" / "part1" / "json" / f"sample_{index:04d}.json",
            {
                "image": f"part1/images/sample_{index:04d}.png",
                "instances": [
                    {"label": "icon", "bbox": [0, 0, 10, 10]},
                    {"label": "shape", "bbox": [22, 22, 42, 42]},
                    {"label": "arrow", "bbox": [90, 90, 100, 100]},
                ],
                "metadata": {},
            },
        )
    _write_json(
        tmp_path / "runs" / "run_perf" / "run.json",
        {
            "run_id": "run_perf",
            "status": "succeeded",
            "model": {"model_id": "model-a", "path": "outputs/model-a/best"},
            "benchmark": {
                "benchmark_id": "bench1",
                "root": str(data_root),
                "split": "val",
                "tasks": ["detection"],
                "manifest_path": str(split_path),
            },
            "spec": {"task": "detection"},
        },
    )

    start = time.perf_counter()
    report_path = evaluate_run(store_root=tmp_path, run_id="run_perf")
    elapsed = time.perf_counter() - start
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["sample_count"] == sample_count
    assert len(report["samples"]) == sample_count
    assert report["samples"][0]["false_positive_count"] == 1
    assert elapsed < 3.0
