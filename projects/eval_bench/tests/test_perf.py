from __future__ import annotations

import json
from pathlib import Path

from eval_bench.perf import run_perf_smoke


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_perf_smoke_reports_common_store_paths(tmp_path: Path) -> None:
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    split_manifest = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    split_manifest.parent.mkdir(parents=True)
    split_manifest.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["layout"],
            "split": "val",
            "sample_count": 1,
            "root": str(data_root),
            "manifest_path": str(split_manifest),
        },
    )
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {"image_path": "part1/images/a.png", "instances": []},
    )

    report = run_perf_smoke(store_root=tmp_path, iterations=2, sample_limit=1)

    assert report["benchmark_count"] == 1
    assert report["iterations"] == 2
    assert report["measurements"]["state_ms"]["max_ms"] >= 0.0
    assert report["measurements"]["benchmark_samples_ms"]["max_ms"] >= 0.0
