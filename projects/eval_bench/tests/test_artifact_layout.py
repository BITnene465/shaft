from __future__ import annotations

from pathlib import Path

from eval_bench.artifacts import DEFAULT_STORE_ROOT, RunArtifacts, StoreLayout


def test_prediction_path_keeps_raw_data_part_layout(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path, "run1")
    path = artifacts.prediction_path("part2/images/pic001.png")
    assert path == tmp_path / "runs" / "run1" / "predictions" / "part2" / "json" / "pic001.json"


def test_store_layout_keeps_db_and_runs_outside_outputs() -> None:
    layout = StoreLayout()
    assert DEFAULT_STORE_ROOT == Path("eval_bench_store")
    assert layout.db_path == Path("eval_bench_store/db/eval_bench.sqlite")
    assert layout.runs_dir == Path("eval_bench_store/runs")
    assert layout.benchmarks_dir == Path("eval_bench_store/benchmarks")
    assert layout.suites_dir == Path("eval_bench_store/suites")
    assert layout.campaigns_dir == Path("eval_bench_store/campaigns")
