from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from .artifacts import DEFAULT_STORE_ROOT
from .comparison import list_comparison_reports
from .store import EvalBenchStore


def run_perf_smoke(
    *,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    iterations: int = 5,
    sample_limit: int = 500,
) -> dict[str, Any]:
    iteration_count = max(1, int(iterations))
    limit = max(1, int(sample_limit))
    store = EvalBenchStore(store_root)
    state = store.state()
    first_benchmark = state.benchmarks[0].benchmark_id if state.benchmarks else None
    first_run = state.runs[0].run_id if state.runs else None
    measurements: dict[str, list[float]] = {
        "state_ms": [],
        "comparisons_ms": [],
    }
    if first_benchmark:
        measurements["benchmark_samples_ms"] = []
    if first_run:
        measurements["run_samples_ms"] = []

    for _ in range(iteration_count):
        measurements["state_ms"].append(_measure_ms(store.state))
        measurements["comparisons_ms"].append(
            _measure_ms(lambda: list_comparison_reports(store_root=store_root))
        )
        if first_benchmark:
            measurements["benchmark_samples_ms"].append(
                _measure_ms(
                    lambda benchmark_id=first_benchmark: store.benchmark_samples(
                        benchmark_id,
                        limit=limit,
                    )
                )
            )
        if first_run:
            measurements["run_samples_ms"].append(
                _measure_ms(lambda run_id=first_run: store.run_samples(run_id, limit=limit))
            )

    return {
        "store_root": str(Path(store_root)),
        "iterations": iteration_count,
        "sample_limit": limit,
        "benchmark_count": state.benchmark_count,
        "run_count": state.run_count,
        "prediction_count": state.prediction_count,
        "first_benchmark": first_benchmark,
        "first_run": first_run,
        "measurements": {
            key: _summarize(values) for key, values in measurements.items() if values
        },
    }


def _measure_ms(fn: Callable[[], Any]) -> float:
    start = perf_counter()
    fn()
    return (perf_counter() - start) * 1000.0


def _summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min_ms": ordered[0],
        "mean_ms": sum(ordered) / len(ordered),
        "max_ms": ordered[-1],
    }
