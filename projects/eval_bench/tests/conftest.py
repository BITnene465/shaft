from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest


_TEST_ROOT = Path(__file__).resolve().parent

# Eval Bench is not a required Shaft framework gate, but its local suites still need
# deterministic membership and collection-time dependency isolation.
_SUITE_FILES: dict[str, tuple[str, ...]] = {
    "backend": (
        "test_artifact_layout.py",
        "test_benchmark_creation.py",
        "test_benchmark_split_resolution.py",
        "test_cli_detail_commands.py",
        "test_cli_import_predictions.py",
        "test_cli_init_run.py",
        "test_cli_jobs_manifest.py",
        "test_cli_listing.py",
        "test_cli_ops_summary.py",
        "test_cli_parser_contract.py",
        "test_cli_rank_board.py",
        "test_cli_run_admin.py",
        "test_cli_run_eval_compare.py",
        "test_cli_run_notes.py",
        "test_cli_services.py",
        "test_cli_target_labels.py",
        "test_cli_templates.py",
        "test_comparison.py",
        "test_database.py",
        "test_eval_manifests.py",
        "test_eval_semantics.py",
        "test_evaluator_run.py",
        "test_inference_schema.py",
        "test_job_spec_manifest.py",
        "test_job_spec_preflight.py",
        "test_job_spec_rejections.py",
        "test_job_spec_runtime.py",
        "test_metric_engine.py",
        "test_orchestrator.py",
        "test_prediction_import.py",
        "test_prediction_parser.py",
        "test_prediction_schema.py",
        "test_runtime_resources.py",
        "test_services.py",
        "test_vllm_adapter.py",
        "test_worker_jobs.py",
    ),
    "visual": (
        "test_dashboard_benchmarks.py",
        "test_dashboard_composite_samples.py",
        "test_dashboard_import_compare.py",
        "test_dashboard_jobs.py",
        "test_dashboard_overview.py",
        "test_dashboard_rank_suite.py",
        "test_dashboard_run_sample_detail.py",
        "test_dashboard_sample_image_urls.py",
        "test_dashboard_services.py",
    ),
    "performance": (
        "test_perf.py",
        "test_store_performance.py",
    ),
    "runtime": (
        "test_inference_docker_contract.py",
        "test_worker_runtime.py",
        "test_worker_vllm.py",
    ),
}


def _validate_suite_manifest() -> None:
    discovered = {
        path.relative_to(_TEST_ROOT).as_posix()
        for path in _TEST_ROOT.rglob("test_*.py")
        if path.is_file()
    }
    declared = [path for paths in _SUITE_FILES.values() for path in paths]
    duplicates = sorted(path for path, count in Counter(declared).items() if count > 1)
    unclassified = sorted(discovered - set(declared))
    if duplicates or unclassified:
        details = []
        if duplicates:
            details.append(f"declared in multiple suites: {duplicates}")
        if unclassified:
            details.append(f"unclassified: {unclassified}")
        raise pytest.UsageError("Invalid Eval Bench test suite manifest; " + "; ".join(details))


def _explicit_test_file_requested(config: pytest.Config) -> bool:
    for argument in config.args:
        path_text = str(argument).split("::", maxsplit=1)[0]
        if Path(path_text).name.startswith("test_") and path_text.endswith(".py"):
            return True
    return False


def _selected_suite(config: pytest.Config) -> str:
    suite = config.getoption("eval_bench_suite")
    if suite:
        return str(suite)
    if config.getoption("markexpr") or _explicit_test_file_requested(config):
        return "all"
    return "backend"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--eval-bench-suite",
        action="store",
        choices=(*_SUITE_FILES, "all"),
        help="Collect one explicitly classified Eval Bench test suite.",
    )


def pytest_configure(config: pytest.Config) -> None:
    _validate_suite_manifest()


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if collection_path.suffix != ".py" or not collection_path.name.startswith("test_"):
        return None
    try:
        relative_path = collection_path.resolve().relative_to(_TEST_ROOT).as_posix()
    except ValueError:
        return None

    suite = _selected_suite(config)
    if suite == "all":
        return None
    return relative_path not in set(_SUITE_FILES[suite])
