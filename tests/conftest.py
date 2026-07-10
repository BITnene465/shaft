from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest


_TEST_ROOT = Path(__file__).resolve().parent

# This manifest is the single source of truth for framework test-suite membership.
# Every present test file must appear in exactly one leaf suite. Pytest validates the
# invariant before collection, so a new unclassified file cannot silently enter required CI.
_SUITE_FILES: dict[str, tuple[str, ...]] = {
    "framework": (
        "test_checkpointing.py",
        "test_batch_planning.py",
        "test_cli_commands.py",
        "test_cli_common.py",
        "test_codec.py",
        "test_collator.py",
        "test_config_catalog.py",
        "test_config_deepspeed.py",
        "test_config_eval_datasets.py",
        "test_config_examples.py",
        "test_config_freeze.py",
        "test_config_loader.py",
        "test_config_online_eval.py",
        "test_config_online_eval_best_metric.py",
        "test_config_online_eval_validation.py",
        "test_config_prompt_sampling.py",
        "test_config_rlhf.py",
        "test_config_validation.py",
        "test_data_center.py",
        "test_data_meta.py",
        "test_data_sources.py",
        "test_distributed_runtime.py",
        "test_eval_metrics.py",
        "test_export_cli.py",
        "test_export_tools.py",
        "test_finetune_modes.py",
        "test_finetune_plan.py",
        "test_grpo_dataset.py",
        "test_hooks.py",
        "test_infer_cli.py",
        "test_infer_engine.py",
        "test_infer_loader.py",
        "test_infer_pipeline.py",
        "test_interceptors.py",
        "test_mixing.py",
        "test_model_adapter_checkpoint.py",
        "test_model_builder_validation.py",
        "test_model_freeze.py",
        "test_model_meta.py",
        "test_model_processor_policy.py",
        "test_model_registry.py",
        "test_online_eval_aggregation.py",
        "test_online_eval_runner.py",
        "test_pipeline_registry.py",
        "test_pipeline_rlhf.py",
        "test_pipeline_sft.py",
        "test_pipeline_training_args.py",
        "test_pixel_budget.py",
        "test_progress_callback.py",
        "test_prompting.py",
        "test_qwen_coordinates.py",
        "test_registry.py",
        "test_rlhf_trl_configs.py",
        "test_rlhf_utils.py",
        "test_sft_trainer.py",
        "test_template_registry.py",
        "test_template_supervision.py",
        "test_training_loss.py",
        "test_training_optimizer.py",
        "test_training_topology.py",
        "test_transforms.py",
        "test_webui_app.py",
        "test_webui_cli.py",
        "test_webui_config_service.py",
        "test_webui_controller.py",
        "test_webui_train_service.py",
    ),
    "smoke": (
        "test_pipeline_rlhf_smoke.py",
        "test_smoke_train_modes.py",
    ),
    "distributed": (
        "test_pipeline_distributed_contract.py",
        "test_smoke_distributed.py",
    ),
    "integration": (
        "test_integration_infer_pipeline.py",
        "test_integration_qwen_standard.py",
    ),
    "gpu": ("test_flash_attn_smoke.py",),
    "task": (
        "test_build_grounding_structured.py",
        "test_build_sft_from_structured.py",
    ),
    "visual": ("test_prediction_visualization.py",),
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
        raise pytest.UsageError("Invalid framework test suite manifest; " + "; ".join(details))


def _explicit_test_file_requested(config: pytest.Config) -> bool:
    for argument in config.args:
        path_text = str(argument).split("::", maxsplit=1)[0]
        if Path(path_text).name.startswith("test_") and path_text.endswith(".py"):
            return True
    return False


def _selected_suite(config: pytest.Config) -> str:
    suite = config.getoption("shaft_suite")
    if suite:
        return str(suite)
    if config.getoption("markexpr") or _explicit_test_file_requested(config):
        return "all"
    return "framework"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--suite",
        action="store",
        choices=(*_SUITE_FILES, "all"),
        dest="shaft_suite",
        help="Collect one explicitly classified Shaft framework test suite.",
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


@pytest.fixture
def repo_root() -> Path:
    return _TEST_ROOT.parent
