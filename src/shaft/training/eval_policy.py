from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from shaft.config.training import EvalConfig, EvalInputPolicy, EvalPixelBudget
from shaft.training.distributed import is_rank_zero

logger = logging.getLogger(__name__)


def _format_pixel_budget(budget: EvalPixelBudget) -> str:
    min_pixels = "none" if budget.min_pixels is None else str(int(budget.min_pixels))
    max_pixels = "none" if budget.max_pixels is None else str(int(budget.max_pixels))
    return f"{min_pixels}:{max_pixels}"


def log_eval_input_policy(
    *,
    policy: EvalInputPolicy,
    model_adapter: Any,
) -> None:
    if not is_rank_zero():
        return
    dataset_summary = ",".join(
        f"{dataset_name}={_format_pixel_budget(budget)}"
        for dataset_name, budget in policy.dataset_pixel_budgets
    ) or "none"
    logger.info(
        "[eval-input] default=%s datasets=%s training_padding=%s generation_padding=%s",
        _format_pixel_budget(policy.default_pixel_budget),
        dataset_summary,
        model_adapter.resolve_processor_padding_side("training"),
        model_adapter.resolve_processor_padding_side("generation"),
    )


def aggregate_weighted_dataset_values(
    *,
    values_by_dataset: Mapping[str, float],
    eval_config: EvalConfig,
    metric_name: str,
) -> float | None:
    weighted_sum = 0.0
    total_weight = 0.0
    for dataset_name in sorted(eval_config.datasets):
        if dataset_name not in values_by_dataset:
            continue
        weight = float(eval_config.datasets[dataset_name].weight)
        weighted_sum += float(values_by_dataset[dataset_name]) * weight
        total_weight += weight
    if total_weight <= 0:
        logger.warning("[eval] no dataset produced %s; weighted aggregation defaults to empty", metric_name)
        return None
    return weighted_sum / total_weight
