from __future__ import annotations

import logging
from collections.abc import Mapping

from shaft.config.training import EvalConfig

logger = logging.getLogger(__name__)


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
