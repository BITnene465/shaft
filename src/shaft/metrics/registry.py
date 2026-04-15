from __future__ import annotations

from shaft.plugins import Registry

from .base import ShaftEvalMetric

EVAL_METRIC_REGISTRY: Registry = Registry("eval_metric")


def register_eval_metric(name: str):
    return EVAL_METRIC_REGISTRY.register(str(name).strip().lower())


def build_eval_metric(name: str, *, params: dict | None = None) -> ShaftEvalMetric:
    metric_name = str(name).strip().lower()
    metric_cls = EVAL_METRIC_REGISTRY.get(metric_name)
    return metric_cls(**dict(params or {}))
