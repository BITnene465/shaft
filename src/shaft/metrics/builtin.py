from __future__ import annotations

from typing import Any

from shaft.codec import ShaftCodecResult

from .base import ShaftEvalMetric
from .registry import register_eval_metric


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, str):
        return value.strip()
    return value


@register_eval_metric("parse_success")
class ParseSuccessMetric(ShaftEvalMetric):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.values: list[float] = []

    def update(
        self,
        *,
        prediction: ShaftCodecResult,
        target: Any,
        sample_meta: dict[str, Any],
    ) -> None:
        _ = target, sample_meta
        self.values.append(1.0 if prediction.valid else 0.0)

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    def reset(self) -> None:
        self.values = []


@register_eval_metric("exact_match")
class ExactMatchMetric(ShaftEvalMetric):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.values: list[float] = []

    def update(
        self,
        *,
        prediction: ShaftCodecResult,
        target: Any,
        sample_meta: dict[str, Any],
    ) -> None:
        _ = sample_meta
        if not prediction.valid:
            self.values.append(0.0)
            return
        prediction_value = prediction.parsed if prediction.parsed is not None else prediction.raw_text
        self.values.append(1.0 if _normalize_value(prediction_value) == _normalize_value(target) else 0.0)

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    def reset(self) -> None:
        self.values = []
