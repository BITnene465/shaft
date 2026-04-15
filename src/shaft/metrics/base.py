from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from shaft.codec import ShaftCodecResult


class ShaftEvalMetric(ABC):
    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    @abstractmethod
    def update(
        self,
        *,
        prediction: ShaftCodecResult,
        target: Any,
        sample_meta: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
