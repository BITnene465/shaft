from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from transformers import Trainer


@dataclass
class AlgorithmContext:
    params: dict[str, Any]


class Algorithm(Protocol):
    name: str

    def build_trainer(self, *, context: AlgorithmContext, **kwargs: Any) -> Trainer: ...

