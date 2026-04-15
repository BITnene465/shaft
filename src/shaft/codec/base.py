from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ShaftCodecResult:
    raw_text: str
    parsed: Any | None
    valid: bool
    partial: bool
    error_type: str | None
    error: str | None
