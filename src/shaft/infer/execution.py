from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time


class ShaftInferExecutionControlUnsupportedError(RuntimeError):
    """Raised before work starts when an adapter cannot honor execution control."""


class ShaftInferCancelledError(RuntimeError):
    """Raised when cooperative inference cancellation is requested."""


@dataclass(frozen=True, slots=True)
class ShaftInferExecutionControl:
    """Absolute deadline and cooperative cancellation contract for one request."""

    deadline_monotonic: float | None = None
    cancellation_event: threading.Event | None = None

    def __post_init__(self) -> None:
        if self.deadline_monotonic is not None:
            deadline = float(self.deadline_monotonic)
            if not math.isfinite(deadline):
                raise ValueError("deadline_monotonic must be finite.")
            object.__setattr__(self, "deadline_monotonic", deadline)

    @property
    def requires_deadline(self) -> bool:
        return self.deadline_monotonic is not None

    @property
    def requires_cancellation(self) -> bool:
        return self.cancellation_event is not None

    def remaining_seconds(self, *, now: float | None = None) -> float | None:
        if self.deadline_monotonic is None:
            return None
        current = time.monotonic() if now is None else float(now)
        return max(self.deadline_monotonic - current, 0.0)

    def checkpoint(self, *, context: str = "Inference request") -> None:
        if self.cancellation_event is not None and self.cancellation_event.is_set():
            raise ShaftInferCancelledError(f"{context} was cancelled.")
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise TimeoutError(f"{context} deadline expired.")


@dataclass(frozen=True, slots=True)
class ShaftInferAdapterCapabilities:
    supports_deadline: bool = False
    supports_cancellation: bool = False
