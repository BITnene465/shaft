from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mixing import ShaftSampleContext, ShaftSampleRef


@dataclass(frozen=True, slots=True)
class ShaftBatchContext:
    """Ephemeral identity carried through Dataset workers for one planned segment."""

    global_microstep: int
    plan_fingerprint: str
    local_batch_id: int
    pack_index: int
    segment_index: int
    pack_segment_count: int

    def __post_init__(self) -> None:
        for name in ("global_microstep", "local_batch_id", "pack_index", "segment_index"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"ShaftBatchContext.{name} must be >= 0.")
        if not str(self.plan_fingerprint).strip():
            raise ValueError("ShaftBatchContext.plan_fingerprint must not be empty.")
        if int(self.pack_segment_count) <= 0:
            raise ValueError("ShaftBatchContext.pack_segment_count must be > 0.")
        if int(self.segment_index) >= int(self.pack_segment_count):
            raise ValueError(
                "ShaftBatchContext.segment_index must be lower than pack_segment_count."
            )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "global_microstep": int(self.global_microstep),
            "plan_fingerprint": str(self.plan_fingerprint),
            "local_batch_id": int(self.local_batch_id),
            "pack_index": int(self.pack_index),
            "segment_index": int(self.segment_index),
            "pack_segment_count": int(self.pack_segment_count),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShaftBatchContext":
        if not isinstance(payload, dict):
            raise TypeError("ShaftBatchContext payload must be a mapping.")
        return cls(
            global_microstep=int(payload["global_microstep"]),
            plan_fingerprint=str(payload["plan_fingerprint"]),
            local_batch_id=int(payload["local_batch_id"]),
            pack_index=int(payload["pack_index"]),
            segment_index=int(payload["segment_index"]),
            pack_segment_count=int(payload["pack_segment_count"]),
        )


@dataclass(frozen=True, slots=True)
class ShaftPlannedSampleRef:
    """A SampleRef plus immutable physical-plan placement metadata."""

    sample_ref: ShaftSampleRef
    batch_context: ShaftBatchContext

    @property
    def dataset_name(self) -> str:
        return self.sample_ref.dataset_name

    @property
    def row_index(self) -> int:
        return int(self.sample_ref.row_index)

    @property
    def context(self) -> ShaftSampleContext:
        return self.sample_ref.context
