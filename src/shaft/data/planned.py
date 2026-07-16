from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shaft.utils.contract_schema import (
    json_int,
    json_string,
    require_exact_keys,
    require_json_mapping,
)

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
        role = "ShaftBatchContext"
        payload = require_json_mapping(payload, role=role)
        require_exact_keys(
            payload,
            role=role,
            expected=frozenset(
                {
                    "global_microstep",
                    "plan_fingerprint",
                    "local_batch_id",
                    "pack_index",
                    "segment_index",
                    "pack_segment_count",
                }
            ),
        )
        return cls(
            global_microstep=json_int(payload, "global_microstep", role=role),
            plan_fingerprint=json_string(payload, "plan_fingerprint", role=role),
            local_batch_id=json_int(payload, "local_batch_id", role=role),
            pack_index=json_int(payload, "pack_index", role=role),
            segment_index=json_int(payload, "segment_index", role=role),
            pack_segment_count=json_int(payload, "pack_segment_count", role=role),
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
