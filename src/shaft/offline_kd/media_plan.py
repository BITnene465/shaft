from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from typing import Any

from shaft.utils.qwen_pixel_budget import QWEN_IMAGE_FACTOR, smart_resize_qwen


MEDIA_PLAN_FIELD = "offline_kd_media_plan"
MEDIA_PLAN_VERSION = "shaft-offline-kd-media-plan-v1"


@dataclass(frozen=True, slots=True)
class OfflineKDMediaPlan:
    min_pixels: int
    max_pixels: int
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    bucket: str
    factor: int = QWEN_IMAGE_FACTOR

    def __post_init__(self) -> None:
        for field_name in (
            "min_pixels",
            "max_pixels",
            "source_width",
            "source_height",
            "target_width",
            "target_height",
            "factor",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Offline KD media plan {field_name} must be a positive integer.")
        if self.max_pixels < self.min_pixels:
            raise ValueError("Offline KD media plan max_pixels must be >= min_pixels.")
        if not str(self.bucket).strip():
            raise ValueError("Offline KD media plan bucket must not be empty.")
        expected = smart_resize_qwen(
            width=self.source_width,
            height=self.source_height,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            factor=self.factor,
        )
        if expected != (self.target_width, self.target_height):
            raise ValueError(
                "Offline KD media plan target dimensions differ from deterministic smart resize."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MEDIA_PLAN_VERSION,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "bucket": self.bucket,
            "factor": self.factor,
        }

    def validate_image_size(self, image: Any) -> None:
        raw_size = getattr(image, "size", None)
        if raw_size is None or tuple(raw_size) != (self.source_width, self.source_height):
            raise ValueError(
                "Offline KD media plan source dimensions differ from the decoded image: "
                f"plan={(self.source_width, self.source_height)}, observed={raw_size}."
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "OfflineKDMediaPlan":
        if not isinstance(value, Mapping):
            raise TypeError("Offline KD media plan must be a mapping.")
        payload = dict(value)
        expected = {
            "version",
            "min_pixels",
            "max_pixels",
            "source_width",
            "source_height",
            "target_width",
            "target_height",
            "bucket",
            "factor",
        }
        if set(payload) != expected:
            raise ValueError(
                "Offline KD media plan fields differ: "
                f"missing={sorted(expected - set(payload))} "
                f"unknown={sorted(set(payload) - expected)}."
            )
        if payload["version"] != MEDIA_PLAN_VERSION:
            raise ValueError(f"Unsupported Offline KD media plan {payload['version']!r}.")
        return cls(
            min_pixels=payload["min_pixels"],
            max_pixels=payload["max_pixels"],
            source_width=payload["source_width"],
            source_height=payload["source_height"],
            target_width=payload["target_width"],
            target_height=payload["target_height"],
            bucket=str(payload["bucket"]),
            factor=payload["factor"],
        )


def deterministic_detection_media_plan(
    *,
    sample_id: str,
    width: int,
    height: int,
    seed: int,
) -> OfflineKDMediaPlan:
    """Choose the frozen 25/50/25 log-uniform detection resolution plan."""

    digest = hashlib.sha256(f"{int(seed)}\n{sample_id}".encode("utf-8")).digest()
    bucket_draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    within_draw = int.from_bytes(digest[8:16], "big") / float(1 << 64)
    if bucket_draw < 0.25:
        bucket, lower, upper = "0.5m-1m", 500_000, 1_000_000
    elif bucket_draw < 0.75:
        bucket, lower, upper = "1m-2m", 1_000_000, 2_000_000
    else:
        bucket, lower, upper = "2m-4m", 2_000_000, 4_000_000
    budget = int(round(math.exp(math.log(lower) + within_draw * math.log(upper / lower))))
    target_width, target_height = smart_resize_qwen(
        width=width,
        height=height,
        min_pixels=budget,
        max_pixels=budget,
    )
    return OfflineKDMediaPlan(
        min_pixels=budget,
        max_pixels=budget,
        source_width=int(width),
        source_height=int(height),
        target_width=target_width,
        target_height=target_height,
        bucket=bucket,
    )


def media_plan_from_item(item: Mapping[str, Any]) -> OfflineKDMediaPlan | None:
    raw = item.get(MEDIA_PLAN_FIELD)
    if raw is None:
        extra = item.get("extra")
        if isinstance(extra, Mapping):
            raw = extra.get(MEDIA_PLAN_FIELD)
    return None if raw is None else OfflineKDMediaPlan.from_mapping(raw)
