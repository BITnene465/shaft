from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

from shaft.plugins import Registry

from .record_store import ShaftRecordSubset

OfflineTransform = Callable[[Sequence[Any]], Sequence[Any]]
OnlineTransform = Callable[[dict[str, Any]], dict[str, Any]]
_PLANNING_POLICY_ATTRIBUTE = "__shaft_planning_policy__"

OFFLINE_TRANSFORM_REGISTRY: Registry[OfflineTransform] = Registry("offline_transform")
ONLINE_TRANSFORM_REGISTRY: Registry[OnlineTransform] = Registry("online_transform")


@dataclass(frozen=True, slots=True)
class ShaftOnlineTransformPlanningPolicy:
    fingerprint: str
    deterministic_from_context: bool = True
    preserves_image_identity: bool = True
    preserves_image_geometry: bool = True
    preserves_media_placeholders: bool = True

    @property
    def planning_safe(self) -> bool:
        return bool(
            self.fingerprint
            and self.deterministic_from_context
            and self.preserves_image_identity
            and self.preserves_image_geometry
            and self.preserves_media_placeholders
        )


def planning_safe_online_transform(
    transform: OnlineTransform | None = None,
    *,
    fingerprint: str | None = None,
    deterministic_from_context: bool = True,
    preserves_image_identity: bool = True,
    preserves_image_geometry: bool = True,
    preserves_media_placeholders: bool = True,
):
    """Declare deterministic, media-identity/geometry/placeholder preserving behavior."""

    def _decorate(target: OnlineTransform) -> OnlineTransform:
        resolved_fingerprint = str(fingerprint or "").strip()
        if not resolved_fingerprint:
            raise ValueError(
                "planning_safe_online_transform requires an explicit stable fingerprint."
            )
        setattr(
            target,
            _PLANNING_POLICY_ATTRIBUTE,
            ShaftOnlineTransformPlanningPolicy(
                fingerprint=resolved_fingerprint,
                deterministic_from_context=deterministic_from_context,
                preserves_image_identity=preserves_image_identity,
                preserves_image_geometry=preserves_image_geometry,
                preserves_media_placeholders=preserves_media_placeholders,
            ),
        )
        return target

    if transform is None:
        return _decorate
    return _decorate(transform)


def is_planning_safe_online_transform(transform: OnlineTransform) -> bool:
    policy = getattr(transform, _PLANNING_POLICY_ATTRIBUTE, None)
    return isinstance(policy, ShaftOnlineTransformPlanningPolicy) and policy.planning_safe


def planning_online_transform_fingerprint(transform: OnlineTransform) -> str:
    policy = getattr(transform, _PLANNING_POLICY_ATTRIBUTE, None)
    if not isinstance(policy, ShaftOnlineTransformPlanningPolicy) or not policy.planning_safe:
        raise ValueError("Online transform has no planning-safe policy.")
    return policy.fingerprint


@OFFLINE_TRANSFORM_REGISTRY.register("identity")
def offline_identity(records: Sequence[Any]) -> Sequence[Any]:
    return records


@OFFLINE_TRANSFORM_REGISTRY.register("dedup_image_target")
def offline_dedup_image_target(records: Sequence[Any]) -> Sequence[Any]:
    seen: set[tuple[str, str, str]] = set()
    indices: list[int] = []
    for index, item in enumerate(records):
        target_text = getattr(item, "target_text", None)
        if target_text is None:
            indices.append(index)
            continue
        key = (
            repr(tuple(getattr(item, "image_paths", ()) or ())),
            str(getattr(item, "target_reasoning_content", None)),
            str(target_text),
        )
        if key in seen:
            continue
        seen.add(key)
        indices.append(index)
    return ShaftRecordSubset(records, indices)


@ONLINE_TRANSFORM_REGISTRY.register("identity")
@planning_safe_online_transform(fingerprint="shaft-online-identity-v1")
def online_identity(sample: dict[str, Any]) -> dict[str, Any]:
    return sample


def build_offline_pipeline(transform_names: list[str]) -> OfflineTransform:
    transforms = [
        OFFLINE_TRANSFORM_REGISTRY.get(name)
        for name in (transform_names or ["identity"])
    ]

    def _run(records: Sequence[Any]) -> Sequence[Any]:
        out = records
        for fn in transforms:
            out = fn(out)
        return out

    return _run


def build_online_pipeline(transform_names: list[str]) -> OnlineTransform:
    transforms = [
        ONLINE_TRANSFORM_REGISTRY.get(name)
        for name in (transform_names or ["identity"])
    ]

    def _run(sample: dict[str, Any]) -> dict[str, Any]:
        out = sample
        for fn in transforms:
            out = fn(out)
        return out

    if all(is_planning_safe_online_transform(transform) for transform in transforms):
        component_fingerprints = tuple(
            planning_online_transform_fingerprint(transform) for transform in transforms
        )
        planning_safe_online_transform(
            _run,
            fingerprint=hashlib.sha256(
                repr(("shaft-online-pipeline-v1", component_fingerprints)).encode("utf-8")
            ).hexdigest(),
        )
    return _run
