from __future__ import annotations

from collections.abc import Callable
from typing import Any

from shaft.plugins import Registry

OfflineTransform = Callable[[list[Any]], list[Any]]
OnlineTransform = Callable[[dict[str, Any]], dict[str, Any]]

OFFLINE_TRANSFORM_REGISTRY: Registry[OfflineTransform] = Registry("offline_transform")
ONLINE_TRANSFORM_REGISTRY: Registry[OnlineTransform] = Registry("online_transform")


@OFFLINE_TRANSFORM_REGISTRY.register("identity")
def offline_identity(records: list[Any]) -> list[Any]:
    return records


@OFFLINE_TRANSFORM_REGISTRY.register("dedup_image_target")
def offline_dedup_image_target(records: list[Any]) -> list[Any]:
    seen: set[tuple[str, str]] = set()
    filtered: list[Any] = []
    for item in records:
        target_text = getattr(item, "target_text", None)
        if target_text is None:
            filtered.append(item)
            continue
        key = (str(getattr(item, "image_path", "")), str(target_text))
        if key in seen:
            continue
        seen.add(key)
        filtered.append(item)
    return filtered


@ONLINE_TRANSFORM_REGISTRY.register("identity")
def online_identity(sample: dict[str, Any]) -> dict[str, Any]:
    return sample


def build_offline_pipeline(transform_names: list[str]) -> OfflineTransform:
    transforms = [
        OFFLINE_TRANSFORM_REGISTRY.get(name)
        for name in (transform_names or ["identity"])
    ]

    def _run(records: list[Any]) -> list[Any]:
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

    return _run
