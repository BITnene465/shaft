from __future__ import annotations

from collections.abc import Callable
from typing import Any

from shaft.plugins import Registry

from .dataset import SFTRecord

OfflineTransform = Callable[[list[SFTRecord]], list[SFTRecord]]
OnlineTransform = Callable[[dict[str, Any]], dict[str, Any]]

OFFLINE_TRANSFORM_REGISTRY: Registry[OfflineTransform] = Registry("offline_transform")
ONLINE_TRANSFORM_REGISTRY: Registry[OnlineTransform] = Registry("online_transform")


@OFFLINE_TRANSFORM_REGISTRY.register("identity")
def offline_identity(records: list[SFTRecord]) -> list[SFTRecord]:
    return records


@OFFLINE_TRANSFORM_REGISTRY.register("dedup_image_target")
def offline_dedup_image_target(records: list[SFTRecord]) -> list[SFTRecord]:
    seen: set[tuple[str, str]] = set()
    filtered: list[SFTRecord] = []
    for item in records:
        key = (item.image_path, item.target_text)
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

    def _run(records: list[SFTRecord]) -> list[SFTRecord]:
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
