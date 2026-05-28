from __future__ import annotations

from collections.abc import Callable
import hashlib
from typing import Any

from shaft.config import PromptSamplingConfig
from shaft.plugins import Registry
from shaft.prompting import ShaftPromptTemplate, load_prompt_pool

OfflineTransform = Callable[[list[Any]], list[Any]]
OnlineTransform = Callable[[dict[str, Any]], dict[str, Any]]

OFFLINE_TRANSFORM_REGISTRY: Registry[OfflineTransform] = Registry("offline_transform")
ONLINE_TRANSFORM_REGISTRY: Registry[OnlineTransform] = Registry("online_transform")


class PromptSamplingTransform:
    """Deterministically sample an equivalent prompt variant at training runtime."""

    def __init__(
        self,
        *,
        pools: dict[str, list[ShaftPromptTemplate]],
        seed: int,
    ) -> None:
        self.pools = {str(name): list(variants) for name, variants in pools.items()}
        self.seed = int(seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        dataset_name = str(sample.get("dataset_name", "")).strip()
        variants = self.pools.get(dataset_name)
        if not variants:
            return sample
        if sample.get("messages"):
            updated = dict(sample)
            extra = dict(updated.get("extra", {}))
            extra["prompt_sampling_skipped"] = "messages_present"
            updated["extra"] = extra
            return updated

        epoch = int(sample.get("_epoch", 0) or 0)
        sample_id = str(sample.get("sample_id") or sample.get("image_path") or "").strip()
        key = f"{self.seed}\n{epoch}\n{dataset_name}\n{sample_id}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        variant = variants[int(digest, 16) % len(variants)]

        updated = dict(sample)
        extra = dict(updated.get("extra", {}))
        extra["runtime_prompt_id"] = variant.prompt_id
        extra["runtime_prompt_version"] = variant.version or ""
        extra["runtime_prompt_source"] = variant.source_path
        extra["runtime_prompt_epoch"] = epoch
        updated["extra"] = extra
        updated["system_prompt"] = variant.system_prompt
        updated["user_prompt"] = variant.user_prompt
        return updated


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


def build_prompt_sampling_transform(
    config: PromptSamplingConfig,
    *,
    default_seed: int = 42,
) -> PromptSamplingTransform | None:
    if not config.enabled:
        return None
    pools = {
        dataset_name: load_prompt_pool(path)
        for dataset_name, path in config.pools.items()
    }
    return PromptSamplingTransform(
        pools=pools,
        seed=int(default_seed) if config.seed is None else int(config.seed),
    )
