from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
from typing import Any

from shaft.config import PromptSamplingConfig
from shaft.plugins import Registry
from shaft.prompting import ShaftPromptTemplate, load_prompt_pool

from .record_store import ShaftRecordSubset

OfflineTransform = Callable[[Sequence[Any]], Sequence[Any]]
OnlineTransform = Callable[[dict[str, Any]], dict[str, Any]]

OFFLINE_TRANSFORM_REGISTRY: Registry[OfflineTransform] = Registry("offline_transform")
ONLINE_TRANSFORM_REGISTRY: Registry[OnlineTransform] = Registry("online_transform")


class PromptSamplingTransform:
    """Sample one prompt per logical sample draw with deterministic weighted probability."""

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

        context = sample.get("_sample_context") or {}
        draw_id = int(context.get("draw_id", 0) or 0)
        transform_seed = int(context.get("transform_seed", 0) or 0)
        sample_id = str(sample.get("sample_id") or sample.get("image_path") or "").strip()
        key = f"{self.seed}\n{transform_seed}\n{dataset_name}\n{sample_id}\n{draw_id}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        max_weight = max(variant.sampling_weight for variant in variants)
        scaled_weights = [variant.sampling_weight / max_weight for variant in variants]
        total_weight = sum(scaled_weights)
        random_bits = int.from_bytes(digest[:8], "big") >> 11
        threshold = (random_bits / float(1 << 53)) * total_weight
        cumulative = 0.0
        variant = next(candidate for candidate in reversed(variants) if candidate.sampling_weight > 0)
        for candidate, scaled_weight in zip(variants, scaled_weights):
            cumulative += scaled_weight
            if threshold < cumulative:
                variant = candidate
                break

        updated = dict(sample)
        extra = dict(updated.get("extra", {}))
        extra["runtime_prompt_id"] = variant.prompt_id
        extra["runtime_prompt_version"] = variant.version or ""
        extra["runtime_prompt_source"] = variant.source_path
        extra["runtime_prompt_draw_id"] = draw_id
        extra["runtime_prompt_sampling_weight"] = variant.sampling_weight
        updated["extra"] = extra
        updated["system_prompt"] = variant.system_prompt
        updated["user_prompt"] = variant.user_prompt
        return updated


@OFFLINE_TRANSFORM_REGISTRY.register("identity")
def offline_identity(records: Sequence[Any]) -> Sequence[Any]:
    return records


@OFFLINE_TRANSFORM_REGISTRY.register("dedup_image_target")
def offline_dedup_image_target(records: Sequence[Any]) -> Sequence[Any]:
    seen: set[tuple[str, str]] = set()
    indices: list[int] = []
    for index, item in enumerate(records):
        target_text = getattr(item, "target_text", None)
        if target_text is None:
            indices.append(index)
            continue
        key = (str(getattr(item, "image_path", "")), str(target_text))
        if key in seen:
            continue
        seen.add(key)
        indices.append(index)
    return ShaftRecordSubset(records, indices)


@ONLINE_TRANSFORM_REGISTRY.register("identity")
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
