from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

from shaft.config import PromptSamplingConfig
from shaft.plugins import Registry
from shaft.prompting import ShaftPromptTemplate, load_prompt_pool

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
        resolved_fingerprint = str(fingerprint or "").strip() or (
            f"{getattr(target, '__module__', '')}.{getattr(target, '__qualname__', '')}"
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
        fingerprint_payload = (
            "shaft-prompt-sampling-v1",
            self.seed,
            tuple(
                (
                    dataset_name,
                    tuple(
                        (
                            variant.prompt_id,
                            variant.variant_id,
                            variant.version,
                            variant.system_prompt,
                            variant.user_prompt,
                            variant.sampling_weight,
                            variant.source_path,
                        )
                        for variant in variants
                    ),
                )
                for dataset_name, variants in sorted(self.pools.items())
            ),
        )
        setattr(
            self,
            _PLANNING_POLICY_ATTRIBUTE,
            ShaftOnlineTransformPlanningPolicy(
                fingerprint=hashlib.sha256(
                    repr(fingerprint_payload).encode("utf-8")
                ).hexdigest()
            ),
        )

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
