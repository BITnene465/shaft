from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ShaftPromptTemplate:
    prompt_id: str
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any]
    source_path: str
    variant_id: str | None = None
    version: str | None = None
    sampling_weight: float = 1.0


def load_prompt_template(path: str | Path, *, variant_id: str = "main") -> ShaftPromptTemplate:
    """Load a prompt from a legacy single prompt YAML or a versioned prompt pool YAML."""

    prompt_path = Path(path)
    payload = yaml.safe_load(prompt_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"prompt file must contain a mapping: {prompt_path}")

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"prompt metadata must be a mapping: {prompt_path}")

    prompts = payload.get("prompts")
    if isinstance(prompts, list):
        return _load_pool_prompt(
            prompt_path,
            prompts=prompts,
            metadata=metadata,
            variant_id=variant_id,
        )

    prompt = payload.get("prompt") or {}
    if not isinstance(prompt, dict):
        raise ValueError(f"prompt file must contain prompt mapping: {prompt_path}")
    prompt_id = str(metadata.get("id") or payload.get("prompt_id") or prompt_path.stem).strip()
    system_prompt = str(prompt.get("system_prompt") or payload.get("system_prompt") or "").strip()
    user_prompt = str(prompt.get("user_prompt") or payload.get("user_prompt") or "").strip()
    if not user_prompt:
        raise ValueError(f"Missing user_prompt in {prompt_path}.")
    return ShaftPromptTemplate(
        prompt_id=prompt_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        metadata=dict(metadata),
        source_path=str(prompt_path),
    )


def load_prompt_pool(path: str | Path) -> list[ShaftPromptTemplate]:
    """Load every prompt variant from a versioned prompt pool YAML."""

    prompt_path = Path(path)
    payload = yaml.safe_load(prompt_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"prompt file must contain a mapping: {prompt_path}")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"prompt metadata must be a mapping: {prompt_path}")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Prompt pool must contain a non-empty prompts list: {prompt_path}")
    return _load_pool_prompts(prompt_path, prompts=prompts, metadata=metadata)


def _load_pool_prompt(
    prompt_path: Path,
    *,
    prompts: list[Any],
    metadata: dict[str, Any],
    variant_id: str,
) -> ShaftPromptTemplate:
    pool_id = str(metadata.get("id") or "").strip()
    if not pool_id:
        raise ValueError(f"Missing prompt pool id in {prompt_path}.")
    version = str(metadata.get("version") or "").strip()
    if not version:
        raise ValueError(f"Missing prompt pool version in {prompt_path}.")
    for index, item in enumerate(prompts):
        if not isinstance(item, dict):
            raise TypeError(f"Prompt pool item must be a mapping: {prompt_path}:prompts[{index}]")
        item_variant_id = str(item.get("id") or "").strip()
        if item_variant_id == variant_id:
            return _load_pool_prompt_item(
                prompt_path,
                item=item,
                metadata=metadata,
                pool_id=pool_id,
                version=version,
                variant_id=variant_id,
            )
    raise ValueError(f"Prompt pool variant {variant_id!r} not found in {prompt_path}.")


def _load_pool_prompts(
    prompt_path: Path,
    *,
    prompts: list[Any],
    metadata: dict[str, Any],
) -> list[ShaftPromptTemplate]:
    pool_id = str(metadata.get("id") or "").strip()
    if not pool_id:
        raise ValueError(f"Missing prompt pool id in {prompt_path}.")
    version = str(metadata.get("version") or "").strip()
    if not version:
        raise ValueError(f"Missing prompt pool version in {prompt_path}.")
    variants: list[ShaftPromptTemplate] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(prompts):
        if not isinstance(item, dict):
            raise TypeError(f"Prompt pool item must be a mapping: {prompt_path}:prompts[{index}]")
        variant_id = str(item.get("id") or "").strip()
        if not variant_id:
            raise ValueError(f"Prompt pool item is missing id: {prompt_path}:prompts[{index}]")
        if variant_id in seen_ids:
            raise ValueError(f"Duplicate prompt variant id {variant_id!r} in {prompt_path}")
        seen_ids.add(variant_id)
        variants.append(
            _load_pool_prompt_item(
                prompt_path,
                item=item,
                metadata=metadata,
                pool_id=pool_id,
                version=version,
                variant_id=variant_id,
            )
        )
    if not any(prompt.sampling_weight > 0 for prompt in variants):
        raise ValueError(f"Prompt pool must have at least one positive sampling_weight: {prompt_path}")
    return variants


def _load_pool_prompt_item(
    prompt_path: Path,
    *,
    item: dict[str, Any],
    metadata: dict[str, Any],
    pool_id: str,
    version: str,
    variant_id: str,
) -> ShaftPromptTemplate:
    try:
        sampling_weight = float(item.get("sampling_weight", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Prompt pool variant {variant_id!r} has invalid sampling_weight in {prompt_path}."
        ) from exc
    if not math.isfinite(sampling_weight) or sampling_weight < 0:
        raise ValueError(
            f"Prompt pool variant {variant_id!r} sampling_weight must be finite and >= 0 "
            f"in {prompt_path}."
        )
    system_prompt = str(item.get("system_prompt") or "").strip()
    user_prompt = str(item.get("user_prompt") or "").strip()
    if not user_prompt:
        raise ValueError(f"Prompt pool variant {variant_id!r} is missing user_prompt in {prompt_path}.")
    prompt_metadata = dict(metadata)
    prompt_metadata["prompt_pool_id"] = pool_id
    prompt_metadata["prompt_version"] = version
    prompt_metadata["prompt_variant_id"] = variant_id
    return ShaftPromptTemplate(
        prompt_id=f"{pool_id}.{variant_id}",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        metadata=prompt_metadata,
        source_path=f"{prompt_path}#{variant_id}",
        variant_id=variant_id,
        version=version,
        sampling_weight=sampling_weight,
    )
