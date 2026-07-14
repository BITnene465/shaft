from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from shaft.config import RuntimeConfig

from .descriptor import ResolvedModelDescriptor, resolve_model_descriptor
from .registry import build_model_meta
from .types import (
    ModelMeta,
    ShaftModelAdapter,
    ShaftSequenceExecutionContract,
)


@dataclass(frozen=True, slots=True)
class ResolvedAdapterInit:
    """Immutable PEFT-init identity resolved before model construction."""

    path: str
    config_json: str
    config_fingerprint: str
    base_model_name_or_path: str
    weight_manifest: tuple[tuple[str, int, str], ...]
    artifact_fingerprint: str

    def config_dict(self) -> dict[str, Any]:
        payload = json.loads(self.config_json)
        if not isinstance(payload, dict):
            raise TypeError("Resolved adapter config is not a JSON object.")
        return payload


@dataclass(frozen=True, slots=True)
class ResolvedModelPlan:
    """Single source of truth for model artifact identity and family capabilities."""

    configured_model_name_or_path: str
    effective_model_name_or_path: str
    init_from_checkpoint: str | None
    init_kind: str
    model_meta: ModelMeta
    descriptor: ResolvedModelDescriptor | None
    model_adapter: ShaftModelAdapter
    revision: str | None
    cache_dir: str | None
    local_files_only: bool
    adapter_init: ResolvedAdapterInit | None = None

    def __post_init__(self) -> None:
        if self.init_kind not in {"base", "adapter", "full_checkpoint"}:
            raise ValueError(f"Unsupported model plan init kind: {self.init_kind!r}.")
        if not self.configured_model_name_or_path or not self.effective_model_name_or_path:
            raise ValueError("Resolved model paths must not be empty.")
        if self.model_adapter.model_meta is not self.model_meta:
            raise ValueError("Resolved model adapter and model metadata do not share a truth source.")
        if self.model_adapter.model_name_or_path != self.effective_model_name_or_path:
            raise ValueError("Resolved model adapter does not target the effective load artifact.")
        if (self.init_kind == "adapter") != (self.adapter_init is not None):
            raise ValueError("Resolved adapter init must exist exactly for adapter plans.")

    @property
    def fingerprint(self) -> str:
        payload = {
            "model_type": self.model_meta.model_type,
            "configured_model_name_or_path": self.configured_model_name_or_path,
            "effective_model_name_or_path": self.effective_model_name_or_path,
            "init_kind": self.init_kind,
            "revision": self.revision,
            "descriptor_fingerprint": (
                None if self.descriptor is None else self.descriptor.config_fingerprint
            ),
            "group_name": self.model_adapter.group_name,
            "template_type": self.model_adapter.template_type,
            "adapter_artifact_fingerprint": (
                None
                if self.adapter_init is None
                else self.adapter_init.artifact_fingerprint
            ),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def build_sequence_execution_contract(
        self,
        **kwargs: Any,
    ) -> ShaftSequenceExecutionContract:
        return self.model_adapter.build_sequence_execution_contract(**kwargs)


def _is_adapter_checkpoint(path: Path) -> bool:
    return bool(
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and (
            (path / "adapter_model.safetensors").is_file()
            or (path / "adapter_model.bin").is_file()
        )
    )


def _resolve_adapter_init(path: Path) -> ResolvedAdapterInit:
    config_path = path / "adapter_config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid adapter config JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Adapter config must be a JSON object: {config_path}")
    base_model = str(payload.get("base_model_name_or_path") or "").strip()
    if not base_model:
        raise ValueError(
            f"Adapter config has no base_model_name_or_path: {config_path}."
        )
    config_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    config_fingerprint = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    weight_manifest = tuple(
        (
            candidate.name,
            int(candidate.stat().st_size),
            _file_sha256(candidate),
        )
        for candidate in (
            path / "adapter_model.safetensors",
            path / "adapter_model.bin",
        )
        if candidate.is_file()
    )
    artifact_payload = {
        "config_fingerprint": config_fingerprint,
        "weight_manifest": weight_manifest,
    }
    artifact_fingerprint = hashlib.sha256(
        json.dumps(artifact_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return ResolvedAdapterInit(
        path=str(path),
        config_json=config_json,
        config_fingerprint=config_fingerprint,
        base_model_name_or_path=base_model,
        weight_manifest=weight_manifest,
        artifact_fingerprint=artifact_fingerprint,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_artifact(left: str, right: str) -> bool:
    if str(left).strip().rstrip("/") == str(right).strip().rstrip("/"):
        return True
    left_path = Path(left)
    right_path = Path(right)
    return bool(
        left_path.exists()
        and right_path.exists()
        and left_path.resolve() == right_path.resolve()
    )


def _validate_adapter_base(
    adapter_init: ResolvedAdapterInit,
    *,
    model_meta: ModelMeta,
    model_adapter: ShaftModelAdapter,
    descriptor: ResolvedModelDescriptor | None,
    revision: str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> None:
    declared_base = adapter_init.base_model_name_or_path
    if _same_artifact(declared_base, model_adapter.model_name_or_path):
        return
    declared_descriptor = resolve_model_descriptor(
        declared_base,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_remote=(
            model_meta.uses_hf_artifacts
            and _looks_like_hub_repo_id(declared_base)
        ),
    )
    declared_adapter = model_meta.resolve_adapter(
        model_name_or_path=declared_base,
        descriptor=declared_descriptor,
    )
    if declared_adapter.group_name != model_adapter.group_name:
        raise ValueError(
            "Adapter base variant differs from the configured model plan: "
            f"adapter={declared_adapter.group_name!r}, "
            f"configured={model_adapter.group_name!r}."
        )
    if descriptor is None or declared_descriptor is None:
        raise ValueError(
            "Adapter base artifact differs from model.model_name_or_path and their "
            "HF config identity cannot be proven equivalent."
        )
    if descriptor.config_fingerprint != declared_descriptor.config_fingerprint:
        raise ValueError(
            "Adapter base HF config differs from the configured model artifact."
        )


def _variant_type_count(model_meta: ModelMeta) -> int:
    values = {
        str(value).strip().lower()
        for value in (
            *model_meta.hf_model_types,
            *(
                item
                for group in model_meta.model_groups
                for item in group.hf_model_types
            ),
        )
        if str(value).strip()
    }
    return len(values)


def _looks_like_hub_repo_id(value: str) -> bool:
    raw = str(value).strip()
    path = Path(raw)
    if not raw or path.exists() or path.is_absolute():
        return False
    normalized = raw.replace("\\", "/")
    if normalized.startswith(("./", "../")):
        return False
    parts = tuple(part for part in normalized.split("/") if part)
    return len(parts) == 2 and parts[0] not in {".", ".."}


def resolve_model_plan(
    config: RuntimeConfig,
    *,
    init_from_checkpoint: str | None = None,
) -> ResolvedModelPlan:
    model_meta = build_model_meta(str(config.model.model_type).strip().lower())
    configured_path = str(config.model.model_name_or_path).strip()
    init_path: Path | None = None
    adapter_init: ResolvedAdapterInit | None = None
    init_kind = "base"
    effective_path = configured_path
    if init_from_checkpoint is not None:
        init_path = Path(init_from_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"init_from checkpoint path not found: {init_path}")
        if _is_adapter_checkpoint(init_path):
            init_kind = "adapter"
            adapter_init = _resolve_adapter_init(init_path)
        else:
            init_kind = "full_checkpoint"
            effective_path = str(init_path)

    descriptor = resolve_model_descriptor(
        effective_path,
        revision=config.model.revision,
        cache_dir=config.model.cache_dir,
        local_files_only=bool(config.model.local_files_only),
        allow_remote=False,
    )
    catalog_match = (
        None
        if descriptor is not None
        else model_meta.get_matched_model_group(effective_path)
    )
    if (
        descriptor is None
        and model_meta.uses_hf_artifacts
        and (
            _looks_like_hub_repo_id(effective_path)
            or (_variant_type_count(model_meta) > 1 and catalog_match is None)
        )
    ):
        descriptor = resolve_model_descriptor(
            effective_path,
            revision=config.model.revision,
            cache_dir=config.model.cache_dir,
            local_files_only=bool(config.model.local_files_only),
            allow_remote=True,
        )

    model_adapter = model_meta.resolve_adapter(
        model_name_or_path=effective_path,
        template_type=config.model.template,
        descriptor=descriptor,
    )
    if adapter_init is not None:
        _validate_adapter_base(
            adapter_init,
            model_meta=model_meta,
            model_adapter=model_adapter,
            descriptor=descriptor,
            revision=config.model.revision,
            cache_dir=config.model.cache_dir,
            local_files_only=bool(config.model.local_files_only),
        )
    return ResolvedModelPlan(
        configured_model_name_or_path=configured_path,
        effective_model_name_or_path=effective_path,
        init_from_checkpoint=None if init_path is None else str(init_path),
        init_kind=init_kind,
        model_meta=model_meta,
        descriptor=descriptor,
        model_adapter=model_adapter,
        revision=config.model.revision,
        cache_dir=config.model.cache_dir,
        local_files_only=bool(config.model.local_files_only),
        adapter_init=adapter_init,
    )
