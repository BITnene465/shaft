from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors

from shaft.config import RuntimeConfig

from .artifact_identity import (
    LocalModelArtifactLoadGuard,
    ResolvedModelArtifactIdentity,
    capture_local_model_artifact_load_guard,
    external_auto_map_repositories,
    resolve_model_artifact_identity,
    validate_local_model_artifact_load_guard,
)
from .descriptor import ResolvedModelDescriptor, resolve_model_descriptor
from .registry import build_model_meta
from .training_identity import model_training_semantic_fingerprint
from .types import (
    ModelMeta,
    ShaftModelAdapter,
    ShaftSequenceExecutionContract,
)


@dataclass(frozen=True, slots=True)
class ResolvedAdapterInit:
    """Immutable PEFT adapter identity resolved before model construction."""

    path: str
    config_json: str
    config_fingerprint: str
    config_size: int
    config_sha256: str
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
    resolved_revision: str | None
    cache_dir: str | None
    local_files_only: bool
    trust_remote_code: bool
    require_immutable_artifact: bool
    artifact_identity: ResolvedModelArtifactIdentity
    model_training_semantic_fingerprint: str
    distributed_artifact_identity: bool = False
    adapter_init: ResolvedAdapterInit | None = None

    def __post_init__(self) -> None:
        if self.init_kind not in {"base", "adapter", "full_checkpoint"}:
            raise ValueError(f"Unsupported model plan init kind: {self.init_kind!r}.")
        if not self.configured_model_name_or_path or not self.effective_model_name_or_path:
            raise ValueError("Resolved model paths must not be empty.")
        if self.model_adapter.model_meta is not self.model_meta:
            raise ValueError(
                "Resolved model adapter and model metadata do not share a truth source."
            )
        if self.model_adapter.model_name_or_path != self.effective_model_name_or_path:
            raise ValueError("Resolved model adapter does not target the effective load artifact.")
        if (self.init_kind == "adapter") != (self.adapter_init is not None):
            raise ValueError("Resolved adapter init must exist exactly for adapter plans.")
        if (
            self.artifact_identity.resolved_revision is not None
            and self.resolved_revision != self.artifact_identity.resolved_revision
        ):
            raise ValueError("Resolved model revision differs from the artifact identity.")
        implementation_fingerprint = self.model_training_semantic_fingerprint
        if len(implementation_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in implementation_fingerprint
        ):
            raise ValueError(
                "Resolved model training semantic fingerprint must be a lowercase SHA-256 digest."
            )

    @property
    def fingerprint(self) -> str:
        # Complete local HF identities are content-addressed by the relative file
        # manifest in ``artifact_identity``. Keep configured/effective paths on the
        # plan for loading and audit, but do not turn a byte-identical relocation
        # into a different training trajectory. Hub and incomplete local identities
        # remain locator-bound and fail closed.
        content_addressed_local = bool(
            self.artifact_identity.kind == "local_hf" and self.artifact_identity.complete
        )
        payload = {
            "version": "shaft-resolved-model-plan-v2",
            "model_type": self.model_meta.model_type,
            "artifact_locator": (
                None
                if content_addressed_local
                else {
                    "configured_model_name_or_path": (self.configured_model_name_or_path),
                    "effective_model_name_or_path": (self.effective_model_name_or_path),
                }
            ),
            "init_kind": self.init_kind,
            "revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "descriptor_fingerprint": (
                None if self.descriptor is None else self.descriptor.config_fingerprint
            ),
            "group_name": self.model_adapter.group_name,
            "template_type": self.model_adapter.template_type,
            "model_training_semantic_fingerprint": (self.model_training_semantic_fingerprint),
            "adapter_artifact_fingerprint": (
                None if self.adapter_init is None else self.adapter_init.artifact_fingerprint
            ),
            "model_artifact_fingerprint": self.artifact_identity.fingerprint,
            "model_artifact_identity_complete": self.artifact_identity.complete,
            "trust_remote_code": self.trust_remote_code,
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
            (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
        )
    )


def resolve_adapter_artifact(path: str | Path) -> ResolvedAdapterInit:
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"PEFT adapter directory is missing: {path}.")
    config_path = path / "adapter_config.json"
    if not config_path.is_file() or config_path.stat().st_size <= 0:
        raise FileNotFoundError(f"PEFT adapter config is missing or empty: {config_path}.")
    config_bytes = config_path.read_bytes()
    try:
        payload = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid adapter config JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Adapter config must be a JSON object: {config_path}")
    base_model = str(payload.get("base_model_name_or_path") or "").strip()
    config_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    config_fingerprint = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    weight_path = next(
        (
            candidate
            for candidate in (
                path / "adapter_model.safetensors",
                path / "adapter_model.bin",
            )
            if candidate.is_file()
        ),
        None,
    )
    if weight_path is None or weight_path.stat().st_size <= 0:
        raise FileNotFoundError(f"PEFT adapter weights are missing or empty: {path}.")
    weight_manifest = (
        (
            weight_path.name,
            int(weight_path.stat().st_size),
            _file_sha256(weight_path),
        ),
    )
    config_size = len(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    artifact_payload = {
        "config_fingerprint": config_fingerprint,
        "config_size": config_size,
        "config_sha256": config_sha256,
        "weight_manifest": weight_manifest,
    }
    artifact_fingerprint = hashlib.sha256(
        json.dumps(artifact_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ResolvedAdapterInit(
        path=str(path),
        config_json=config_json,
        config_fingerprint=config_fingerprint,
        config_size=config_size,
        config_sha256=config_sha256,
        base_model_name_or_path=base_model,
        weight_manifest=weight_manifest,
        artifact_fingerprint=artifact_fingerprint,
    )


def validate_resolved_adapter_artifact(adapter: ResolvedAdapterInit) -> None:
    """Reject adapter bytes that changed after immutable resolution."""

    directory = Path(adapter.path)
    config_path = directory / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"PEFT adapter config is missing: {config_path}.")
    config_bytes = config_path.read_bytes()
    config_identity = (len(config_bytes), hashlib.sha256(config_bytes).hexdigest())
    if config_identity != (adapter.config_size, adapter.config_sha256):
        raise ValueError("PEFT adapter config changed after artifact resolution.")

    if len(adapter.weight_manifest) != 1:
        raise ValueError("Resolved PEFT adapter must select exactly one weight file.")
    weight_name, expected_size, expected_sha256 = adapter.weight_manifest[0]
    weight_path = directory / weight_name
    if not weight_path.is_file():
        raise FileNotFoundError(f"PEFT adapter weights are missing: {weight_path}.")
    actual = (int(weight_path.stat().st_size), _file_sha256(weight_path))
    if actual != (expected_size, expected_sha256):
        raise ValueError("PEFT adapter weights changed after artifact resolution.")


def load_resolved_adapter_weights(
    adapter: ResolvedAdapterInit,
) -> dict[str, torch.Tensor]:
    """Deserialize the exact adapter bytes bound by ``adapter``."""

    config_path = Path(adapter.path) / "adapter_config.json"
    config_bytes = config_path.read_bytes()
    if (len(config_bytes), hashlib.sha256(config_bytes).hexdigest()) != (
        adapter.config_size,
        adapter.config_sha256,
    ):
        raise ValueError("PEFT adapter config changed after artifact resolution.")
    if len(adapter.weight_manifest) != 1:
        raise ValueError("Resolved PEFT adapter must select exactly one weight file.")
    weight_name, expected_size, expected_sha256 = adapter.weight_manifest[0]
    weight_path = Path(adapter.path) / weight_name
    payload = weight_path.read_bytes()
    actual = (len(payload), hashlib.sha256(payload).hexdigest())
    if actual != (expected_size, expected_sha256):
        raise ValueError("PEFT adapter weights changed while being loaded.")
    if weight_path.suffix == ".safetensors":
        state = load_safetensors(payload)
    else:
        state = torch.load(
            io.BytesIO(payload),
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    if not isinstance(state, dict) or not state:
        raise TypeError("PEFT adapter weights must deserialize to a non-empty state dictionary.")
    return state


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
        left_path.exists() and right_path.exists() and left_path.resolve() == right_path.resolve()
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
    if not declared_base:
        raise ValueError(
            "Adapter init requires adapter_config.json.base_model_name_or_path."
        )
    if _same_artifact(declared_base, model_adapter.model_name_or_path):
        return
    declared_descriptor = resolve_model_descriptor(
        declared_base,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_remote=(model_meta.uses_hf_artifacts and _looks_like_hub_repo_id(declared_base)),
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
        raise ValueError("Adapter base HF config differs from the configured model artifact.")


def _variant_type_count(model_meta: ModelMeta) -> int:
    values = {
        str(value).strip().lower()
        for value in (
            *model_meta.hf_model_types,
            *(item for group in model_meta.model_groups for item in group.hf_model_types),
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
    require_immutable_artifact: bool = False,
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
            adapter_init = resolve_adapter_artifact(init_path)
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
        None if descriptor is not None else model_meta.get_matched_model_group(effective_path)
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
    is_hub_repo = _looks_like_hub_repo_id(effective_path)
    artifact_identity = resolve_model_artifact_identity(
        effective_path,
        model_type=model_meta.model_type,
        uses_hf_artifacts=model_meta.uses_hf_artifacts,
        trust_remote_code=bool(config.model.trust_remote_code),
        requested_revision=config.model.revision,
        resolved_commit_hash=(None if descriptor is None else descriptor.commit_hash),
        is_hub_repo=is_hub_repo,
        require_immutable_local=bool(require_immutable_artifact),
        external_remote_code_repositories=(
            ()
            if descriptor is None
            else external_auto_map_repositories(
                json.loads(descriptor.config_json),
                model_repo_id=effective_path,
            )
        ),
        distributed_consensus=False,
    )
    resolved_revision = (
        artifact_identity.resolved_revision
        if artifact_identity.resolved_revision is not None
        else config.model.revision
    )
    plan = ResolvedModelPlan(
        configured_model_name_or_path=configured_path,
        effective_model_name_or_path=effective_path,
        init_from_checkpoint=None if init_path is None else str(init_path),
        init_kind=init_kind,
        model_meta=model_meta,
        descriptor=descriptor,
        model_adapter=model_adapter,
        revision=config.model.revision,
        resolved_revision=resolved_revision,
        cache_dir=config.model.cache_dir,
        local_files_only=bool(config.model.local_files_only),
        trust_remote_code=bool(config.model.trust_remote_code),
        require_immutable_artifact=bool(require_immutable_artifact),
        artifact_identity=artifact_identity,
        model_training_semantic_fingerprint=(model_training_semantic_fingerprint(model_adapter)),
        distributed_artifact_identity=False,
        adapter_init=adapter_init,
    )
    if bool(require_immutable_artifact):
        validate_resolved_model_descriptor(plan)
    return plan


def materialize_resolved_model_artifact_identity(
    plan: ResolvedModelPlan,
) -> ResolvedModelPlan:
    """Materialize immutable local bytes after rank-local plan consensus."""

    if plan.artifact_identity.kind != "local_hf":
        return replace(plan, require_immutable_artifact=True)
    descriptor = plan.descriptor
    identity = resolve_model_artifact_identity(
        plan.effective_model_name_or_path,
        model_type=plan.model_meta.model_type,
        uses_hf_artifacts=plan.model_meta.uses_hf_artifacts,
        trust_remote_code=plan.trust_remote_code,
        requested_revision=plan.revision,
        resolved_commit_hash=None if descriptor is None else descriptor.commit_hash,
        is_hub_repo=False,
        require_immutable_local=True,
        external_remote_code_repositories=(
            ()
            if descriptor is None
            else external_auto_map_repositories(
                json.loads(descriptor.config_json),
                model_repo_id=plan.effective_model_name_or_path,
            )
        ),
        distributed_consensus=True,
    )
    return replace(
        plan,
        require_immutable_artifact=True,
        artifact_identity=identity,
        distributed_artifact_identity=True,
    )


def validate_resolved_model_descriptor(plan: ResolvedModelPlan) -> None:
    if plan.artifact_identity.kind != "local_hf" or not plan.artifact_identity.complete:
        return
    stable_descriptor = resolve_model_descriptor(
        plan.effective_model_name_or_path,
        revision=plan.revision,
        cache_dir=plan.cache_dir,
        local_files_only=plan.local_files_only,
        allow_remote=False,
    )
    if plan.descriptor is None or stable_descriptor is None:
        raise ValueError(
            "Exact checkpoint save/resume requires a valid local HF config descriptor."
        )
    if stable_descriptor.config_fingerprint != plan.descriptor.config_fingerprint:
        raise RuntimeError(
            "Local HF config changed while the resolved model plan was being constructed."
        )


def validate_model_artifact_checkpointability(
    plan: ResolvedModelPlan,
    *,
    save_strategy: str,
    resume_requested: bool,
) -> None:
    """Fail before data/model loading when base bytes are not immutable."""

    checkpointing_requested = str(save_strategy).strip().lower() != "no" or bool(resume_requested)
    if checkpointing_requested and not plan.artifact_identity.complete:
        raise ValueError(
            "Exact checkpoint save/resume requires an immutable base-model "
            "artifact identity: "
            f"{list(plan.artifact_identity.incomplete_reasons)}. Use a local HF "
            "directory with complete weight files or a Hub revision that resolves "
            "to an immutable commit SHA; otherwise set train.save_strategy='no'."
        )


def prepare_resolved_model_artifact_load(
    plan: ResolvedModelPlan,
) -> LocalModelArtifactLoadGuard | None:
    """Open a cheap metadata guard for one local HF loader invocation."""

    if plan.artifact_identity.kind != "local_hf" or not plan.artifact_identity.complete:
        return None
    return capture_local_model_artifact_load_guard(
        plan.artifact_identity,
        root=Path(plan.effective_model_name_or_path),
        trust_remote_code=plan.trust_remote_code,
    )


def validate_resolved_model_artifact(
    plan: ResolvedModelPlan,
    *,
    load_guard: LocalModelArtifactLoadGuard | None = None,
) -> None:
    """Verify current local bytes, optionally closing an HF loader window."""

    if plan.artifact_identity.kind != "local_hf" or not plan.artifact_identity.complete:
        if load_guard is not None:
            raise ValueError("A local artifact load guard cannot validate a non-local plan.")
        return
    current = resolve_model_artifact_identity(
        plan.effective_model_name_or_path,
        model_type=plan.model_meta.model_type,
        uses_hf_artifacts=plan.model_meta.uses_hf_artifacts,
        trust_remote_code=plan.trust_remote_code,
        requested_revision=plan.revision,
        resolved_commit_hash=None,
        is_hub_repo=False,
        require_immutable_local=True,
        external_remote_code_repositories=(),
        distributed_consensus=plan.distributed_artifact_identity,
    )
    if current.fingerprint != plan.artifact_identity.fingerprint:
        raise ValueError(
            "Base-model weights or local model code changed after ResolvedModelPlan construction."
        )
    if load_guard is not None:
        validate_local_model_artifact_load_guard(
            plan.artifact_identity,
            load_guard,
            root=Path(plan.effective_model_name_or_path),
            trust_remote_code=plan.trust_remote_code,
        )
