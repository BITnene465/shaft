from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig


@dataclass(frozen=True, slots=True)
class ResolvedModelDescriptor:
    """Architecture facts resolved from the upstream HF config."""

    hf_model_type: str
    architectures: tuple[str, ...]
    text_layer_types: tuple[str, ...]
    source: str
    config_fingerprint: str
    config_json: str
    commit_hash: str | None = None

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        source: str,
        commit_hash: str | None = None,
    ) -> "ResolvedModelDescriptor":
        model_type = str(payload.get("model_type") or "").strip().lower()
        if not model_type:
            raise ValueError("HF config has no non-empty model_type.")
        architectures = tuple(
            str(value).strip()
            for value in (payload.get("architectures") or ())
            if str(value).strip()
        )
        text_config = payload.get("text_config")
        text_payload = text_config if isinstance(text_config, dict) else payload
        layer_types = tuple(
            str(value).strip().lower()
            for value in (text_payload.get("layer_types") or ())
            if str(value).strip()
        )
        config_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return cls(
            hf_model_type=model_type,
            architectures=architectures,
            text_layer_types=layer_types,
            source=str(source),
            config_fingerprint=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
            config_json=config_json,
            commit_hash=(
                str(commit_hash).strip() if str(commit_hash or "").strip() else None
            ),
        )

    def config_value(self, dotted_path: str, default: Any = None) -> Any:
        value: Any = json.loads(self.config_json)
        for part in str(dotted_path).split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


def resolve_local_model_descriptor(
    model_name_or_path: str | Path,
) -> ResolvedModelDescriptor | None:
    path = Path(model_name_or_path)
    config_path = path / "config.json" if path.is_dir() else None
    if config_path is None or not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid HF config JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"HF config must be a JSON object: {config_path}")
    if not str(payload.get("model_type") or "").strip():
        return None
    return ResolvedModelDescriptor.from_payload(payload, source=str(config_path))


def resolve_model_descriptor(
    model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    allow_remote: bool = True,
) -> ResolvedModelDescriptor | None:
    """Resolve immutable HF config facts without importing model implementation code."""

    local = resolve_local_model_descriptor(model_name_or_path)
    if local is not None:
        return local
    path = Path(model_name_or_path)
    if path.exists() or not allow_remote:
        return None
    try:
        payload, resolved_kwargs = PretrainedConfig.get_config_dict(
            str(model_name_or_path),
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_files_only=bool(local_files_only),
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Unable to resolve HF config for {str(model_name_or_path)!r}. "
            "Provide a valid local directory, a reachable Hub repo/revision, or a "
            "catalog model id whose architecture is unambiguous."
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError("Transformers returned a non-object HF config payload.")
    if not str(payload.get("model_type") or "").strip():
        return None
    commit_hash = str(
        resolved_kwargs.get("_commit_hash")
        or payload.get("_commit_hash")
        or ""
    ).strip()
    source = f"hf://{model_name_or_path}"
    if revision:
        source += f"@{revision}"
    return ResolvedModelDescriptor.from_payload(
        payload,
        source=source,
        commit_hash=commit_hash or None,
    )
