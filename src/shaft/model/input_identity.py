from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any


_ARTIFACT_LOCATOR_FIELDS = frozenset(
    {
        "_name_or_path",
        "cache_dir",
        "is_local",
        "local_files_only",
        "name_or_path",
        "pretrained_model_name_or_path",
    }
)


def _is_artifact_locator_field(value: Any) -> bool:
    if type(value) is not str:
        return False
    normalized = value.strip().lower()
    return bool(
        normalized in _ARTIFACT_LOCATOR_FIELDS
        or normalized.endswith(("_file", "_name_or_path", "_path"))
    )


def artifact_semantic_value(value: Any) -> Any:
    """Remove loader locators while retaining behavior-bearing artifact config.

    Hugging Face injects the selected directory and resolved vocabulary-file paths
    into ``name_or_path``/``init_kwargs``. Those values explain where an artifact
    was loaded from, but they do not change the behavior of a tokenizer whose
    backend model and complete vocabulary are already content-bound. Revision and
    repository identity are deliberately *not* removed here; the resolved model
    plan remains their immutable identity source.
    """

    if isinstance(value, Mapping):
        return {
            key: artifact_semantic_value(item)
            for key, item in value.items()
            if not _is_artifact_locator_field(key)
        }
    if isinstance(value, list):
        return [artifact_semantic_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(artifact_semantic_value(item) for item in value)
    if isinstance(value, set):
        return {artifact_semantic_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(artifact_semantic_value(item) for item in value)
    return value


def added_token_semantic_value(value: Any) -> dict[str, Any] | None:
    """Return the stable behavior-bearing state of tokenizers.AddedToken."""

    qualified_type = f"{type(value).__module__}.{type(value).__qualname__}"
    if qualified_type != "tokenizers.AddedToken":
        return None
    content = getattr(value, "content", None)
    if type(content) is not str:
        raise TypeError("tokenizers.AddedToken.content must be a string.")
    flags: dict[str, bool] = {}
    for field_name in (
        "single_word",
        "lstrip",
        "rstrip",
        "normalized",
        "special",
    ):
        field_value = getattr(value, field_name, None)
        if type(field_value) is not bool:
            raise TypeError(f"tokenizers.AddedToken.{field_name} must be a boolean.")
        flags[field_name] = field_value
    return {
        "added_token_type": qualified_type,
        "content": content,
        **flags,
    }


def stable_artifact_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return tuple(
            (str(key), stable_artifact_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(stable_artifact_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        resolved = [stable_artifact_value(item) for item in value]
        return tuple(sorted(resolved, key=repr))
    added_token = added_token_semantic_value(value)
    if added_token is not None:
        return added_token
    return f"{type(value).__module__}.{type(value).__qualname__}"


def tokenizer_artifact_fingerprint(tokenizer: Any) -> str:
    """Bind tokenizer behavior to its serialized vocabulary/tokenization model."""

    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend, "to_str", None)
    if callable(backend_to_str):
        artifact_kind = "backend-tokenizer-json"
        artifact_payload = str(backend_to_str())
    else:
        declared = getattr(tokenizer, "shaft_tokenizer_fingerprint", None)
        if declared is None:
            # Compatibility for model adapters that declared the identity before
            # tokenizer semantics were shared with exact-resume contracts.
            declared = getattr(tokenizer, "shaft_cost_fingerprint", None)
        declared = declared() if callable(declared) else declared
        artifact_payload = str(declared or "").strip()
        if not artifact_payload:
            raise ValueError(
                "Exact tokenizer identity requires tokenizer.backend_tokenizer.to_str() "
                "or an explicit tokenizer.shaft_tokenizer_fingerprint (legacy "
                "shaft_cost_fingerprint) covering the full vocabulary and tokenization "
                "model (including merges/unigram state)."
            )
        artifact_kind = "declared-shaft-tokenizer-fingerprint"

    get_vocab = getattr(tokenizer, "get_vocab", None)
    base_vocab = get_vocab() if callable(get_vocab) else None
    if base_vocab is not None and not isinstance(base_vocab, Mapping):
        raise TypeError("tokenizer.get_vocab() must return a mapping.")
    base_vocab_fingerprint = (
        None
        if base_vocab is None
        else hashlib.sha256(repr(stable_artifact_value(base_vocab)).encode("utf-8")).hexdigest()
    )

    metadata = (
        "shaft-tokenizer-artifact-v3",
        artifact_kind,
        hashlib.sha256(artifact_payload.encode("utf-8")).hexdigest(),
        f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        getattr(tokenizer, "vocab_size", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "model_max_length", None),
        getattr(tokenizer, "padding_side", None),
        getattr(tokenizer, "truncation_side", None),
        stable_artifact_value(getattr(tokenizer, "special_tokens_map", {})),
        stable_artifact_value(artifact_semantic_value(getattr(tokenizer, "init_kwargs", {}))),
        stable_artifact_value(getattr(tokenizer, "added_tokens_encoder", {})),
        base_vocab_fingerprint,
    )
    return hashlib.sha256(repr(metadata).encode("utf-8")).hexdigest()
