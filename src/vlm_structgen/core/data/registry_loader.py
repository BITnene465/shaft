from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vlm_structgen.core.config import ExperimentRuntimeConfig, load_prompt_profile_payload
from vlm_structgen.core.registry import parse_route_key


@dataclass(frozen=True)
class ResolvedDatasetSources:
    train_paths: list[str]
    val_paths: list[str]
    train_routes: list[str]
    val_routes: list[str]
    route_option_defaults: dict[str, dict[str, Any]]
    route_prompt_defaults: dict[str, dict[str, Any]]
    source_mode: str


@dataclass(frozen=True)
class _DatasetRegistryEntry:
    dataset_id: str
    task_type: str
    domain_type: str
    train_path: str
    val_path: str
    default_mix_weight: float
    prompt_profile: str | None
    enabled: bool

    @property
    def route_key(self) -> str:
        return f"{self.task_type}/{self.domain_type}"


def resolve_training_data_sources(
    config: ExperimentRuntimeConfig,
    *,
    config_path: str | Path,
) -> ResolvedDatasetSources:
    return _resolve_from_registry(config, config_path=Path(config_path))


def _resolve_from_registry(
    config: ExperimentRuntimeConfig,
    *,
    config_path: Path,
) -> ResolvedDatasetSources:
    train_dataset_ids = _normalize_dataset_ids(config.data.train_datasets, field_name="data.train_datasets")
    val_dataset_ids = _normalize_dataset_ids(config.data.val_datasets, field_name="data.val_datasets")
    if not train_dataset_ids:
        raise ValueError("Training requires non-empty data.train_datasets.")
    if not val_dataset_ids:
        raise ValueError("Training requires non-empty data.val_datasets.")
    registry_path = _resolve_registry_path(config.data.registry_path, config_path=config_path)
    entries = _load_registry_entries(registry_path)

    route_option_defaults: dict[str, dict[str, Any]] = {}
    route_prompt_defaults: dict[str, dict[str, Any]] = {}

    train_paths, train_routes = _resolve_registry_split(
        dataset_ids=train_dataset_ids,
        entries=entries,
        split_name="train",
        config_path=config_path,
        route_option_defaults=route_option_defaults,
        route_prompt_defaults=route_prompt_defaults,
    )
    val_paths, val_routes = _resolve_registry_split(
        dataset_ids=val_dataset_ids,
        entries=entries,
        split_name="val",
        config_path=config_path,
        route_option_defaults=route_option_defaults,
        route_prompt_defaults=route_prompt_defaults,
    )

    return ResolvedDatasetSources(
        train_paths=train_paths,
        val_paths=val_paths,
        train_routes=train_routes,
        val_routes=val_routes,
        route_option_defaults=route_option_defaults,
        route_prompt_defaults=route_prompt_defaults,
        source_mode="registry_only",
    )


def _normalize_dataset_ids(dataset_ids: list[str], *, field_name: str) -> list[str]:
    normalized = [str(dataset_id).strip() for dataset_id in list(dataset_ids) if str(dataset_id).strip()]
    duplicates = sorted({dataset_id for dataset_id in normalized if normalized.count(dataset_id) > 1})
    if duplicates:
        raise ValueError(
            f"{field_name} contains duplicated dataset ids: {duplicates}. "
            "Please keep each dataset id unique per split."
        )
    return normalized


def _resolve_registry_path(registry_path: str | None, *, config_path: Path) -> Path:
    if registry_path is None or not str(registry_path).strip():
        raise ValueError(
            "Registry mode requires data.registry_path. "
            "Please set data.registry_path when using data.train_datasets/data.val_datasets."
        )
    raw_path = Path(str(registry_path).strip())
    candidate_paths: list[Path] = []
    if raw_path.is_absolute():
        candidate_paths.append(raw_path)
    else:
        candidate_paths.append(config_path.parent / raw_path)
        candidate_paths.append(Path.cwd() / raw_path)
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Dataset registry not found: {registry_path!r}. "
        f"Tried: {[str(candidate) for candidate in candidate_paths]}"
    )


def _load_registry_entries(registry_path: Path) -> dict[str, _DatasetRegistryEntry]:
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    datasets_payload = payload.get("datasets")
    if not isinstance(datasets_payload, dict) or not datasets_payload:
        raise ValueError(
            f"Dataset registry must contain a non-empty `datasets` mapping: {registry_path}"
        )
    entries: dict[str, _DatasetRegistryEntry] = {}
    for dataset_id, raw_entry in datasets_payload.items():
        normalized_id = str(dataset_id).strip()
        if not normalized_id:
            raise ValueError(f"Dataset registry contains an empty dataset id: {registry_path}")
        entries[normalized_id] = _parse_registry_entry(
            dataset_id=normalized_id,
            raw_entry=raw_entry,
            registry_path=registry_path,
        )
    return entries


def _parse_registry_entry(
    *,
    dataset_id: str,
    raw_entry: Any,
    registry_path: Path,
) -> _DatasetRegistryEntry:
    if not isinstance(raw_entry, dict):
        raise ValueError(
            f"Dataset registry entry must be a mapping: dataset_id={dataset_id!r}, file={registry_path}"
        )
    allowed_keys = {
        "task_type",
        "domain_type",
        "train_path",
        "val_path",
        "default_mix_weight",
        "prompt_profile",
        "enabled",
        "tags",
    }
    unknown_keys = sorted(set(raw_entry.keys()) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unsupported keys in dataset registry entry {dataset_id!r}: {unknown_keys}. "
            f"Supported keys: {sorted(allowed_keys)}"
        )

    task_type, domain_type = parse_route_key(
        f"{raw_entry.get('task_type', '')}/{raw_entry.get('domain_type', '')}"
    )
    train_path = str(raw_entry.get("train_path", "")).strip()
    val_path = str(raw_entry.get("val_path", "")).strip()
    if not train_path or not val_path:
        raise ValueError(
            f"Dataset registry entry must define both train_path and val_path: dataset_id={dataset_id!r}, file={registry_path}"
        )

    default_mix_weight_raw = raw_entry.get("default_mix_weight", 1.0)
    try:
        default_mix_weight = float(default_mix_weight_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid default_mix_weight for dataset {dataset_id!r}: {default_mix_weight_raw!r}"
        ) from exc
    if default_mix_weight < 0.0:
        raise ValueError(
            f"default_mix_weight must be >= 0 for dataset {dataset_id!r}, got {default_mix_weight!r}"
        )

    prompt_profile = raw_entry.get("prompt_profile")
    normalized_prompt_profile = str(prompt_profile).strip() if prompt_profile is not None else None
    if normalized_prompt_profile == "":
        normalized_prompt_profile = None

    return _DatasetRegistryEntry(
        dataset_id=dataset_id,
        task_type=task_type,
        domain_type=domain_type,
        train_path=str(Path(train_path)),
        val_path=str(Path(val_path)),
        default_mix_weight=default_mix_weight,
        prompt_profile=normalized_prompt_profile,
        enabled=bool(raw_entry.get("enabled", True)),
    )


def _resolve_registry_split(
    *,
    dataset_ids: list[str],
    entries: dict[str, _DatasetRegistryEntry],
    split_name: str,
    config_path: Path,
    route_option_defaults: dict[str, dict[str, Any]],
    route_prompt_defaults: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    if split_name not in {"train", "val"}:
        raise ValueError(f"Unsupported split_name={split_name!r}.")
    split_path_field = "train_path" if split_name == "train" else "val_path"
    resolved_paths: list[str] = []
    resolved_routes: list[str] = []
    path_to_route: dict[str, str] = {}

    for dataset_id in dataset_ids:
        entry = entries.get(dataset_id)
        if entry is None:
            known_ids = sorted(entries.keys())
            raise ValueError(
                f"Unknown dataset id in data.{split_name}_datasets: {dataset_id!r}. "
                f"Known ids: {known_ids}"
            )
        if not entry.enabled:
            raise ValueError(
                f"Dataset id {dataset_id!r} is disabled in registry but referenced by data.{split_name}_datasets."
            )

        route_key = entry.route_key
        path_value = str(getattr(entry, split_path_field))
        previous_route = path_to_route.get(path_value)
        if previous_route is not None and previous_route != route_key:
            raise ValueError(
                f"Path {path_value!r} is mapped to conflicting routes in {split_name} split: "
                f"{previous_route!r} vs {route_key!r}."
            )
        path_to_route[path_value] = route_key
        resolved_paths.append(path_value)
        resolved_routes.append(route_key)

        _merge_route_option_default(
            route_option_defaults=route_option_defaults,
            route_key=route_key,
            mix_weight=entry.default_mix_weight,
        )
        if entry.prompt_profile is not None:
            _merge_route_prompt_default(
                route_prompt_defaults=route_prompt_defaults,
                route_key=route_key,
                prompt_profile=entry.prompt_profile,
                config_path=config_path,
            )

    return resolved_paths, resolved_routes


def _merge_route_option_default(
    *,
    route_option_defaults: dict[str, dict[str, Any]],
    route_key: str,
    mix_weight: float,
) -> None:
    incoming = {"mix_weight": float(mix_weight)}
    existing = route_option_defaults.get(route_key)
    if existing is None:
        route_option_defaults[route_key] = incoming
        return
    if float(existing.get("mix_weight", 0.0)) != incoming["mix_weight"]:
        raise ValueError(
            f"Conflicting default_mix_weight for route {route_key!r}: "
            f"{existing.get('mix_weight')!r} vs {incoming['mix_weight']!r}."
        )


def _merge_route_prompt_default(
    *,
    route_prompt_defaults: dict[str, dict[str, Any]],
    route_key: str,
    prompt_profile: str,
    config_path: Path,
) -> None:
    prompt_payload = load_prompt_profile_payload(prompt_profile, config_path=config_path)
    incoming = dict(prompt_payload)
    incoming["profile"] = str(prompt_profile)
    existing = route_prompt_defaults.get(route_key)
    if existing is None:
        route_prompt_defaults[route_key] = incoming
        return
    if existing != incoming:
        raise ValueError(
            f"Conflicting prompt defaults for route {route_key!r}. "
            f"Existing={existing.get('profile')!r}, incoming={prompt_profile!r}."
        )
