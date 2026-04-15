from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


_PATH_KEYS = {"train_path", "val_path"}
_PATH_LIST_KEYS = {"train_paths", "val_paths"}


def _resolve_path_value(value: Any, *, base_dir: Path) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return text
    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_dataset_paths(dataset_payload: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(dataset_payload)
    for key in _PATH_KEYS:
        if key in resolved:
            resolved[key] = _resolve_path_value(resolved.get(key), base_dir=base_dir)
    for key in _PATH_LIST_KEYS:
        values = resolved.get(key)
        if values is None:
            continue
        if not isinstance(values, list):
            raise TypeError(f"Config key {key!r} must be a list when provided.")
        resolved[key] = [_resolve_path_value(value, base_dir=base_dir) for value in values]
    return resolved


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config registry root must be a mapping: {path}")
    return payload


def _coerce_registry_datasets(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_yaml_mapping(path)
    raw_datasets = payload.get("datasets", payload)
    if isinstance(raw_datasets, dict):
        items = raw_datasets.items()
    elif isinstance(raw_datasets, list):
        items = []
        for index, item in enumerate(raw_datasets):
            if not isinstance(item, dict):
                raise TypeError(f"Registry dataset entry must be a mapping: {path}:datasets[{index}]")
            dataset_name = str(item.get("name", "")).strip()
            if not dataset_name:
                raise ValueError(f"Registry dataset entry is missing name: {path}:datasets[{index}]")
            items.append((dataset_name, item))
    else:
        raise TypeError(f"Registry datasets must be a mapping or list: {path}")

    resolved: dict[str, dict[str, Any]] = {}
    for dataset_name, item in items:
        normalized_name = str(dataset_name).strip()
        if not normalized_name:
            raise ValueError(f"Registry dataset name cannot be empty: {path}")
        if not isinstance(item, dict):
            raise TypeError(f"Registry dataset {normalized_name!r} must be a mapping: {path}")
        current = copy.deepcopy(item)
        current.setdefault("name", normalized_name)
        resolved_name = str(current.get("name", "")).strip()
        if not resolved_name:
            raise ValueError(f"Registry dataset name cannot be empty: {path}")
        if resolved_name in resolved:
            raise ValueError(f"Duplicate dataset name {resolved_name!r} in registry: {path}")
        resolved[resolved_name] = _resolve_dataset_paths(current, base_dir=path.parent)
    return resolved


def _ensure_dataset_name(dataset_payload: dict[str, Any], *, scope: str) -> str:
    dataset_name = str(dataset_payload.get("name", "")).strip()
    if not dataset_name:
        raise ValueError(f"Dataset name cannot be empty in {scope}.")
    return dataset_name


def resolve_data_sources(payload: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    resolved_payload = copy.deepcopy(payload)
    data_payload = resolved_payload.get("data")
    if data_payload is None:
        return resolved_payload
    if not isinstance(data_payload, dict):
        raise TypeError("Config key `data` must be a mapping.")

    inline_datasets = data_payload.get("datasets", [])
    if inline_datasets is None:
        inline_datasets = []
    if not isinstance(inline_datasets, list):
        raise TypeError("Config key `data.datasets` must be a list.")
    resolved_inline_datasets: list[dict[str, Any]] = []
    for index, item in enumerate(inline_datasets):
        if not isinstance(item, dict):
            raise TypeError(f"Config key `data.datasets[{index}]` must be a mapping.")
        resolved_item = _resolve_dataset_paths(item, base_dir=config_path.parent)
        _ensure_dataset_name(resolved_item, scope=f"data.datasets[{index}]")
        resolved_inline_datasets.append(resolved_item)

    registry_path_value = data_payload.get("registry_path")
    dataset_refs = data_payload.get("dataset_refs", [])
    if dataset_refs is None:
        dataset_refs = []
    if not isinstance(dataset_refs, list):
        raise TypeError("Config key `data.dataset_refs` must be a list when provided.")
    normalized_refs = [str(item).strip() for item in dataset_refs if str(item).strip()]

    if normalized_refs and not registry_path_value:
        raise ValueError("Config key `data.registry_path` is required when `data.dataset_refs` is set.")

    resolved_registry_datasets: list[dict[str, Any]] = []
    if registry_path_value:
        registry_path = Path(str(registry_path_value))
        if not registry_path.is_absolute():
            registry_path = (config_path.parent / registry_path).resolve()
        if not registry_path.exists():
            raise FileNotFoundError(f"Data registry path not found: {registry_path}")
        registry_entries = _coerce_registry_datasets(registry_path)
        data_payload["registry_path"] = str(registry_path)
        for dataset_ref in normalized_refs:
            if dataset_ref not in registry_entries:
                available = ", ".join(sorted(registry_entries.keys()))
                raise KeyError(
                    f"Dataset ref {dataset_ref!r} not found in registry {registry_path}. "
                    f"Available: [{available}]"
                )
            resolved_registry_datasets.append(copy.deepcopy(registry_entries[dataset_ref]))

    dataset_name_to_scope: dict[str, str] = {}
    merged_datasets = [*resolved_registry_datasets, *resolved_inline_datasets]
    for index, dataset_payload in enumerate(merged_datasets):
        dataset_name = _ensure_dataset_name(dataset_payload, scope=f"data.datasets[{index}]")
        previous_scope = dataset_name_to_scope.get(dataset_name)
        current_scope = "data.dataset_refs" if index < len(resolved_registry_datasets) else "data.datasets"
        if previous_scope is not None:
            raise ValueError(
                f"Duplicate dataset name {dataset_name!r} across resolved data sources: "
                f"{previous_scope} and {current_scope}."
            )
        dataset_name_to_scope[dataset_name] = current_scope

    data_payload["dataset_refs"] = normalized_refs
    data_payload["datasets"] = merged_datasets
    resolved_payload["data"] = data_payload
    return resolved_payload
