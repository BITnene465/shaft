from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import shutil
from pathlib import Path
from typing import Any

from .artifacts import BenchmarkArtifacts
from .schema import BenchmarkManifest, TaskKind


@dataclass(frozen=True)
class BenchmarkSliceSpec:
    split: str
    tasks: list[TaskKind]
    source_manifest: str | Path | None = None
    entries: list[str] | None = None
    layers: list[str] = field(default_factory=list)
    target_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_manifest_entries(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        return _read_json_manifest_entries(path)
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        entries.append(item)
    return entries


def _read_json_manifest_entries(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON split manifest must be an object: {path}")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"JSON split manifest must contain an items list: {path}")
    entries: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"JSON split manifest item {index} must be an object: {path}")
        explicit_json_path = item.get("json_path") or item.get("annotation_path")
        if isinstance(explicit_json_path, str) and explicit_json_path.strip():
            entries.append(explicit_json_path.strip())
            continue
        sample_id = str(item.get("id") or "").strip()
        if not sample_id:
            image_path = str(item.get("image_path") or "").strip()
            sample_id = Path(image_path).stem
        if not sample_id:
            raise ValueError(
                f"JSON split manifest item {index} needs json_path, annotation_path, id, "
                f"or image_path: {path}"
            )
        entries.append(str(Path("json") / f"{sample_id}.json"))
    return entries


def _load_raw_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"raw json must be an object: {path}")
    return payload


def _resolve_image_path(raw_root: Path, json_relative_path: Path, payload: dict) -> Path:
    image_path = payload.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        resolved = raw_root / image_path
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"image_path in {json_relative_path} does not exist: {image_path}")

    stem = json_relative_path.stem
    part = json_relative_path.parts[0] if json_relative_path.parts else ""
    image_dir = raw_root / part / "images"
    matches = sorted(image_dir.glob(f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(f"could not infer image for raw json: {json_relative_path}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous image files for raw json {json_relative_path}: {matches}")
    return matches[0]


def _flattened_sample_stem(json_relative_path: Path) -> str:
    parts = json_relative_path.parts
    prefix = parts[0] if parts else "sample"
    return f"{prefix}__{json_relative_path.stem}"


def resolve_benchmark_split_path(
    benchmark: Mapping[str, Any],
    *,
    split: str | None = None,
) -> Path:
    requested = str(split or benchmark.get("split") or "").strip()
    split_manifests = benchmark.get("split_manifests")
    if isinstance(split_manifests, Mapping) and split_manifests:
        if not requested:
            available = ", ".join(sorted(str(item) for item in split_manifests))
            raise FileNotFoundError(
                f"benchmark split is not declared; available splits: {available}"
            )
        manifest = split_manifests.get(requested)
        if isinstance(manifest, str) and manifest.strip():
            return Path(manifest)
        available = ", ".join(sorted(str(item) for item in split_manifests))
        raise FileNotFoundError(
            f"benchmark split {requested!r} is not available; available splits: {available}"
        )
    manifest_path = str(benchmark.get("manifest_path") or "").strip()
    if manifest_path:
        return Path(manifest_path)
    raise FileNotFoundError("benchmark manifest does not declare a split manifest path.")


def infer_benchmark_split(*, task: str, prompt_id: str, target_labels: Sequence[str]) -> str:
    prompt = prompt_id.strip().lower()
    labels = tuple(sorted(label.strip().lower() for label in target_labels if label.strip()))
    if task == "keypoint" or prompt.startswith(
        ("point_arrow.", "keypoint_arrow.", "arrow_keypoint.")
    ):
        return "point_arrow"
    if prompt.startswith("grounding_icon_image.") or labels == ("icon", "image"):
        return "grounding_icon_image"
    if prompt.startswith("grounding_shape.") or labels == ("shape",):
        return "grounding_shape"
    if prompt.startswith("grounding_layout.") or labels == ("icon", "image", "shape"):
        return "grounding_layout"
    if prompt.startswith("grounding_arrow.") or labels == ("arrow",):
        return "grounding_arrow"
    return ""


def resolve_benchmark_split_name(
    benchmark: Mapping[str, Any],
    *,
    split: str | None = None,
    task: str = "",
    prompt_id: str = "",
    target_labels: Sequence[str] = (),
) -> str:
    explicit = str(split or "").strip()
    if explicit:
        return explicit
    split_manifests = benchmark.get("split_manifests")
    if not isinstance(split_manifests, Mapping) or not split_manifests:
        return str(benchmark.get("split") or "val")
    inferred = infer_benchmark_split(
        task=task,
        prompt_id=prompt_id,
        target_labels=target_labels,
    )
    if inferred in split_manifests:
        return inferred
    default_split = str(benchmark.get("split") or "").strip()
    if default_split and default_split in split_manifests:
        return default_split
    available = ", ".join(sorted(str(item) for item in split_manifests))
    raise FileNotFoundError(
        f"benchmark default split {default_split!r} is not available; available splits: {available}"
    )


def _copy_raw_entries(
    *,
    artifacts: BenchmarkArtifacts,
    raw_root: Path,
    entries: Sequence[str],
    flatten: bool = False,
) -> tuple[dict[str, str], set[str]]:
    entry_map: dict[str, str] = {}
    labels: set[str] = set()
    for entry in entries:
        json_relative_path = Path(entry)
        source_json = raw_root / json_relative_path
        if not source_json.exists():
            raise FileNotFoundError(f"raw json listed in manifest does not exist: {entry}")
        payload = _load_raw_json(source_json)
        for instance in payload.get("instances") or []:
            if not isinstance(instance, dict):
                continue
            label = str(instance.get("label") or "").strip()
            if label:
                labels.add(label)
        source_image = _resolve_image_path(raw_root, json_relative_path, payload)

        if flatten:
            sample_stem = _flattened_sample_stem(json_relative_path)
            target_json_relative = Path("json") / f"{sample_stem}.json"
            image_relative_path = Path("images") / f"{sample_stem}{source_image.suffix}"
            payload = dict(payload)
            extra = dict(payload.get("extra") or {})
            extra.setdefault("source_json", str(json_relative_path))
            extra.setdefault("source_image", str(source_image.relative_to(raw_root)))
            payload["extra"] = extra
            payload["image_path"] = str(image_relative_path)
        else:
            target_json_relative = json_relative_path
            image_relative_path = Path(str(payload.get("image_path") or ""))
            if not image_relative_path.parts:
                image_relative_path = json_relative_path.with_name(source_image.name).with_suffix(
                    source_image.suffix
                )
                if len(json_relative_path.parts) >= 2 and json_relative_path.parts[1] == "json":
                    image_relative_path = Path(json_relative_path.parts[0]) / "images" / source_image.name
        target_json = artifacts.data_dir / target_json_relative
        target_image = artifacts.data_dir / image_relative_path
        target_json.parent.mkdir(parents=True, exist_ok=True)
        target_image.parent.mkdir(parents=True, exist_ok=True)
        if flatten:
            target_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            shutil.copy2(source_json, target_json)
        shutil.copy2(source_image, target_image)
        entry_map[entry] = str(target_json_relative)
    return entry_map, labels


def create_benchmark_from_raw_data(
    *,
    store_root: str | Path,
    benchmark_id: str,
    tasks: list[TaskKind],
    source_root: str | Path,
    source_manifest: str | Path,
    split: str,
    layers: list[str] | None = None,
    flatten: bool = False,
    overwrite: bool = False,
) -> BenchmarkManifest:
    artifacts = BenchmarkArtifacts(store_root, benchmark_id)
    raw_root = Path(source_root)
    source_manifest_path = Path(source_manifest)
    if not raw_root.exists():
        raise FileNotFoundError(f"source raw root does not exist: {raw_root}")
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"source manifest does not exist: {source_manifest_path}")
    if artifacts.benchmark_dir.exists() and not overwrite:
        raise FileExistsError(
            f"benchmark already exists: {artifacts.benchmark_dir}. Pass --overwrite to replace files."
        )
    if artifacts.benchmark_dir.exists() and overwrite:
        shutil.rmtree(artifacts.benchmark_dir)

    artifacts.ensure()
    split_entries = _read_manifest_entries(source_manifest_path)
    seen: set[str] = set()
    unique_entries = [entry for entry in split_entries if not (entry in seen or seen.add(entry))]
    entry_map, labels = _copy_raw_entries(
        artifacts=artifacts,
        raw_root=raw_root,
        entries=unique_entries,
        flatten=flatten,
    )
    copied_entries = [entry_map[entry] for entry in unique_entries]

    split_path = artifacts.split_path(split)
    split_path.write_text("\n".join(copied_entries) + ("\n" if copied_entries else ""), encoding="utf-8")
    manifest = BenchmarkManifest(
        benchmark_id=benchmark_id,
        tasks=tasks,
        root=str(artifacts.data_dir),
        split=split,
        manifest_path=str(split_path),
        sample_count=len(copied_entries),
        source_raw_root=str(raw_root),
        source_manifest_path=str(source_manifest_path),
        split_manifests={split: str(split_path)},
        sample_counts={split: len(copied_entries)},
        layers=list(layers or []),
        labels=sorted(labels),
    )
    artifacts.write_manifest(manifest)
    return manifest


def create_benchmark_suite_from_raw_data(
    *,
    store_root: str | Path,
    benchmark_id: str,
    source_root: str | Path,
    slices: Sequence[BenchmarkSliceSpec],
    split: str = "suite",
    default_slice: str | None = None,
    layers: list[str] | None = None,
    flatten: bool = False,
    overwrite: bool = False,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkManifest:
    if not slices:
        raise ValueError("slices must be a non-empty sequence.")
    artifacts = BenchmarkArtifacts(store_root, benchmark_id)
    raw_root = Path(source_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"source raw root does not exist: {raw_root}")
    if artifacts.benchmark_dir.exists() and not overwrite:
        raise FileExistsError(
            f"benchmark already exists: {artifacts.benchmark_dir}. Pass --overwrite to replace files."
        )
    if artifacts.benchmark_dir.exists() and overwrite:
        shutil.rmtree(artifacts.benchmark_dir)
    artifacts.ensure()

    split_entries: dict[str, list[str]] = {}
    source_manifests: dict[str, str] = {}
    union_entries: list[str] = []
    seen_union: set[str] = set()
    all_tasks: list[TaskKind] = []
    all_layers: list[str] = []
    slice_metadata: dict[str, Any] = {}
    for item in slices:
        normalized_split = str(item.split).strip()
        if not normalized_split:
            raise ValueError("slice split must be a non-empty string.")
        source_manifest = Path(item.source_manifest) if item.source_manifest is not None else None
        if item.entries is None:
            if source_manifest is None:
                raise ValueError(f"slice {normalized_split!r} needs source_manifest or entries.")
            if not source_manifest.exists():
                raise FileNotFoundError(f"source manifest does not exist: {source_manifest}")
            entries = _read_manifest_entries(source_manifest)
        else:
            entries = [str(entry).strip() for entry in item.entries if str(entry).strip()]
        seen_split: set[str] = set()
        unique_entries = [entry for entry in entries if not (entry in seen_split or seen_split.add(entry))]
        split_entries[normalized_split] = unique_entries
        source_manifests[normalized_split] = str(source_manifest) if source_manifest is not None else ""
        for entry in unique_entries:
            if entry not in seen_union:
                seen_union.add(entry)
                union_entries.append(entry)
        for task in item.tasks:
            if task not in all_tasks:
                all_tasks.append(task)
        for layer in item.layers:
            if layer not in all_layers:
                all_layers.append(layer)
        slice_metadata[normalized_split] = {
            "sample_count": len(unique_entries),
            "source_manifest_path": str(source_manifest) if source_manifest is not None else "",
            "tasks": list(item.tasks),
            "layers": list(item.layers),
            "target_labels": list(item.target_labels),
            **dict(item.metadata),
        }

    entry_map, labels = _copy_raw_entries(
        artifacts=artifacts,
        raw_root=raw_root,
        entries=union_entries,
        flatten=flatten,
    )
    split_manifests: dict[str, str] = {}
    sample_counts: dict[str, int] = {}
    for split_name, entries in split_entries.items():
        mapped_entries = [entry_map[entry] for entry in entries]
        split_path = artifacts.split_path(split_name)
        split_path.write_text(
            "\n".join(mapped_entries) + ("\n" if mapped_entries else ""),
            encoding="utf-8",
        )
        split_manifests[split_name] = str(split_path)
        sample_counts[split_name] = len(entries)

    resolved_default = default_slice or slices[0].split
    if resolved_default not in split_manifests:
        raise ValueError(f"default_slice must be one of: {sorted(split_manifests)}")
    top_level_split = str(split or "").strip() or resolved_default
    if top_level_split in split_manifests:
        manifest_path = split_manifests[top_level_split]
        sample_count = sample_counts[top_level_split]
    else:
        suite_entries = [entry_map[entry] for entry in union_entries]
        suite_path = artifacts.split_path(top_level_split)
        suite_path.write_text(
            "\n".join(suite_entries) + ("\n" if suite_entries else ""),
            encoding="utf-8",
        )
        split_manifests[top_level_split] = str(suite_path)
        sample_counts[top_level_split] = len(suite_entries)
        manifest_path = str(suite_path)
        sample_count = len(suite_entries)
    manifest_metadata = dict(metadata or {})
    manifest_metadata.setdefault("source_manifest_paths", source_manifests)
    manifest_metadata.setdefault("slices", slice_metadata)
    manifest = BenchmarkManifest(
        benchmark_id=benchmark_id,
        tasks=all_tasks,
        root=str(artifacts.data_dir),
        split=top_level_split,
        manifest_path=manifest_path,
        sample_count=sample_count,
        source_raw_root=str(raw_root),
        source_manifest_path=source_manifests.get(str(resolved_default)),
        split_manifests=split_manifests,
        sample_counts=sample_counts,
        layers=sorted(set([*(layers or []), *all_layers])),
        labels=sorted(labels),
        metadata=manifest_metadata,
    )
    artifacts.write_manifest(manifest)
    return manifest
