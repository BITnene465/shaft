from __future__ import annotations

import json
import shutil
from pathlib import Path

from .artifacts import BenchmarkArtifacts
from .schema import BenchmarkManifest, TaskKind


def _read_manifest_entries(path: Path) -> list[str]:
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        entries.append(item)
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


def create_benchmark_from_raw_data(
    *,
    store_root: str | Path,
    benchmark_id: str,
    tasks: list[TaskKind],
    source_root: str | Path,
    source_manifest: str | Path,
    split: str,
    layers: list[str] | None = None,
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
    copied_entries: list[str] = []
    labels: set[str] = set()
    seen: set[str] = set()
    for entry in split_entries:
        if entry in seen:
            continue
        seen.add(entry)
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

        target_json = artifacts.data_dir / json_relative_path
        image_relative_path = Path(str(payload.get("image_path") or ""))
        if not image_relative_path.parts:
            image_relative_path = json_relative_path.with_name(source_image.name).with_suffix(
                source_image.suffix
            )
            if len(json_relative_path.parts) >= 2 and json_relative_path.parts[1] == "json":
                image_relative_path = Path(json_relative_path.parts[0]) / "images" / source_image.name
        target_image = artifacts.data_dir / image_relative_path
        target_json.parent.mkdir(parents=True, exist_ok=True)
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_json, target_json)
        shutil.copy2(source_image, target_image)
        copied_entries.append(entry)

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
        layers=list(layers or []),
        labels=sorted(labels),
    )
    artifacts.write_manifest(manifest)
    return manifest
