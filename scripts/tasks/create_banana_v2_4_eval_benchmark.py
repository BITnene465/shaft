#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_BENCH_ROOT = REPO_ROOT / "projects" / "eval_bench"
if str(EVAL_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_BENCH_ROOT))

from eval_bench.benchmark import (  # noqa: E402
    BenchmarkSliceSpec,
    create_benchmark_suite_from_raw_data,
)
from eval_bench.artifacts import BenchmarkArtifacts  # noqa: E402
from eval_bench.schema import BenchmarkManifest  # noqa: E402


GROUNDING_SLICES = {
    "grounding_arrow": {
        "source_split": "grounding_val.txt",
        "required_layers": ["arrow"],
        "tasks": ["detection"],
        "layers": ["arrow"],
        "target_labels": ["arrow"],
    },
    "grounding_layout": {
        "source_split": "grounding_val.txt",
        "required_layers": ["layout"],
        "tasks": ["detection"],
        "layers": ["layout"],
        "target_labels": ["icon", "image", "shape"],
    },
    "grounding_shape": {
        "source_split": "grounding_val.txt",
        "required_layers": ["layout"],
        "tasks": ["detection"],
        "layers": ["layout"],
        "target_labels": ["shape"],
    },
    "grounding_icon_image": {
        "source_split": "grounding_val.txt",
        "required_layers": ["layout"],
        "tasks": ["detection"],
        "layers": ["layout"],
        "target_labels": ["icon", "image"],
    },
}


def _read_split(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _load_raw(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"raw json must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"structured row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def _resolve_structured_image_path(row: dict[str, Any], structured_path: Path) -> Path:
    image_path = Path(str(row.get("image_path") or ""))
    if image_path.is_absolute():
        return image_path
    candidates = [
        (structured_path.parent / image_path).resolve(),
        (structured_path.parent.parent / image_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"point_arrow structured image does not exist: {image_path}")


def add_point_arrow_crop_split(
    *,
    store_root: str | Path,
    benchmark_id: str,
    structured_path: str | Path,
) -> BenchmarkManifest:
    artifacts = BenchmarkArtifacts(store_root, benchmark_id)
    manifest_payload = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"benchmark manifest must be an object: {artifacts.manifest_path}")
    structured_file = Path(structured_path)
    if not structured_file.exists():
        raise FileNotFoundError(f"point_arrow structured val jsonl does not exist: {structured_file}")

    rows = _read_jsonl(structured_file)
    entries: list[str] = []
    labels: set[str] = {str(item) for item in manifest_payload.get("labels") or [] if str(item)}
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"point_arrow structured row is missing sample_id: {structured_file}")
        source_image = _resolve_structured_image_path(row, structured_file)
        image_relative = Path("point_arrow") / "images" / f"{sample_id}{source_image.suffix}"
        json_relative = Path("point_arrow") / "json" / f"{sample_id}.json"
        target_image = artifacts.data_dir / image_relative
        target_json = artifacts.data_dir / json_relative
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, target_image)
        instances = [dict(item) for item in row.get("instances") or [] if isinstance(item, dict)]
        for instance in instances:
            label = str(instance.get("label") or "").strip()
            if label:
                labels.add(label)
        gt_payload = {
            "image_path": str(image_relative),
            "image_width": int(row.get("image_width") or 0),
            "image_height": int(row.get("image_height") or 0),
            "instances": instances,
            "extra": {
                "task": "point_arrow",
                "view_type": "arrow_crop",
                "structured_sample_id": sample_id,
                "structured_source": str(structured_file),
                **dict(row.get("extra") or {}),
            },
        }
        target_json.write_text(
            json.dumps(gt_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        entries.append(str(json_relative))

    split_path = artifacts.split_path("point_arrow")
    split_path.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
    split_manifests = dict(manifest_payload.get("split_manifests") or {})
    split_manifests["point_arrow"] = str(split_path)
    sample_counts = dict(manifest_payload.get("sample_counts") or {})
    sample_counts["point_arrow"] = len(entries)
    tasks = list(manifest_payload.get("tasks") or [])
    if "keypoint" not in tasks:
        tasks.append("keypoint")
    layers = sorted({*(str(item) for item in manifest_payload.get("layers") or []), "arrow"})
    metadata = dict(manifest_payload.get("metadata") or {})
    source_manifest_paths = dict(metadata.get("source_manifest_paths") or {})
    source_manifest_paths["point_arrow"] = str(structured_file)
    slices = dict(metadata.get("slices") or {})
    slices["point_arrow"] = {
        "sample_count": len(entries),
        "source_manifest_path": str(structured_file),
        "tasks": ["keypoint"],
        "layers": ["arrow"],
        "target_labels": ["arrow"],
        "view_type": "arrow_crop",
        "source_type": "structured_point_arrow_val",
    }
    metadata["source_manifest_paths"] = source_manifest_paths
    metadata["slices"] = slices
    manifest = BenchmarkManifest(
        benchmark_id=str(manifest_payload["benchmark_id"]),
        tasks=tasks,  # type: ignore[arg-type]
        root=str(manifest_payload["root"]),
        split=str(manifest_payload["split"]),
        manifest_path=str(manifest_payload["manifest_path"]),
        sample_count=int(manifest_payload.get("sample_count") or 0),
        created_at=str(manifest_payload.get("created_at") or ""),
        source_raw_root=manifest_payload.get("source_raw_root"),
        source_manifest_path=manifest_payload.get("source_manifest_path"),
        split_manifests=split_manifests,
        sample_counts=sample_counts,
        layers=layers,
        labels=sorted(labels),
        metadata=metadata,
    )
    artifacts.write_manifest(manifest)
    return manifest


def _annotation_layers(payload: dict[str, Any]) -> set[str]:
    annotation = payload.get("annotation")
    if not isinstance(annotation, dict):
        return set()
    layers = {str(item).strip() for item in annotation.get("layers") or [] if str(item).strip()}
    status = annotation.get("status")
    if isinstance(status, dict):
        for layer, value in status.items():
            if str(value).strip().lower() in {"annotated", "complete", "done", "true"}:
                layers.add(str(layer).strip())
    return {item for item in layers if item}


def _filter_entries(raw_root: Path, entries: list[str], required_layers: list[str]) -> list[str]:
    required = {str(layer).strip() for layer in required_layers if str(layer).strip()}
    output: list[str] = []
    for entry in entries:
        payload = _load_raw(raw_root / entry)
        layers = _annotation_layers(payload)
        if required.issubset(layers):
            output.append(entry)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Banana v2.4 validation benchmark suite in Eval Bench."
    )
    parser.add_argument("--raw-root", default="data/raw_data")
    parser.add_argument("--point-arrow-structured-val", default="data/point_arrow/structured/val.jsonl")
    parser.add_argument("--store-root", default="eval_bench_store")
    parser.add_argument("--benchmark-id", default="banana_v2_4_val")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    split_root = raw_root / "splits"
    source_entries = {
        "grounding_val.txt": _read_split(split_root / "grounding_val.txt"),
    }
    slices: list[BenchmarkSliceSpec] = []
    for split_name, spec in GROUNDING_SLICES.items():
        entries = _filter_entries(
            raw_root,
            source_entries[str(spec["source_split"])],
            list(spec["required_layers"]),
        )
        slices.append(
            BenchmarkSliceSpec(
                split=split_name,
                source_manifest=split_root / str(spec["source_split"]),
                entries=entries,
                tasks=list(spec["tasks"]),  # type: ignore[arg-type]
                layers=list(spec["layers"]),
                target_labels=list(spec["target_labels"]),
                metadata={"required_layers": list(spec["required_layers"])},
            )
        )

    manifest = create_benchmark_suite_from_raw_data(
        store_root=args.store_root,
        benchmark_id=args.benchmark_id,
        source_root=raw_root,
        slices=slices,
        split="suite",
        default_slice="grounding_arrow",
        layers=["layout", "arrow"],
        flatten=True,
        overwrite=bool(args.overwrite),
        metadata={
            "benchmark_family": "banana",
            "benchmark_version": "v2.4",
            "source_splits": {"grounding": str(split_root / "grounding_val.txt")},
        },
    )
    manifest = add_point_arrow_crop_split(
        store_root=args.store_root,
        benchmark_id=args.benchmark_id,
        structured_path=args.point_arrow_structured_val,
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
