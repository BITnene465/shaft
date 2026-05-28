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
from eval_bench.schema import BenchmarkManifest, utc_now_iso  # noqa: E402


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
    parser.add_argument("--store-root", default="eval_bench_store")
    parser.add_argument("--benchmark-id", default="banana_v2_4_val")
    parser.add_argument("--point-arrow-structured", default="data/point_arrow/structured/val.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_point_arrow_crop_split(
    *,
    store_root: str | Path,
    benchmark_id: str,
    structured_path: str | Path,
) -> BenchmarkManifest:
    artifacts = BenchmarkArtifacts(store_root, benchmark_id)
    structured = Path(structured_path)
    if not artifacts.manifest_path.exists():
        raise FileNotFoundError(f"benchmark manifest does not exist: {artifacts.manifest_path}")
    if not structured.exists():
        raise FileNotFoundError(f"point_arrow structured val does not exist: {structured}")

    manifest_payload = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"benchmark manifest must be an object: {artifacts.manifest_path}")

    split_entries: list[str] = []
    labels = {str(label) for label in manifest_payload.get("labels") or [] if str(label).strip()}
    for line_number, line in enumerate(structured.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"point_arrow structured line {line_number} must be an object.")
        sample_id = str(payload.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"point_arrow structured line {line_number} missing sample_id.")
        source_image = _resolve_structured_image_path(structured, payload, line_number=line_number)
        target_json_relative = Path("point_arrow") / "json" / f"{sample_id}.json"
        target_image_relative = Path("point_arrow") / "images" / f"{sample_id}{source_image.suffix}"
        target_payload = dict(payload)
        target_payload["image_path"] = str(target_image_relative)
        extra = dict(target_payload.get("extra") or {})
        extra.setdefault("task", "point_arrow")
        extra.setdefault("split", "val")
        extra.setdefault("view_type", "arrow_crop")
        extra.setdefault("structured_source_path", str(structured))
        target_payload["extra"] = extra
        for instance in target_payload.get("instances") or []:
            if isinstance(instance, dict):
                label = str(instance.get("label") or "").strip()
                if label:
                    labels.add(label)

        target_json = artifacts.data_dir / target_json_relative
        target_image = artifacts.data_dir / target_image_relative
        target_json.parent.mkdir(parents=True, exist_ok=True)
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_json.write_text(
            json.dumps(target_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(source_image, target_image)
        split_entries.append(str(target_json_relative))

    split_path = artifacts.split_path("point_arrow")
    split_path.write_text(
        "\n".join(split_entries) + ("\n" if split_entries else ""),
        encoding="utf-8",
    )

    split_manifests = dict(manifest_payload.get("split_manifests") or {})
    split_manifests["point_arrow"] = str(split_path)
    sample_counts = {str(key): int(value) for key, value in (manifest_payload.get("sample_counts") or {}).items()}
    sample_counts["point_arrow"] = len(split_entries)
    tasks = list(manifest_payload.get("tasks") or [])
    if "keypoint" not in tasks:
        tasks.append("keypoint")
    layers = list(manifest_payload.get("layers") or [])
    if "arrow" not in layers:
        layers.append("arrow")
    metadata = dict(manifest_payload.get("metadata") or {})
    source_manifest_paths = dict(metadata.get("source_manifest_paths") or {})
    source_manifest_paths["point_arrow"] = str(structured)
    slices = dict(metadata.get("slices") or {})
    slices["point_arrow"] = {
        "sample_count": len(split_entries),
        "source_manifest_path": str(structured),
        "tasks": ["keypoint"],
        "layers": ["arrow"],
        "target_labels": ["arrow"],
        "view_type": "arrow_crop",
    }
    metadata["source_manifest_paths"] = source_manifest_paths
    metadata["slices"] = slices
    updated = BenchmarkManifest(
        benchmark_id=str(manifest_payload["benchmark_id"]),
        tasks=tasks,
        root=str(manifest_payload["root"]),
        split=str(manifest_payload["split"]),
        manifest_path=str(manifest_payload["manifest_path"]),
        sample_count=int(manifest_payload["sample_count"]),
        created_at=str(manifest_payload.get("created_at") or utc_now_iso()),
        source_raw_root=manifest_payload.get("source_raw_root"),
        source_manifest_path=manifest_payload.get("source_manifest_path"),
        split_manifests=split_manifests,
        sample_counts=sample_counts,
        layers=sorted(set(str(layer) for layer in layers if str(layer).strip())),
        labels=sorted(labels),
        metadata=metadata,
    )
    artifacts.write_manifest(updated)
    return updated


def _resolve_structured_image_path(
    structured_path: Path,
    payload: dict[str, Any],
    *,
    line_number: int,
) -> Path:
    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError(f"point_arrow structured line {line_number} missing image_path.")
    source_image = Path(image_path)
    if not source_image.is_absolute():
        source_image = structured_path.parent / source_image
    source_image = source_image.resolve()
    if not source_image.exists():
        raise FileNotFoundError(
            f"point_arrow structured line {line_number} image does not exist: {source_image}"
        )
    return source_image


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
        structured_path=args.point_arrow_structured,
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
