#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    "grounding_shape_arrow": {
        "source_split": "grounding_val.txt",
        "required_layers": ["layout", "arrow"],
        "tasks": ["detection"],
        "layers": ["layout", "arrow"],
        "target_labels": ["shape", "arrow"],
    },
    "point_arrow": {
        "source_split": "point_arrow_val.txt",
        "required_layers": ["arrow"],
        "tasks": ["keypoint"],
        "layers": ["arrow"],
        "target_labels": ["arrow"],
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    split_root = raw_root / "splits"
    source_entries = {
        "grounding_val.txt": _read_split(split_root / "grounding_val.txt"),
        "point_arrow_val.txt": _read_split(split_root / "point_arrow_val.txt"),
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
            "source_splits": {
                "grounding": str(split_root / "grounding_val.txt"),
                "point_arrow": str(split_root / "point_arrow_val.txt"),
            },
        },
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
