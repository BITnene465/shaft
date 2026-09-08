#!/usr/bin/env python3
"""Deterministic line calibration boards; never a production augmentation recipe."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, features

import build_context_reconstruction_sft as context
import prepare_gt_standard_v5_7 as contract
from shaft.data.synthetic_realism import apply_synthetic_realism_augmentation


REPO = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plans(size: tuple[int, int], identity: str, recipe: dict[str, Any]) -> dict[str, Any]:
    result = {}
    noise_seed = int.from_bytes(
        hashlib.sha256(f"{recipe['seed']}:{identity}:noise".encode()).digest()[:8], "big"
    )
    for level, values in recipe["levels"].items():
        base = {
            "profile": "synthetic_realism_v1",
            "severity": level,
            "dimensions_unchanged": True,
            "input_size": list(size),
            "output_size": list(size),
        }
        result[f"stack_{level}"] = {
            **base,
            "operations": [
                {
                    "name": "resample_roundtrip",
                    "scale_down_ratio": values["resample_ratio"],
                    "down_kernel": "BICUBIC",
                    "up_kernel": "BICUBIC",
                },
                {"name": "gaussian_noise", "sigma_255": values["noise_sigma"], "seed": noise_seed},
                {"name": "jpeg_compression", "quality": values["jpeg_quality"], "subsampling": 1},
            ],
        }
        result[f"blur_{level}"] = {
            **base,
            "operations": [{"name": "gaussian_blur", "radius": values["blur_radius"]}],
        }
    return result


def select(root: Path, inventory: dict[str, Any], count: int) -> list[dict[str, Any]]:
    audit = inventory["splits"]["train"]
    identities = sorted(
        {tuple(value) for key, value in audit["examples"].items() if key.startswith("stratum:")}
    )
    candidates = []
    for stem, index in identities:
        source = root / "gt_standard" / f"{stem}.json"
        if sha(source) != audit["gt_sha256"][stem]:
            raise ValueError(f"GT changed since audit: {stem}")
        doc = json.loads(source.read_text())
        obj = doc["layout"][index]
        p = obj["parameters"]
        width, height = doc["size"]
        if contract._validate_line(p, width=width, height=height):
            raise ValueError(f"Audited example is invalid: {stem}:{index}")
        bbox = obj["bbox"]
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        color = p.get("fill", {}).get("color", "#000000")
        luminance = (
            sum(int(color[i : i + 2], 16) for i in (1, 3, 5)) / 765
            if isinstance(color, str) and len(color) == 7
            else 0
        )
        candidates.append(
            {
                "stem": stem,
                "index": index,
                "p": p,
                "bbox": bbox,
                "area": bw * bh,
                "short_edge": min(bw, bh),
                "lightness": luminance,
            }
        )
    rules = {
        "dense_multi": (
            lambda c: len(c["p"]["points"]) >= 10,
            lambda c: (-len(c["p"]["points"]), c["area"]),
        ),
        "curved": (lambda c: c["p"]["line_type"] == "curved", lambda c: c["area"]),
        "dashed": (lambda c: c["p"]["dash_style"] == "dash", lambda c: c["area"]),
        "small_arrow": (lambda c: c["p"]["end_arrow"] != "none", lambda c: c["area"]),
        "slender_path_proxy": (
            lambda c: c["p"]["line_style"] == "path",
            lambda c: (c["short_edge"], c["area"]),
        ),
        "light_color_proxy": (lambda c: c["lightness"] > 0.55, lambda c: -c["lightness"]),
    }
    selected, seen = [], set()
    for category, (predicate, rank) in rules.items():
        eligible = sorted(
            (c for c in candidates if predicate(c)), key=lambda c: (rank(c), c["stem"], c["index"])
        )
        picked = 0
        for c in eligible:
            identity = (c["stem"], c["index"])
            if identity in seen:
                continue
            selected.append({"category": category, "stem": c["stem"], "index": c["index"]})
            seen.add(identity)
            picked += 1
            if picked == count:
                break
        if picked != count:
            raise ValueError(f"Insufficient examples for {category}: {picked}")
    return selected


def render(item: tuple[Path, Path, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    root, output, selection, recipe = item
    stem, index = selection["stem"], selection["index"]
    identity = f"{stem}__line_{index:04d}"
    doc = json.loads((root / "gt_standard" / f"{stem}.json").read_text())
    obj = doc["layout"][index]
    width, height = doc["size"]
    original_bbox = tuple(obj["bbox"])
    bbox = list(original_bbox)
    # Locator padding only; source points and source annotation are never modified.
    for axis, limit in ((0, width), (1, height)):
        if bbox[axis] == bbox[axis + 2]:
            bbox[axis] = max(0, bbox[axis] - 1)
            bbox[axis + 2] = min(limit, bbox[axis + 2] + 1)
    view = context._sample_context_view(
        source_bbox=tuple(bbox),
        image_width=width,
        image_height=height,
        task="line_context_reconstruction",
        sample_id=identity,
        seed=recipe["seed"],
        max_aspect_ratio=60,
        geometry_bbox=context._geometry_bbox("line", tuple(bbox), obj["parameters"]),
    )
    folder = output / f"{selection['category']}_{identity}"
    folder.mkdir()
    with Image.open(root / "img" / f"{stem}.png") as source:
        if source.size != (width, height):
            raise ValueError(f"Dimension mismatch: {stem}")
        clean = source.convert("RGB").crop(view.crop_box)
    clean.save(folder / "clean.png")
    augmentations = plans(clean.size, identity, recipe)
    for name, plan in augmentations.items():
        image = apply_synthetic_realism_augmentation(clean, plan)
        if image.size != clean.size:
            raise ValueError("Augmentation changed dimensions")
        image.save(folder / f"{name}.png")
        image.close()
    for family in ("stack", "blur"):
        names = ["clean"] + [f"{family}_{level}" for level in recipe["levels"]]
        panel_width, panel_height = min(640, clean.width), min(460, clean.height)
        panel_width = max(panel_width, 190)
        board = Image.new("RGB", (4 * panel_width, panel_height + 55), "#eeeeee")
        draw = ImageDraw.Draw(board)
        for column, name in enumerate(names):
            with Image.open(folder / f"{name}.png") as panel:
                panel.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
                board.paste(panel, (column * panel_width, 50))
            draw.text((column * panel_width + 4, 5), name, fill="black")
            draw.text((column * panel_width + 4, 23), identity, fill="black")
        board.save(folder / f"compare_{family}.png")
        board.close()
    metadata = {
        **selection,
        "sample_id": identity,
        "source_bbox": original_bbox,
        "locator_bbox": bbox,
        "crop_box": view.crop_box,
        "native_size": clean.size,
        "proposal_bbox": view.proposal_bbox,
        "plans": augmentations,
        "source_image_sha256": sha(root / "img" / f"{stem}.png"),
        "files_sha256": {p.name: sha(p) for p in sorted(folder.glob("*.png"))},
    }
    (folder / "plans.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    clean.close()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=REPO / "configs/data/preparation/banana_v5_10_line_preview.json",
    )
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if not recipe.get("review_only") or args.workers < 1:
        raise ValueError("Review-only recipe and positive workers required")
    inventory = json.loads(args.inventory.read_text())
    selections = select(args.synthetic_root, inventory, recipe["examples_per_category"])
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(
            pool.map(render, [(args.synthetic_root, args.output, s, recipe) for s in selections])
        )
    report = {
        "environment": {
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in ("Pillow", "numpy")},
            "codecs": {name: features.version(name) for name in ("jpg", "zlib", "libjpeg_turbo")},
        },
        "recipe": recipe,
        "inventory_sha256": sha(args.inventory),
        "script_sha256": sha(Path(__file__)),
        "rows": rows,
        "pixel_operator_sha256": sha(REPO / "src/shaft/data/synthetic_realism.py"),
        "context_builder_sha256": sha(Path(context.__file__)),
        "note": "Review only. Clean twins and fixed severity levels are not production sampling.",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"examples": len(rows), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
