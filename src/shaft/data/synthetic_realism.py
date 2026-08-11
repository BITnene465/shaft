from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


SYNTHETIC_REALISM_PROFILE = "synthetic_realism_v1"
PIXEL_OPERATION_WEIGHTS = {
    "resample_roundtrip": 0.35,
    "gaussian_blur": 0.20,
    "gaussian_noise": 0.20,
    "jpeg_compression": 0.25,
}
PIXEL_OPERATION_ORDER = tuple(PIXEL_OPERATION_WEIGHTS)


def _choose_bucket(
    rng: random.Random,
    buckets: tuple[tuple[str, float, float, float], ...]
    | tuple[tuple[int, float, float, float], ...],
) -> tuple[Any, ...]:
    draw = rng.random()
    cumulative = 0.0
    for bucket in buckets:
        cumulative += float(bucket[1])
        if draw < cumulative:
            return bucket
    return buckets[-1]


def _weighted_sample_without_replacement(
    rng: random.Random,
    weights: dict[str, float],
    *,
    count: int,
) -> list[str]:
    available = dict(weights)
    selected: list[str] = []
    for _ in range(min(count, len(available))):
        total = sum(available.values())
        draw = rng.random() * total
        cumulative = 0.0
        choice = next(iter(available))
        for name, weight in available.items():
            cumulative += weight
            if draw < cumulative:
                choice = name
                break
        selected.append(choice)
        del available[choice]
    return selected


def sample_synthetic_realism_augmentation(
    *,
    task: str,
    sample_id: str,
    seed: int,
    target_short_span: int,
    image_width: int,
    image_height: int,
    required_operations: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Create a deterministic, non-empty synthetic pixel-degradation plan."""

    rng = random.Random(f"{seed}:synthetic-pixel:{task}:{sample_id}")
    if target_short_span < 80:
        severity = "mild"
        stack_depth = 1
    elif target_short_span < 200:
        severity = str(
            _choose_bucket(rng, (("mild", 0.70, 0.0, 0.0), ("moderate", 0.30, 0.0, 0.0)))[0]
        )
        stack_depth = int(
            _choose_bucket(rng, ((1, 0.40, 0.0, 0.0), (2, 0.50, 0.0, 0.0), (3, 0.10, 0.0, 0.0)))[0]
        )
    else:
        severity = str(
            _choose_bucket(
                rng,
                (
                    ("mild", 0.45, 0.0, 0.0),
                    ("moderate", 0.40, 0.0, 0.0),
                    ("strong", 0.15, 0.0, 0.0),
                ),
            )[0]
        )
        depth_buckets = {
            "mild": ((1, 0.15, 0.0, 0.0), (2, 0.55, 0.0, 0.0), (3, 0.30, 0.0, 0.0)),
            "moderate": ((1, 0.20, 0.0, 0.0), (2, 0.60, 0.0, 0.0), (3, 0.20, 0.0, 0.0)),
            "strong": ((1, 0.45, 0.0, 0.0), (2, 0.55, 0.0, 0.0)),
        }
        stack_depth = int(_choose_bucket(rng, depth_buckets[severity])[0])

    selected = _weighted_sample_without_replacement(
        rng,
        PIXEL_OPERATION_WEIGHTS,
        count=stack_depth,
    )
    unknown_required = sorted(set(required_operations) - set(PIXEL_OPERATION_WEIGHTS))
    if unknown_required:
        raise ValueError(f"Unsupported required synthetic operations: {unknown_required}")
    if len(set(required_operations)) > 3:
        raise ValueError("At most three required synthetic operations are supported.")
    for required in required_operations:
        if required in selected:
            continue
        if len(selected) < 3:
            selected.append(required)
        else:
            selected[-1] = required
    selected = list(dict.fromkeys(selected))
    selected.sort(key=PIXEL_OPERATION_ORDER.index)
    short_edge = min(image_width, image_height)
    operations: list[dict[str, Any]] = []
    for name in selected:
        if name == "resample_roundtrip":
            ratio_ranges = {
                "mild": (0.82, 0.96),
                "moderate": (0.62, 0.85),
                "strong": (0.42, 0.68),
            }
            kernels = {
                "mild": ("BICUBIC", "LANCZOS"),
                "moderate": ("BILINEAR", "BICUBIC", "LANCZOS"),
                "strong": ("BILINEAR", "BICUBIC"),
            }
            low, high = ratio_ranges[severity]
            operations.append(
                {
                    "name": name,
                    "scale_down_ratio": round(rng.uniform(low, high), 6),
                    "down_kernel": rng.choice(kernels[severity]),
                    "up_kernel": rng.choice(kernels[severity]),
                }
            )
        elif name == "gaussian_blur":
            ranges = {
                "mild": (max(0.25, short_edge * 0.00020), max(0.55, short_edge * 0.00045)),
                "moderate": (max(0.55, short_edge * 0.00045), max(1.10, short_edge * 0.00090)),
                "strong": (max(1.00, short_edge * 0.00085), max(1.75, short_edge * 0.00140)),
            }
            low, high = ranges[severity]
            operations.append({"name": name, "radius": round(rng.uniform(low, high), 6)})
        elif name == "gaussian_noise":
            sigma_ranges = {"mild": (1.0, 3.0), "moderate": (3.0, 7.0), "strong": (7.0, 12.0)}
            low, high = sigma_ranges[severity]
            operations.append(
                {
                    "name": name,
                    "sigma_255": round(rng.uniform(low, high), 6),
                    "seed": rng.getrandbits(63),
                }
            )
        elif name == "jpeg_compression":
            quality_ranges = {"mild": (82, 95), "moderate": (62, 84), "strong": (42, 68)}
            low, high = quality_ranges[severity]
            subsampling = rng.choice((0, 1)) if severity == "mild" else rng.choice((1, 2))
            operations.append(
                {
                    "name": name,
                    "quality": rng.randint(low, high),
                    "subsampling": subsampling,
                }
            )
    if not operations:
        raise RuntimeError("Synthetic pixel augmentation must contain at least one operation.")
    return {
        "profile": SYNTHETIC_REALISM_PROFILE,
        "severity": severity,
        "operations": operations,
        "dimensions_unchanged": True,
        "input_size": [image_width, image_height],
        "output_size": [image_width, image_height],
    }


def _resampling(name: str) -> Any:
    namespace = getattr(Image, "Resampling", Image)
    return getattr(namespace, name)


def apply_synthetic_realism_augmentation(
    image: Image.Image,
    augmentation: dict[str, Any],
) -> Image.Image:
    """Apply a synthetic-realism plan while preserving the input dimensions."""

    if augmentation.get("profile") != SYNTHETIC_REALISM_PROFILE:
        raise ValueError(f"Unsupported synthetic pixel profile: {augmentation.get('profile')!r}")
    current = image.convert("RGB")
    for operation in augmentation.get("operations") or []:
        name = operation.get("name")
        if name == "resample_roundtrip":
            width, height = current.size
            ratio = float(operation["scale_down_ratio"])
            down_size = (
                max(1, min(width, int(round(width * ratio)))),
                max(1, min(height, int(round(height * ratio)))),
            )
            updated = current.resize(down_size, _resampling(str(operation["down_kernel"]))).resize(
                (width, height),
                _resampling(str(operation["up_kernel"])),
            )
        elif name == "gaussian_blur":
            updated = current.filter(ImageFilter.GaussianBlur(radius=float(operation["radius"])))
        elif name == "gaussian_noise":
            array = np.asarray(current, dtype=np.float32)
            noise = np.random.default_rng(int(operation["seed"])).normal(
                0.0,
                float(operation["sigma_255"]),
                size=array.shape,
            )
            updated = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")
        elif name == "jpeg_compression":
            with io.BytesIO() as buffer:
                current.save(
                    buffer,
                    format="JPEG",
                    quality=int(operation["quality"]),
                    subsampling=int(operation["subsampling"]),
                )
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    updated = decoded.convert("RGB")
        else:
            current.close()
            raise ValueError(f"Unsupported synthetic pixel operation: {name!r}")
        current.close()
        current = updated
    if current.size != image.size:
        actual_size = current.size
        current.close()
        raise ValueError(
            f"Synthetic pixel augmentation changed image size: {actual_size} != {image.size}"
        )
    return current


__all__ = [
    "SYNTHETIC_REALISM_PROFILE",
    "apply_synthetic_realism_augmentation",
    "sample_synthetic_realism_augmentation",
]
