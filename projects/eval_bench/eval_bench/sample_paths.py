from __future__ import annotations

from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def sample_image_path(
    json_relative: Path,
    payload: dict[str, Any],
    *,
    root: Path | None = None,
) -> Path:
    explicit = _explicit_image_path(payload)
    if explicit is not None:
        return explicit

    image_stem = _image_stem_from_json(json_relative)
    if image_stem is None:
        return json_relative.with_suffix(".png")

    if root is not None:
        for suffix in IMAGE_SUFFIXES:
            candidate = image_stem.with_suffix(suffix)
            if (root / candidate).exists():
                return candidate
    return image_stem.with_suffix(".png")


def sample_image_string(
    json_path: Path,
    payload: dict[str, Any],
    *,
    root: Path,
) -> str:
    try:
        json_relative = json_path.relative_to(root)
    except ValueError:
        json_relative = json_path
    return str(sample_image_path(json_relative, payload, root=root))


def prediction_json_path(predictions_dir: Path, image: str | Path) -> Path:
    return predictions_dir / prediction_json_relative_path(image)


def prediction_json_relative_path(image: str | Path) -> Path:
    image_path = Path(image)
    parts = image_path.parts
    if len(parts) >= 3 and parts[1] == "images":
        return Path(parts[0]) / "json" / image_path.with_suffix(".json").name
    return image_path.with_suffix(".json")


def _explicit_image_path(payload: dict[str, Any]) -> Path | None:
    for key in ("image_path", "image", "imagePath"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value.strip())
    return None


def _image_stem_from_json(json_relative: Path) -> Path | None:
    parts = json_relative.parts
    if len(parts) >= 3 and parts[1] == "json":
        return Path(parts[0]) / "images" / json_relative.with_suffix("").name
    return None
