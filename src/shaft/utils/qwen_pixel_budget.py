from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from io import BytesIO
import math
import mimetypes
from pathlib import Path
from typing import Any

from PIL import Image


QWEN_IMAGE_FACTOR = 32
QWEN_MAX_ASPECT_RATIO = 200.0


@dataclass(frozen=True)
class QwenPixelBudgetResult:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    min_pixels: int | None = None
    max_pixels: int | None = None
    factor: int = QWEN_IMAGE_FACTOR
    resized: bool = False

    @property
    def source_pixels(self) -> int:
        return self.source_width * self.source_height

    @property
    def target_pixels(self) -> int:
        return self.target_width * self.target_height

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_pixels"] = self.source_pixels
        payload["target_pixels"] = self.target_pixels
        return payload


def smart_resize_qwen(
    *,
    width: int,
    height: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    factor: int = QWEN_IMAGE_FACTOR,
) -> tuple[int, int]:
    """Return Qwen-style resized dimensions for a pixel budget."""

    width = int(width)
    height = int(height)
    factor = int(factor)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    if factor <= 0:
        raise ValueError("factor must be positive.")
    if min_pixels is not None and int(min_pixels) <= 0:
        raise ValueError("min_pixels must be positive when set.")
    if max_pixels is not None and int(max_pixels) <= 0:
        raise ValueError("max_pixels must be positive when set.")
    if min_pixels is not None and max_pixels is not None and int(max_pixels) < int(min_pixels):
        raise ValueError("max_pixels must be >= min_pixels.")
    if max(width, height) / min(width, height) > QWEN_MAX_ASPECT_RATIO:
        raise ValueError(
            "absolute aspect ratio must be smaller than "
            f"{QWEN_MAX_ASPECT_RATIO:g}, got {max(width, height) / min(width, height)}"
        )

    target_height = max(factor, round(height / factor) * factor)
    target_width = max(factor, round(width / factor) * factor)

    area = target_height * target_width
    if max_pixels is not None and area > int(max_pixels):
        beta = math.sqrt((height * width) / int(max_pixels))
        target_height = max(factor, math.floor(height / beta / factor) * factor)
        target_width = max(factor, math.floor(width / beta / factor) * factor)
    elif min_pixels is not None and area < int(min_pixels):
        beta = math.sqrt(int(min_pixels) / (height * width))
        target_height = math.ceil(height * beta / factor) * factor
        target_width = math.ceil(width * beta / factor) * factor

    return int(target_width), int(target_height)


def apply_qwen_pixel_budget(
    image: Image.Image,
    *,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    factor: int = QWEN_IMAGE_FACTOR,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> tuple[Image.Image, QwenPixelBudgetResult]:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image.")
    source_width, source_height = image.size
    if min_pixels is None and max_pixels is None:
        return image, QwenPixelBudgetResult(
            source_width=source_width,
            source_height=source_height,
            target_width=source_width,
            target_height=source_height,
            min_pixels=None,
            max_pixels=None,
            factor=factor,
            resized=False,
        )
    target_width, target_height = smart_resize_qwen(
        width=source_width,
        height=source_height,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        factor=factor,
    )
    resized = (target_width, target_height) != (source_width, source_height)
    result = QwenPixelBudgetResult(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        factor=factor,
        resized=resized,
    )
    if not resized:
        return image, result
    return image.resize((target_width, target_height), resample), result


def image_to_data_url_with_qwen_pixel_budget(
    image_path: Path | str,
    *,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    factor: int = QWEN_IMAGE_FACTOR,
) -> tuple[str, QwenPixelBudgetResult]:
    path = Path(image_path)
    with Image.open(path) as image:
        resized, budget = apply_qwen_pixel_budget(
            image,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            factor=factor,
        )
        if not budget.resized:
            mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{encoded}", budget

        image_format = _encode_format_for_path(path, resized)
        buffer = BytesIO()
        save_kwargs: dict[str, Any] = {}
        if image_format == "JPEG":
            if resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
            save_kwargs["quality"] = 95
            save_kwargs["subsampling"] = 0
        resized.save(buffer, format=image_format, **save_kwargs)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:{_mime_for_format(image_format)};base64,{encoded}", budget


def _encode_format_for_path(path: Path, image: Image.Image) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG"
    if suffix == ".webp":
        return "WEBP"
    if image.mode in {"RGBA", "LA", "P"}:
        return "PNG"
    return "PNG"


def _mime_for_format(image_format: str) -> str:
    if image_format == "JPEG":
        return "image/jpeg"
    if image_format == "WEBP":
        return "image/webp"
    return "image/png"
