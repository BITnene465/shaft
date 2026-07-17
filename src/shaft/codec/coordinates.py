from __future__ import annotations

import math
from typing import Sequence

QWEN_COORD_NUM_BINS = 1000


def qwen_coordinate_max(num_bins: int = QWEN_COORD_NUM_BINS) -> int:
    if int(num_bins) < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins!r}.")
    return int(num_bins) - 1


def quantize_qwen_coordinate(
    value: float,
    *,
    size: int,
    num_bins: int = QWEN_COORD_NUM_BINS,
) -> int:
    """Map a pixel coordinate to the project-standard Qwen-style 0..999 bin space.

    The standard is an inclusive integer coordinate range: with the default 1000 bins, valid
    output values are 0..999. Encoding uses nearest-integer rounding instead of truncation so
    model-facing targets do not inherit a systematic left/top bias.
    """

    max_index = qwen_coordinate_max(num_bins)
    if int(size) <= 1:
        return 0
    clipped = min(max(float(value), 0.0), float(int(size) - 1))
    normalized = clipped / float(int(size) - 1)
    return min(max_index, max(0, int(math.floor(normalized * float(max_index) + 0.5))))


def dequantize_qwen_coordinate(
    value: float,
    *,
    size: int,
    num_bins: int = QWEN_COORD_NUM_BINS,
) -> float:
    """Map a Qwen-style 0..999 coordinate back to pixel space.

    Values are clamped to 0..999. This deliberately accepts 1000 as a legacy/model-output
    right-edge alias and maps it to the image edge by clamping before scaling.
    """

    max_index = qwen_coordinate_max(num_bins)
    if int(size) <= 1:
        return 0.0
    clipped = min(max(float(value), 0.0), float(max_index))
    return clipped / float(max_index) * float(int(size) - 1)


def quantize_qwen_point(
    point: Sequence[float],
    *,
    width: int,
    height: int,
    num_bins: int = QWEN_COORD_NUM_BINS,
) -> list[int]:
    if len(point) != 2:
        raise ValueError(f"Expected a 2D point, got {point!r}.")
    return [
        quantize_qwen_coordinate(float(point[0]), size=width, num_bins=num_bins),
        quantize_qwen_coordinate(float(point[1]), size=height, num_bins=num_bins),
    ]


def dequantize_qwen_point(
    point: Sequence[float],
    *,
    width: int,
    height: int,
    num_bins: int = QWEN_COORD_NUM_BINS,
) -> list[float]:
    if len(point) != 2:
        raise ValueError(f"Expected a 2D point, got {point!r}.")
    return [
        dequantize_qwen_coordinate(float(point[0]), size=width, num_bins=num_bins),
        dequantize_qwen_coordinate(float(point[1]), size=height, num_bins=num_bins),
    ]


def quantize_qwen_bbox(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    num_bins: int = QWEN_COORD_NUM_BINS,
    minimum_extent_bins: int = 0,
) -> list[int]:
    if len(bbox) != 4:
        raise ValueError(f"Expected a 4D bbox, got {bbox!r}.")
    max_index = qwen_coordinate_max(num_bins)
    if not 0 <= int(minimum_extent_bins) <= max_index:
        raise ValueError(
            f"minimum_extent_bins must be between 0 and {max_index}, "
            f"got {minimum_extent_bins!r}."
        )
    x1, y1, x2, y2 = [float(value) for value in bbox]
    result = [
        quantize_qwen_coordinate(x1, size=width, num_bins=num_bins),
        quantize_qwen_coordinate(y1, size=height, num_bins=num_bins),
        quantize_qwen_coordinate(x2, size=width, num_bins=num_bins),
        quantize_qwen_coordinate(y2, size=height, num_bins=num_bins),
    ]
    for start, stop in ((0, 2), (1, 3)):
        missing = int(minimum_extent_bins) - (result[stop] - result[start])
        if missing <= 0:
            continue
        grow_right = min(missing, max_index - result[stop])
        result[stop] += grow_right
        missing -= grow_right
        result[start] = max(0, result[start] - missing)
    return result


def dequantize_qwen_bbox(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    num_bins: int = QWEN_COORD_NUM_BINS,
) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(f"Expected a 4D bbox, got {bbox!r}.")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    left = dequantize_qwen_coordinate(x1, size=width, num_bins=num_bins)
    top = dequantize_qwen_coordinate(y1, size=height, num_bins=num_bins)
    right = dequantize_qwen_coordinate(x2, size=width, num_bins=num_bins)
    bottom = dequantize_qwen_coordinate(y2, size=height, num_bins=num_bins)
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    return [left, top, right, bottom]


def maybe_qwen_coordinate_payload(values: Sequence[float], *, num_bins: int = QWEN_COORD_NUM_BINS) -> bool:
    if not values:
        return False
    max_index = qwen_coordinate_max(num_bins)
    # Accept 1000 as a legacy inclusive edge value while keeping the canonical range 0..999.
    return max(abs(float(value)) for value in values) <= float(max_index + 1)
