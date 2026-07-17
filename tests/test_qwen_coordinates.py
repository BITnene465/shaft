from __future__ import annotations

import pytest

from shaft.codec.coordinates import (
    dequantize_qwen_bbox,
    dequantize_qwen_coordinate,
    dequantize_qwen_point,
    maybe_qwen_coordinate_payload,
    quantize_qwen_bbox,
    quantize_qwen_coordinate,
    quantize_qwen_point,
)


def test_qwen_coordinate_round_trips_without_truncation_bias() -> None:
    size = 2000
    for pixel in [0, 1, 10, 500, 1000, 1500, 1999]:
        quantized = quantize_qwen_coordinate(pixel, size=size)
        decoded = dequantize_qwen_coordinate(quantized, size=size)
        assert abs(decoded - pixel) <= 1.1


def test_qwen_coordinate_uses_nearest_integer_not_floor() -> None:
    # 250.25 on a 1001-pixel axis is just over the 250th bin midpoint. A floor-based
    # quantizer would emit 249 and create a systematic negative bias.
    assert quantize_qwen_coordinate(250.25, size=1001) == 250


def test_qwen_bbox_and_point_helpers_share_the_same_scale() -> None:
    bbox = [10.0, 20.0, 900.0, 700.0]
    quantized = quantize_qwen_bbox(bbox, width=1000, height=800)
    decoded = dequantize_qwen_bbox(quantized, width=1000, height=800)

    assert quantized == [10, 25, 900, 875]
    assert decoded == pytest.approx(bbox, abs=1.1)
    assert quantize_qwen_point([500.0, 400.0], width=1000, height=800) == [500, 500]
    assert dequantize_qwen_point([500, 500], width=1000, height=800) == pytest.approx(
        [500.0, 399.9],
        abs=1.1,
    )


def test_qwen_bbox_can_enforce_minimum_extent_at_both_edges() -> None:
    assert quantize_qwen_bbox(
        [1, 1, 2, 2],
        width=10_000,
        height=10_000,
        minimum_extent_bins=1,
    ) == [0, 0, 1, 1]
    assert quantize_qwen_bbox(
        [9998, 9998, 9999, 9999],
        width=10_000,
        height=10_000,
        minimum_extent_bins=1,
    ) == [998, 998, 999, 999]


def test_qwen_decode_accepts_legacy_1000_as_edge_alias() -> None:
    assert maybe_qwen_coordinate_payload([0, 1000])
    assert dequantize_qwen_coordinate(1000, size=1200) == pytest.approx(1199.0)
