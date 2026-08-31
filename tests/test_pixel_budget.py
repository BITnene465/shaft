from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

from shaft.utils.qwen_pixel_budget import (
    apply_qwen_pixel_budget,
    image_to_data_url_with_qwen_pixel_budget,
    smart_resize_qwen,
)
from shaft.offline_kd.media_plan import (
    MEDIA_PLAN_VERSION,
    OfflineKDMediaPlan,
    deterministic_detection_media_plan,
)


def test_smart_resize_qwen_applies_patch_factor_and_max_pixels() -> None:
    width, height = smart_resize_qwen(width=4096, height=2748, max_pixels=1_000_000)

    assert width % 32 == 0
    assert height % 32 == 0
    assert width * height <= 1_000_000
    assert (width, height) == (1216, 800)


def test_apply_qwen_pixel_budget_does_not_resize_without_budget() -> None:
    image = Image.new("RGB", (101, 57), color=(255, 255, 255))

    resized, metadata = apply_qwen_pixel_budget(image, min_pixels=None, max_pixels=None)

    assert resized is image
    assert metadata.target_width == 101
    assert metadata.target_height == 57
    assert metadata.resized is False


def test_detection_media_plan_is_deterministic_and_round_trips() -> None:
    first = deterministic_detection_media_plan(
        sample_id="paper:sample-1", width=3840, height=2160, seed=465
    )
    second = deterministic_detection_media_plan(
        sample_id="paper:sample-1", width=3840, height=2160, seed=465
    )

    assert first == second
    assert 500_000 <= first.min_pixels <= 4_000_000
    assert first.min_pixels == first.max_pixels
    assert first.target_width % 32 == first.target_height % 32 == 0
    assert OfflineKDMediaPlan.from_mapping(first.to_dict()) == first
    assert first.to_dict()["version"] == MEDIA_PLAN_VERSION


def test_image_to_data_url_with_qwen_budget_resizes_payload(tmp_path: Path) -> None:
    image_path = tmp_path / "large.jpg"
    Image.new("RGB", (4096, 2748), color=(255, 255, 255)).save(image_path)

    data_url, metadata = image_to_data_url_with_qwen_pixel_budget(
        image_path,
        max_pixels=1_000_000,
    )

    assert metadata.resized is True
    assert metadata.target_width * metadata.target_height <= 1_000_000
    assert data_url.startswith("data:image/jpeg;base64,")
    encoded = data_url.split(",", 1)[1]
    decoded_path = tmp_path / "decoded.jpg"
    decoded_path.write_bytes(base64.b64decode(encoded))
    with Image.open(decoded_path) as decoded:
        assert decoded.size == (metadata.target_width, metadata.target_height)
