from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TMP_SCRIPTS = ROOT / "scripts" / "tmp"
if str(TMP_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TMP_SCRIPTS))

import eval_common  # noqa: E402


def test_eval_common_visual_style_scales_with_image_size() -> None:
    small_width = eval_common._resolve_box_line_width(512, 512)
    large_width = eval_common._resolve_box_line_width(2048, 1536)
    assert small_width >= 3
    assert large_width > small_width

    small_radius = eval_common._resolve_point_radius(512, 512)
    large_radius = eval_common._resolve_point_radius(2048, 1536)
    assert small_radius >= 4
    assert large_radius > small_radius

    small_font = eval_common._resolve_annotation_font_size(512, 512)
    large_font = eval_common._resolve_annotation_font_size(2048, 1536)
    assert small_font >= 12
    assert large_font > small_font


def test_eval_common_box_palette_uses_multiple_colors() -> None:
    image_first = eval_common._resolve_box_color("image", 1)
    image_second = eval_common._resolve_box_color("image", 2)
    icon_first = eval_common._resolve_box_color("icon", 1)
    assert image_first != image_second
    assert image_first != icon_first
