from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from shaft.codec.base import ShaftCodecResult
from shaft.metrics import prediction_visualization
from shaft.metrics.visualization import (
    ShaftVisualBox,
    ShaftVisualLineStrip,
    ShaftVisualPoint,
    render_labeled_visualization,
    resolve_annotation_font_size,
    resolve_box_line_width,
    resolve_label_color,
    resolve_point_radius,
)


def test_visualization_style_scales_with_image_size() -> None:
    small_width = resolve_box_line_width(512, 512)
    large_width = resolve_box_line_width(2048, 1536)
    assert small_width >= 2
    assert large_width > small_width
    assert resolve_box_line_width(2048, 1536, shape_count=180) < large_width

    small_radius = resolve_point_radius(512, 512)
    large_radius = resolve_point_radius(2048, 1536)
    assert small_radius >= 3
    assert large_radius > small_radius
    assert resolve_point_radius(2048, 1536, shape_count=180) < large_radius

    small_font = resolve_annotation_font_size(512, 512)
    large_font = resolve_annotation_font_size(2048, 1536)
    assert small_font >= 12
    assert large_font > small_font
    assert resolve_annotation_font_size(2048, 1536, shape_count=180) < large_font


def test_visualization_uses_layout_preview_label_colors() -> None:
    image_color = resolve_label_color("image", 1)
    icon_color = resolve_label_color("icon", 1)
    fallback_first = resolve_label_color("other", 1)
    fallback_second = resolve_label_color("other", 2)

    assert image_color == (220, 60, 115)
    assert icon_color == (0, 180, 120)
    assert image_color != icon_color
    assert fallback_first != fallback_second


def test_visualization_draws_visible_boxes_and_labels() -> None:
    image = Image.new("RGB", (320, 220), "white")
    rendered = render_labeled_visualization(
        image,
        boxes=[
            ShaftVisualBox(label="icon", bbox=(30, 30, 140, 150), index=1),
            ShaftVisualBox(label="image", bbox=(170, 40, 300, 190), index=2),
        ],
        footer_lines=["id=sample idx=000001"],
    )

    colors = {color for _, color in rendered.getcolors(maxcolors=1_000_000) or []}
    assert rendered.width == 320
    assert rendered.height > 220
    assert (0, 180, 120) in colors
    assert (220, 60, 115) in colors
    assert (0, 0, 0) in colors


def test_visualization_keeps_dense_boxes_on_base_image_by_default() -> None:
    image = Image.new("RGB", (360, 260), "white")
    boxes = [
        ShaftVisualBox(label="icon", bbox=(40, 40 + index * 18, 62, 55 + index * 18), index=index + 1)
        for index in range(7)
    ]

    rendered = render_labeled_visualization(image, boxes=boxes)
    colors = {color for _, color in rendered.getcolors(maxcolors=1_000_000) or []}
    color_counts = {color: count for count, color in rendered.getcolors(maxcolors=1_000_000) or []}

    assert rendered.size == image.size
    assert (245, 158, 11) not in colors
    assert (0, 180, 120) in colors
    assert color_counts[(0, 180, 120)] > 2000
    assert rendered.getpixel((40, 40)) in {(0, 0, 0), (0, 180, 120)}


def test_visualization_draws_directional_linestrip_arrowheads() -> None:
    image = Image.new("RGB", (260, 180), "white")

    rendered = render_labeled_visualization(
        image,
        line_strips=[
            ShaftVisualLineStrip(
                points=(
                    ShaftVisualPoint(x=50, y=90),
                    ShaftVisualPoint(x=210, y=90),
                )
            )
        ],
    )

    # The arrowhead sits above/below the horizontal segment; this pixel would remain white
    # if the line strip had no directional marker.
    assert rendered.getpixel((153, 87)) in {(0, 0, 0), (20, 184, 166)}


def test_visualization_keeps_multiple_linestrips_independent() -> None:
    image = Image.new("RGB", (280, 180), "white")

    rendered = render_labeled_visualization(
        image,
        line_strips=[
            ShaftVisualLineStrip(
                points=(ShaftVisualPoint(x=30, y=50), ShaftVisualPoint(x=90, y=50))
            ),
            ShaftVisualLineStrip(
                points=(ShaftVisualPoint(x=190, y=130), ShaftVisualPoint(x=250, y=130))
            ),
        ],
    )

    assert rendered.getpixel((140, 90)) == (255, 255, 255)


def test_visualization_ignores_duplicate_linestrip_points() -> None:
    image = Image.new("RGB", (180, 120), "white")

    rendered = render_labeled_visualization(
        image,
        line_strips=[
            ShaftVisualLineStrip(
                points=(
                    ShaftVisualPoint(x=40, y=50),
                    ShaftVisualPoint(x=40, y=50),
                    ShaftVisualPoint(x=130, y=50),
                )
            )
        ],
    )

    assert rendered.size == image.size


def test_prediction_visualization_saves_shared_style(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 160), "white").save(image_path)
    prediction = ShaftCodecResult(
        raw_text="",
        parsed=[{"label": "icon", "bbox_2d": [100, 100, 800, 800]}],
        valid=True,
        partial=False,
        error_type=None,
        error=None,
    )

    output = prediction_visualization.render_prediction_visualization(
        image_path=str(image_path),
        sample_id="sample",
        sample_index=1,
        prediction=prediction,
        out_dir=tmp_path,
    )

    assert output is not None
    output_path = Path(output)
    assert output_path.exists()
    assert output_path.parent.name == "predictions"
    with Image.open(output_path) as rendered:
        assert rendered.width == 200
        assert rendered.height > 160


def test_prediction_visualization_keypoint_uses_line_strip(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 160), "white").save(image_path)
    prediction = ShaftCodecResult(
        raw_text="",
        parsed={"keypoints_2d": [[100, 100], [500, 500], [900, 900]]},
        valid=True,
        partial=False,
        error_type=None,
        error=None,
    )
    captured: dict[str, object] = {}

    def fake_save_labeled_visualization(**kwargs) -> str:
        captured.update(kwargs)
        return str(tmp_path / "predictions" / "sample.jpg")

    monkeypatch.setattr(
        prediction_visualization,
        "save_labeled_visualization",
        fake_save_labeled_visualization,
    )

    output = prediction_visualization.render_prediction_visualization(
        image_path=str(image_path),
        sample_id="sample",
        sample_index=1,
        prediction=prediction,
        out_dir=tmp_path,
    )

    assert output is not None
    assert captured["points"] == []
    line_strips = captured["line_strips"]
    assert isinstance(line_strips, list)
    assert len(line_strips) == 1
    assert len(line_strips[0].points) == 3


def test_prediction_visualization_points_2d_uses_line_strip(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 160), "white").save(image_path)
    prediction = ShaftCodecResult(
        raw_text="",
        parsed={"label": "line", "points_2d": [[100, 100], [900, 100]]},
        valid=True,
        partial=False,
        error_type=None,
        error=None,
    )
    captured: dict[str, object] = {}

    def fake_save_labeled_visualization(**kwargs) -> str:
        captured.update(kwargs)
        return str(tmp_path / "predictions" / "sample.jpg")

    monkeypatch.setattr(
        prediction_visualization,
        "save_labeled_visualization",
        fake_save_labeled_visualization,
    )

    output = prediction_visualization.render_prediction_visualization(
        image_path=str(image_path),
        sample_id="sample",
        sample_index=1,
        prediction=prediction,
        out_dir=tmp_path,
    )

    assert output is not None
    line_strips = captured["line_strips"]
    assert isinstance(line_strips, list)
    assert len(line_strips) == 1
    assert len(line_strips[0].points) == 2
    assert line_strips[0].points[0].x == pytest.approx(19.9199, abs=1e-4)
    assert line_strips[0].points[1].x == pytest.approx(179.2793, abs=1e-4)


def test_prediction_visualization_line_parameters_points_use_line_strips(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 160), "white").save(image_path)
    prediction = ShaftCodecResult(
        raw_text="",
        parsed={
            "type": "line",
            "parameters": {
                "is_single": False,
                "points": [
                    [[100, 100], [500, 100]],
                    [[500, 100], [900, 300]],
                ],
            },
        },
        valid=True,
        partial=False,
        error_type=None,
        error=None,
    )
    captured: dict[str, object] = {}

    def fake_save_labeled_visualization(**kwargs) -> str:
        captured.update(kwargs)
        return str(tmp_path / "predictions" / "sample.jpg")

    monkeypatch.setattr(
        prediction_visualization,
        "save_labeled_visualization",
        fake_save_labeled_visualization,
    )

    output = prediction_visualization.render_prediction_visualization(
        image_path=str(image_path),
        sample_id="sample",
        sample_index=1,
        prediction=prediction,
        out_dir=tmp_path,
    )

    assert output is not None
    line_strips = captured["line_strips"]
    assert isinstance(line_strips, list)
    assert len(line_strips) == 2
    assert len(line_strips[0].points) == 2
    assert len(line_strips[1].points) == 2
