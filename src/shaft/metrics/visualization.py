from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps

RGBColor = tuple[int, int, int]
BBox = tuple[float, float, float, float]
PixelBBox = tuple[int, int, int, int]

_LABEL_COLORS: dict[str, RGBColor] = {
    "icon": (0, 180, 120),
    "image": (220, 60, 115),
    "keypoint": (20, 184, 166),
}
_FALLBACK_COLORS: tuple[RGBColor, ...] = (
    (217, 119, 6),
    (37, 99, 235),
    (124, 58, 237),
    (8, 145, 178),
    (101, 163, 13),
    (190, 18, 60),
)


@dataclass(frozen=True)
class ShaftVisualBox:
    label: str
    bbox: BBox
    index: int | None = None


@dataclass(frozen=True)
class ShaftVisualPoint:
    x: float
    y: float
    label: str | None = None
    index: int | None = None


@dataclass(frozen=True)
class ShaftVisualLineStrip:
    points: tuple[ShaftVisualPoint, ...]
    label: str | None = None
    index: int | None = None


@dataclass(frozen=True)
class ShaftVisualizationStyle:
    base_max_side: int = 2400
    medium_density_max_side: int = 3200
    high_density_max_side: int = 3800
    medium_density_shapes: int = 40
    high_density_shapes: int = 80
    jpeg_quality: int = 92
    enable_dense_zoom: bool = False
    dense_region_min_boxes: int = 6
    dense_region_max_panels: int = 6
    dense_region_padding_ratio: float = 0.035
    dense_zoom_min_scale: float = 1.6
    dense_zoom_max_scale: float = 3.0
    dense_zoom_max_height: int = 900
    dense_zoom_panel_max_width_ratio: float = 0.86


DEFAULT_VISUALIZATION_STYLE = ShaftVisualizationStyle()


@dataclass(frozen=True)
class _DenseRegion:
    bbox: PixelBBox
    box_indices: tuple[int, ...]


def resolve_box_line_width(
    image_width: int,
    image_height: int,
    *,
    shape_count: int = 0,
) -> int:
    return _resolve_box_line_width(image_width, image_height, shape_count=shape_count)


def _resolve_box_line_width(image_width: int, image_height: int, *, shape_count: int) -> int:
    short_edge = max(1, min(int(image_width), int(image_height)))
    density_scale = _density_scale(shape_count)
    width = int(round(short_edge / 280.0 * density_scale))
    return max(2, min(8, width))


def resolve_point_radius(
    image_width: int,
    image_height: int,
    *,
    shape_count: int = 0,
) -> int:
    return _resolve_point_radius(image_width, image_height, shape_count=shape_count)


def _resolve_point_radius(image_width: int, image_height: int, *, shape_count: int) -> int:
    short_edge = max(1, min(int(image_width), int(image_height)))
    density_scale = _density_scale(shape_count)
    radius = int(round(short_edge / 215.0 * density_scale))
    return max(3, min(9, radius))


def resolve_annotation_font_size(
    image_width: int,
    image_height: int,
    *,
    shape_count: int = 0,
) -> int:
    return _resolve_annotation_font_size(image_width, image_height, shape_count=shape_count)


def _resolve_annotation_font_size(image_width: int, image_height: int, *, shape_count: int) -> int:
    short_edge = max(1, min(int(image_width), int(image_height)))
    density_scale = max(0.72, _density_scale(shape_count))
    return max(12, min(34, int(round(short_edge / 52.0 * density_scale))))


def _density_scale(shape_count: int) -> float:
    if shape_count >= 160:
        return 0.54
    if shape_count >= 100:
        return 0.62
    if shape_count >= 60:
        return 0.72
    if shape_count >= 32:
        return 0.84
    return 1.0


def resolve_label_color(label: str, index: int = 1) -> RGBColor:
    normalized = str(label or "").strip().lower()
    if normalized in _LABEL_COLORS:
        return _LABEL_COLORS[normalized]
    palette_index = max(0, int(index) - 1) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[palette_index]


def render_labeled_visualization(
    image: Image.Image,
    *,
    boxes: Iterable[ShaftVisualBox] = (),
    points: Iterable[ShaftVisualPoint] = (),
    line_strips: Iterable[ShaftVisualLineStrip] = (),
    footer_lines: Iterable[str] = (),
    style: ShaftVisualizationStyle = DEFAULT_VISUALIZATION_STYLE,
) -> Image.Image:
    source = ImageOps.exif_transpose(image).convert("RGB")
    box_list = list(boxes)
    point_list = list(points)
    line_strip_list = list(line_strips)
    line_strip_point_count = sum(len(strip.points) for strip in line_strip_list)
    shape_count = len(box_list) + len(point_list) + line_strip_point_count
    render_scale = _resolve_render_scale(
        image_width=source.width,
        image_height=source.height,
        shape_count=shape_count,
        style=style,
    )
    if render_scale < 1.0:
        width = max(1, int(round(source.width * render_scale)))
        height = max(1, int(round(source.height * render_scale)))
        canvas = source.resize((width, height), Image.Resampling.LANCZOS)
    else:
        canvas = source.copy()
    base_canvas = canvas.copy() if style.enable_dense_zoom else None

    draw = ImageDraw.Draw(canvas)
    font = load_annotation_font(
        _resolve_annotation_font_size(canvas.width, canvas.height, shape_count=shape_count)
    )
    line_width = _resolve_box_line_width(canvas.width, canvas.height, shape_count=shape_count)
    label_rects: list[PixelBBox] = []

    scaled_boxes = [
        (idx, box, _scale_bbox(box.bbox, render_scale, canvas.width, canvas.height))
        for idx, box in enumerate(box_list)
    ]
    scaled_boxes = [item for item in scaled_boxes if item[2] is not None]
    dense_regions = (
        _find_dense_regions(
            scaled_boxes=scaled_boxes,
            image_size=(canvas.width, canvas.height),
            style=style,
        )
        if style.enable_dense_zoom
        else []
    )
    dense_box_indices = {
        box_index for region in dense_regions for box_index in region.box_indices
    }
    zoom_boundary_margin = max(4, line_width * 3)
    scaled_boxes.sort(key=lambda item: (-_pixel_bbox_area(item[2]), item[0]))
    for box_index, box, pixel_bbox in scaled_boxes:
        if pixel_bbox is None:
            continue
        if _bbox_inside_zoom_interior(
            pixel_bbox,
            regions=dense_regions,
            boundary_margin=zoom_boundary_margin,
        ):
            continue
        color = resolve_label_color(box.label, box.index or 1)
        if box_index in dense_box_indices:
            _draw_box_outline(draw=draw, bbox=pixel_bbox, color=color, line_width=line_width)
        else:
            _draw_labeled_box(
                draw=draw,
                image_size=(canvas.width, canvas.height),
                bbox=pixel_bbox,
                label=_format_box_label(box),
                color=color,
                line_width=line_width,
                font=font,
                occupied_labels=label_rects,
            )

    scaled_points = [
        _scale_point(point, render_scale, canvas.width, canvas.height) for point in point_list
    ]
    scaled_line_strips = [
        _scale_line_strip(strip, render_scale, canvas.width, canvas.height)
        for strip in line_strip_list
    ]
    main_line_strips = [
        None
        if _line_strip_inside_zoom_interior(
            strip_item,
            regions=dense_regions,
            boundary_margin=zoom_boundary_margin,
        )
        else strip_item
        for strip_item in scaled_line_strips
    ]
    main_points = [
        None
        if _point_inside_zoom_interior(
            item,
            regions=dense_regions,
            boundary_margin=zoom_boundary_margin,
        )
        else item
        for item in scaled_points
    ]
    _draw_line_strips(
        draw=draw,
        image_size=(canvas.width, canvas.height),
        line_strips=main_line_strips,
        line_width=line_width,
        font=font,
        occupied_labels=label_rects,
        shape_count=shape_count,
    )
    _draw_points(
        draw=draw,
        image_size=(canvas.width, canvas.height),
        points=main_points,
        line_width=line_width,
        font=font,
        occupied_labels=label_rects,
        shape_count=shape_count,
    )

    if dense_regions and base_canvas is not None:
        region_color = (245, 158, 11)
        for region in dense_regions:
            _draw_dashed_rectangle(
                draw=draw,
                bbox=region.bbox,
                color=region_color,
                line_width=line_width,
            )
        canvas = _append_dense_zoom_panels(
            canvas=canvas,
            base_canvas=base_canvas,
            regions=dense_regions,
            scaled_boxes=scaled_boxes,
            scaled_points=scaled_points,
            scaled_line_strips=scaled_line_strips,
            line_width=line_width,
            style=style,
        )

    return append_footer(canvas, footer_lines)


def save_labeled_visualization(
    *,
    image_path: str | Path,
    output_path: str | Path,
    boxes: Iterable[ShaftVisualBox] = (),
    points: Iterable[ShaftVisualPoint] = (),
    line_strips: Iterable[ShaftVisualLineStrip] = (),
    footer_lines: Iterable[str] = (),
    style: ShaftVisualizationStyle = DEFAULT_VISUALIZATION_STYLE,
) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        rendered = render_labeled_visualization(
            image,
            boxes=boxes,
            points=points,
            line_strips=line_strips,
            footer_lines=footer_lines,
            style=style,
        )
    rendered.save(output, format="JPEG", quality=int(style.jpeg_quality), optimize=True)
    return str(output)


def append_footer(image: Image.Image, lines: Iterable[str]) -> Image.Image:
    footer_lines = [line for line in lines if str(line).strip()]
    if not footer_lines:
        return image

    footer_font = load_annotation_font(max(12, min(18, int(round(image.width / 150.0)))))
    measure_image = Image.new("RGB", (1, 1), "white")
    measure_draw = ImageDraw.Draw(measure_image)
    wrapped_lines = _wrap_footer_lines(
        draw=measure_draw,
        lines=[str(line) for line in footer_lines],
        font=footer_font,
        max_width=max(1, image.width - 16),
    )
    spacing = 4
    footer_text = "\n".join(wrapped_lines)
    text_bbox = measure_draw.multiline_textbbox((0, 0), footer_text, font=footer_font, spacing=spacing)
    footer_height = max(32, (text_bbox[3] - text_bbox[1]) + 20)
    canvas = Image.new("RGB", (image.width, image.height + footer_height), "white")
    canvas.paste(image, (0, 0))
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.multiline_text(
        (8, image.height + 8),
        footer_text,
        fill=(0, 0, 0),
        font=footer_font,
        spacing=spacing,
    )
    return canvas


def load_annotation_font(font_size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=int(font_size))
    except Exception:
        return ImageFont.load_default()


def _resolve_render_scale(
    *,
    image_width: int,
    image_height: int,
    shape_count: int,
    style: ShaftVisualizationStyle,
) -> float:
    max_side = int(style.base_max_side)
    if shape_count >= int(style.high_density_shapes):
        max_side = int(style.high_density_max_side)
    elif shape_count >= int(style.medium_density_shapes):
        max_side = int(style.medium_density_max_side)

    longest = max(1, int(image_width), int(image_height))
    if longest <= max_side:
        return 1.0
    return max_side / float(longest)


def _scale_bbox(
    bbox: BBox,
    scale: float,
    image_width: int,
    image_height: int,
) -> PixelBBox | None:
    try:
        x1, y1, x2, y2 = [float(value) * float(scale) for value in bbox]
    except (TypeError, ValueError):
        return None
    if not (x2 > x1 and y2 > y1):
        return None

    max_x = max(0, image_width - 1)
    max_y = max(0, image_height - 1)
    ix1 = int(round(max(0.0, min(float(max_x), x1))))
    iy1 = int(round(max(0.0, min(float(max_y), y1))))
    ix2 = int(round(max(0.0, min(float(max_x), x2))))
    iy2 = int(round(max(0.0, min(float(max_y), y2))))
    if ix2 <= ix1:
        ix2 = min(max_x, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(max_y, iy1 + 1)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def _scale_point(
    point: ShaftVisualPoint,
    scale: float,
    image_width: int,
    image_height: int,
) -> tuple[ShaftVisualPoint, tuple[int, int]] | None:
    try:
        x = float(point.x) * float(scale)
        y = float(point.y) * float(scale)
    except (TypeError, ValueError):
        return None
    x = max(0.0, min(float(max(0, image_width - 1)), x))
    y = max(0.0, min(float(max(0, image_height - 1)), y))
    return point, (int(round(x)), int(round(y)))


def _scale_line_strip(
    strip: ShaftVisualLineStrip,
    scale: float,
    image_width: int,
    image_height: int,
) -> tuple[ShaftVisualLineStrip, list[tuple[ShaftVisualPoint, tuple[int, int]]]] | None:
    points = [
        scaled_point
        for point in strip.points
        if (scaled_point := _scale_point(point, scale, image_width, image_height)) is not None
    ]
    if not points:
        return None
    return strip, points


def _draw_labeled_box(
    *,
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    bbox: PixelBBox,
    label: str,
    color: RGBColor,
    line_width: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    occupied_labels: list[PixelBBox],
) -> None:
    _draw_box_outline(draw=draw, bbox=bbox, color=color, line_width=line_width)
    _draw_label(
        draw=draw,
        image_size=image_size,
        anchor_bbox=bbox,
        text=label,
        color=color,
        line_width=line_width,
        font=font,
        occupied_labels=occupied_labels,
    )


def _draw_box_outline(
    *,
    draw: ImageDraw.ImageDraw,
    bbox: PixelBBox,
    color: RGBColor,
    line_width: int,
) -> None:
    outline_width = line_width + max(1, line_width // 2)
    draw.rectangle(bbox, outline=(0, 0, 0), width=outline_width)
    draw.rectangle(bbox, outline=color, width=line_width)


def _draw_line_strips(
    *,
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    line_strips: list[
        tuple[ShaftVisualLineStrip, list[tuple[ShaftVisualPoint, tuple[int, int]]]] | None
    ],
    line_width: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    occupied_labels: list[PixelBBox],
    shape_count: int,
) -> None:
    for strip_item in line_strips:
        if strip_item is None:
            continue
        _, strip_points = strip_item
        _draw_points(
            draw=draw,
            image_size=image_size,
            points=strip_points,
            line_width=line_width,
            font=font,
            occupied_labels=occupied_labels,
            shape_count=shape_count,
        )


def _draw_points(
    *,
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    points: list[tuple[ShaftVisualPoint, tuple[int, int]] | None],
    line_width: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    occupied_labels: list[PixelBBox],
    shape_count: int,
) -> None:
    valid_points = [item for item in points if item is not None]
    if not valid_points:
        return

    xy_points = [xy for _, xy in valid_points]
    point_color = resolve_label_color("keypoint", 1)
    if len(xy_points) >= 2:
        outline_width = line_width + max(1, line_width // 2)
        draw.line(xy_points, fill=(0, 0, 0), width=outline_width, joint="curve")
        draw.line(xy_points, fill=point_color, width=line_width, joint="curve")
        _draw_line_strip_arrowheads(
            draw=draw,
            points=xy_points,
            color=point_color,
            line_width=line_width,
        )

    radius = _resolve_point_radius(*image_size, shape_count=shape_count)
    for point, (cx, cy) in valid_points:
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(0, 0, 0),
            outline=(0, 0, 0),
            width=max(1, line_width),
        )
        inset = max(1, min(radius - 1, line_width // 2))
        draw.ellipse(
            [
                cx - radius + inset,
                cy - radius + inset,
                cx + radius - inset,
                cy + radius - inset,
            ],
            fill=point_color,
            outline=point_color,
            width=max(1, line_width - 1),
        )
        label = _format_point_label(point)
        if label:
            anchor = (cx - radius, cy - radius, cx + radius, cy + radius)
            _draw_label(
                draw=draw,
                image_size=image_size,
                anchor_bbox=anchor,
                text=label,
                color=point_color,
                line_width=line_width,
                font=font,
                occupied_labels=occupied_labels,
            )


def _draw_label(
    *,
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    anchor_bbox: PixelBBox,
    text: str,
    color: RGBColor,
    line_width: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    occupied_labels: list[PixelBBox],
) -> None:
    image_width, image_height = image_size
    label_text = _fit_label_text(
        draw=draw,
        text=str(text or "box"),
        font=font,
        max_width=max(1, image_width - 8),
    )
    stroke_width = max(1, line_width // 4)
    text_bbox = draw.textbbox((0, 0), label_text, font=font, stroke_width=stroke_width)
    text_width = max(1, text_bbox[2] - text_bbox[0])
    text_height = max(1, text_bbox[3] - text_bbox[1])
    pad_x = max(5, line_width)
    pad_y = max(3, line_width // 2)
    label_width = min(image_width, text_width + (2 * pad_x))
    label_height = min(image_height, text_height + (2 * pad_y))
    label_rect = _place_label_rect(
        anchor_bbox=anchor_bbox,
        label_size=(label_width, label_height),
        image_size=image_size,
        margin=max(3, line_width),
        occupied_labels=occupied_labels,
    )

    draw.rectangle(label_rect, fill=color)
    draw.rectangle(label_rect, outline=(0, 0, 0), width=max(2, line_width // 2))
    draw.text(
        (label_rect[0] + pad_x - text_bbox[0], label_rect[1] + pad_y - text_bbox[1]),
        label_text,
        fill=(255, 255, 255),
        font=font,
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0),
    )
    occupied_labels.append(label_rect)


def _place_label_rect(
    *,
    anchor_bbox: PixelBBox,
    label_size: tuple[int, int],
    image_size: tuple[int, int],
    margin: int,
    occupied_labels: list[PixelBBox],
) -> PixelBBox:
    x1, y1, x2, y2 = anchor_bbox
    label_width, label_height = label_size
    raw_candidates = (
        (x1, y1 - label_height - margin),
        (x1, y1 + margin),
        (x1, y2 + margin),
        (x2 - label_width, y1 - label_height - margin),
        (x2 - label_width, y1 + margin),
        (x2 - label_width, y2 - label_height - margin),
        (x2 + margin, y1),
        (x1 - label_width - margin, y1),
    )

    candidates: list[PixelBBox] = []
    seen: set[PixelBBox] = set()
    for raw_x, raw_y in raw_candidates:
        rect = _clamp_rect(raw_x, raw_y, label_width, label_height, image_size)
        if rect not in seen:
            candidates.append(rect)
            seen.add(rect)

    best_rect = candidates[0]
    best_score: tuple[int, int, int] | None = None
    for rect in candidates:
        label_overlap = sum(_intersection_area(rect, occupied) for occupied in occupied_labels)
        anchor_overlap = _intersection_area(rect, anchor_bbox)
        distance = abs(rect[0] - x1) + abs(rect[1] - y1)
        score = (label_overlap, anchor_overlap, distance)
        if label_overlap == 0 and anchor_overlap == 0:
            return rect
        if best_score is None or score < best_score:
            best_rect = rect
            best_score = score
    return best_rect


def _find_dense_regions(
    *,
    scaled_boxes: list[tuple[int, ShaftVisualBox, PixelBBox | None]],
    image_size: tuple[int, int],
    style: ShaftVisualizationStyle,
) -> list[_DenseRegion]:
    valid = [(box_index, bbox) for box_index, _, bbox in scaled_boxes if bbox is not None]
    min_boxes = max(2, int(style.dense_region_min_boxes))
    if len(valid) < min_boxes:
        return []

    image_width, image_height = image_size
    short_edge = max(1, min(image_width, image_height))
    pad = max(10, int(round(short_edge * float(style.dense_region_padding_ratio))))
    min_zoom_scale = max(1.0, float(style.dense_zoom_min_scale))
    max_region_width = max(
        96,
        int(round(image_width * float(style.dense_zoom_panel_max_width_ratio) / min_zoom_scale)),
    )
    max_region_height = max(96, int(round(int(style.dense_zoom_max_height) / min_zoom_scale)))
    expanded = [
        (box_index, bbox, _expand_rect(bbox, pad, image_size)) for box_index, bbox in valid
    ]

    visited: set[int] = set()
    regions: list[_DenseRegion] = []
    for start in range(len(expanded)):
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        visited.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            current_expanded = expanded[current][2]
            for candidate in range(len(expanded)):
                if candidate in visited:
                    continue
                if _rects_touch(current_expanded, expanded[candidate][2]):
                    visited.add(candidate)
                    stack.append(candidate)
        if len(component) < min_boxes:
            continue

        component_items = [(expanded[index][0], expanded[index][1]) for index in component]
        regions.extend(
            _split_dense_region_items(
                component_items,
                image_size=image_size,
                padding=pad,
                min_boxes=min_boxes,
                max_region_size=(max_region_width, max_region_height),
            )
        )

    regions.sort(key=lambda item: (-len(item.box_indices), _pixel_bbox_area(item.bbox)))
    return regions[: max(0, int(style.dense_region_max_panels))]


def _split_dense_region_items(
    items: list[tuple[int, PixelBBox]],
    *,
    image_size: tuple[int, int],
    padding: int,
    min_boxes: int,
    max_region_size: tuple[int, int],
) -> list[_DenseRegion]:
    if len(items) < min_boxes:
        return []

    region = _build_dense_region(items, image_size=image_size, padding=padding)
    region_width = max(1, region.bbox[2] - region.bbox[0])
    region_height = max(1, region.bbox[3] - region.bbox[1])
    max_width, max_height = max_region_size
    image_area = max(1, image_size[0] * image_size[1])
    if (
        region_width <= max_width
        and region_height <= max_height
        and _pixel_bbox_area(region.bbox) / float(image_area) <= 0.65
    ):
        return [region]
    if len(items) < min_boxes * 2:
        return [region]

    split_axis = 0 if region_width / float(max_width) >= region_height / float(max_height) else 1
    sorted_items = sorted(items, key=lambda item: _rect_center(item[1])[split_axis])
    midpoint = len(sorted_items) // 2
    left = sorted_items[:midpoint]
    right = sorted_items[midpoint:]
    if len(left) < min_boxes or len(right) < min_boxes:
        return [region]

    split_regions = []
    split_regions.extend(
        _split_dense_region_items(
            left,
            image_size=image_size,
            padding=padding,
            min_boxes=min_boxes,
            max_region_size=max_region_size,
        )
    )
    split_regions.extend(
        _split_dense_region_items(
            right,
            image_size=image_size,
            padding=padding,
            min_boxes=min_boxes,
            max_region_size=max_region_size,
        )
    )
    return split_regions or [region]


def _build_dense_region(
    items: list[tuple[int, PixelBBox]],
    *,
    image_size: tuple[int, int],
    padding: int,
) -> _DenseRegion:
    bbox = _union_rects([item[1] for item in items])
    return _DenseRegion(
        bbox=_expand_rect(bbox, padding, image_size),
        box_indices=tuple(item[0] for item in items),
    )


def _append_dense_zoom_panels(
    *,
    canvas: Image.Image,
    base_canvas: Image.Image,
    regions: list[_DenseRegion],
    scaled_boxes: list[tuple[int, ShaftVisualBox, PixelBBox | None]],
    scaled_points: list[tuple[ShaftVisualPoint, tuple[int, int]] | None],
    scaled_line_strips: list[
        tuple[ShaftVisualLineStrip, list[tuple[ShaftVisualPoint, tuple[int, int]]]] | None
    ],
    line_width: int,
    style: ShaftVisualizationStyle,
) -> Image.Image:
    image_size = (canvas.width, canvas.height)
    short_edge = max(1, min(canvas.width, canvas.height))
    crop_margin = max(12, int(round(short_edge * 0.025)))
    gap = max(8, line_width * 2)
    panels: list[Image.Image] = []

    for region_index, region in enumerate(regions, start=1):
        panel = _render_dense_zoom_panel(
            base_canvas=base_canvas,
            region=region,
            region_index=region_index,
            image_size=image_size,
            crop_margin=crop_margin,
            scaled_boxes=scaled_boxes,
            scaled_points=scaled_points,
            scaled_line_strips=scaled_line_strips,
            line_width=line_width,
            gap=gap,
            style=style,
        )
        if panel is not None:
            panels.append(panel)

    if not panels:
        return canvas

    return _compose_dense_zoom_layout(canvas=canvas, panels=panels, gap=gap)


def _render_dense_zoom_panel(
    *,
    base_canvas: Image.Image,
    region: _DenseRegion,
    region_index: int,
    image_size: tuple[int, int],
    crop_margin: int,
    scaled_boxes: list[tuple[int, ShaftVisualBox, PixelBBox | None]],
    scaled_points: list[tuple[ShaftVisualPoint, tuple[int, int]] | None],
    scaled_line_strips: list[
        tuple[ShaftVisualLineStrip, list[tuple[ShaftVisualPoint, tuple[int, int]]]] | None
    ],
    line_width: int,
    gap: int,
    style: ShaftVisualizationStyle,
) -> Image.Image | None:
    crop_bbox = _expand_rect(region.bbox, crop_margin, image_size)
    crop = base_canvas.crop(crop_bbox)
    if crop.width <= 0 or crop.height <= 0:
        return None

    max_panel_width = max(
        360,
        int(round(base_canvas.width * float(style.dense_zoom_panel_max_width_ratio))),
    )
    available_width = max(1, max_panel_width - (2 * gap))
    max_scale_by_width = available_width / float(crop.width)
    max_scale_by_height = int(style.dense_zoom_max_height) / float(crop.height)
    scale = min(float(style.dense_zoom_max_scale), max_scale_by_width, max_scale_by_height)
    scale = max(float(style.dense_zoom_min_scale), scale)
    zoom_width = max(1, int(round(crop.width * scale)))
    zoom_height = max(1, int(round(crop.height * scale)))
    zoom = crop.resize((zoom_width, zoom_height), Image.Resampling.LANCZOS)

    header_height = max(24, line_width * 6)
    panel_width = zoom_width + (2 * gap)
    panel_height = zoom_height + header_height + (2 * gap)
    panel = Image.new("RGB", (panel_width, panel_height), (248, 250, 252))
    x_offset = gap
    y_offset = header_height + gap
    panel.paste(zoom, (x_offset, y_offset))
    panel_draw = ImageDraw.Draw(panel)
    panel_font = load_annotation_font(max(12, min(20, int(round(panel_width / 80.0)))))
    title = f"zoom {region_index}: dense region, {len(region.box_indices)} boxes"
    title = _fit_label_text(
        draw=panel_draw,
        text=title,
        font=panel_font,
        max_width=max(1, panel_width - (2 * gap)),
    )
    panel_draw.text(
        (gap, max(4, (header_height - _font_height(panel_font)) // 2)),
        title,
        fill=(15, 23, 42),
        font=panel_font,
    )
    _draw_dashed_rectangle(
        draw=panel_draw,
        bbox=(x_offset, y_offset, x_offset + zoom_width - 1, y_offset + zoom_height - 1),
        color=(245, 158, 11),
        line_width=max(2, line_width),
    )

    crop_x1, crop_y1, _, _ = crop_bbox
    local_boxes = [
        item for item in scaled_boxes if item[2] is not None and _rects_touch(item[2], crop_bbox)
    ]
    local_points: list[tuple[ShaftVisualPoint, tuple[int, int]] | None] = []
    for item in scaled_points:
        if item is None:
            continue
        point, (x, y) = item
        if crop_x1 <= x <= crop_bbox[2] and crop_y1 <= y <= crop_bbox[3]:
            local_points.append((point, _translate_scale_point((x, y), crop_bbox, scale, (x_offset, y_offset))))

    local_line_strips: list[
        tuple[ShaftVisualLineStrip, list[tuple[ShaftVisualPoint, tuple[int, int]]]] | None
    ] = []
    local_line_point_count = 0
    for strip_item in scaled_line_strips:
        if strip_item is None:
            continue
        strip, strip_points = strip_item
        local_strip_points: list[tuple[ShaftVisualPoint, tuple[int, int]]] = []
        for point, xy in strip_points:
            x, y = xy
            if crop_x1 <= x <= crop_bbox[2] and crop_y1 <= y <= crop_bbox[3]:
                local_strip_points.append(
                    (point, _translate_scale_point(xy, crop_bbox, scale, (x_offset, y_offset)))
                )
        if local_strip_points:
            local_line_point_count += len(local_strip_points)
            local_line_strips.append((strip, local_strip_points))

    local_shape_count = len(local_boxes) + len(local_points) + local_line_point_count
    local_font = load_annotation_font(
        _resolve_annotation_font_size(zoom_width, zoom_height, shape_count=local_shape_count)
    )
    local_line_width = _resolve_box_line_width(
        zoom_width,
        zoom_height,
        shape_count=local_shape_count,
    )
    occupied_labels: list[PixelBBox] = []
    for _, box, bbox in sorted(
        local_boxes,
        key=lambda item: (-_pixel_bbox_area(item[2]), item[0]),  # type: ignore[arg-type]
    ):
        assert bbox is not None
        local_bbox = _translate_scale_bbox(
            bbox=bbox,
            crop_origin=(crop_x1, crop_y1),
            scale=scale,
            offset=(x_offset, y_offset),
            image_size=panel.size,
        )
        _draw_labeled_box(
            draw=panel_draw,
            image_size=panel.size,
            bbox=local_bbox,
            label=_format_box_label(box),
            color=resolve_label_color(box.label, box.index or 1),
            line_width=local_line_width,
            font=local_font,
            occupied_labels=occupied_labels,
        )

    _draw_line_strips(
        draw=panel_draw,
        image_size=panel.size,
        line_strips=local_line_strips,
        line_width=local_line_width,
        font=local_font,
        occupied_labels=occupied_labels,
        shape_count=local_shape_count,
    )
    _draw_points(
        draw=panel_draw,
        image_size=panel.size,
        points=local_points,
        line_width=local_line_width,
        font=local_font,
        occupied_labels=occupied_labels,
        shape_count=local_shape_count,
    )
    return panel


def _compose_dense_zoom_layout(
    *,
    canvas: Image.Image,
    panels: list[Image.Image],
    gap: int,
) -> Image.Image:
    candidates: list[tuple[float, str, Image.Image, int, int]] = []
    for columns in range(1, len(panels) + 1):
        grid = _build_panel_grid(panels=panels, columns=columns, gap=gap)
        below_width = max(canvas.width, grid.width)
        below_height = canvas.height + gap + grid.height
        candidates.append(
            (
                _layout_score(below_width, below_height, base_size=canvas.size),
                "below",
                grid,
                below_width,
                below_height,
            )
        )
        right_width = canvas.width + gap + grid.width
        right_height = max(canvas.height, grid.height)
        candidates.append(
            (
                _layout_score(right_width, right_height, base_size=canvas.size),
                "right",
                grid,
                right_width,
                right_height,
            )
        )

    _, placement, grid, output_width, output_height = min(candidates, key=lambda item: item[0])
    output = Image.new("RGB", (output_width, output_height), "white")
    if placement == "right":
        canvas_y = max(0, (output_height - canvas.height) // 2)
        grid_y = max(0, (output_height - grid.height) // 2)
        output.paste(canvas, (0, canvas_y))
        output.paste(grid, (canvas.width + gap, grid_y))
    else:
        canvas_x = max(0, (output_width - canvas.width) // 2)
        grid_x = max(0, (output_width - grid.width) // 2)
        output.paste(canvas, (canvas_x, 0))
        output.paste(grid, (grid_x, canvas.height + gap))
    return output


def _build_panel_grid(*, panels: list[Image.Image], columns: int, gap: int) -> Image.Image:
    rows = [panels[index : index + columns] for index in range(0, len(panels), columns)]
    row_widths = [sum(panel.width for panel in row) + gap * max(0, len(row) - 1) for row in rows]
    row_heights = [max(panel.height for panel in row) for row in rows]
    grid_width = max(row_widths)
    grid_height = sum(row_heights) + gap * max(0, len(rows) - 1)
    grid = Image.new("RGB", (grid_width, grid_height), (248, 250, 252))
    y = 0
    for row, row_width, row_height in zip(rows, row_widths, row_heights, strict=True):
        x = max(0, (grid_width - row_width) // 2)
        for panel in row:
            panel_y = y + max(0, (row_height - panel.height) // 2)
            grid.paste(panel, (x, panel_y))
            x += panel.width + gap
        y += row_height + gap
    return grid


def _layout_score(width: int, height: int, *, base_size: tuple[int, int]) -> float:
    aspect = max(width / float(max(1, height)), height / float(max(1, width)))
    base_area = max(1, base_size[0] * base_size[1])
    area_penalty = (width * height / float(base_area)) * 0.025
    width_penalty = 0.2 if width > base_size[0] * 2.2 else 0.0
    return aspect + area_penalty + width_penalty


def _draw_dashed_rectangle(
    *,
    draw: ImageDraw.ImageDraw,
    bbox: PixelBBox,
    color: RGBColor,
    line_width: int,
) -> None:
    x1, y1, x2, y2 = bbox
    dash = max(8, int(line_width * 4))
    gap = max(5, int(line_width * 2))
    _draw_dashed_line(draw, (x1, y1), (x2, y1), color, line_width, dash, gap)
    _draw_dashed_line(draw, (x2, y1), (x2, y2), color, line_width, dash, gap)
    _draw_dashed_line(draw, (x2, y2), (x1, y2), color, line_width, dash, gap)
    _draw_dashed_line(draw, (x1, y2), (x1, y1), color, line_width, dash, gap)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: RGBColor,
    line_width: int,
    dash: int,
    gap: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 0:
        return
    ux = dx / length
    uy = dy / length
    position = 0.0
    while position < length:
        segment_end = min(length, position + dash)
        sx = int(round(x1 + ux * position))
        sy = int(round(y1 + uy * position))
        ex = int(round(x1 + ux * segment_end))
        ey = int(round(y1 + uy * segment_end))
        draw.line((sx, sy, ex, ey), fill=(0, 0, 0), width=line_width + 1)
        draw.line((sx, sy, ex, ey), fill=color, width=line_width)
        position += dash + gap


def _draw_line_strip_arrowheads(
    *,
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: RGBColor,
    line_width: int,
) -> None:
    if len(points) < 2:
        return
    base_size = max(6.0, min(14.0, float(line_width) * 2.4))
    for start, end in zip(points, points[1:], strict=False):
        x1, y1 = start
        x2, y2 = end
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length <= 1.0:
            continue
        size = min(base_size, length * 0.18)
        if length < size * 1.8:
            continue
        wing = size * 0.45
        ux = dx / length
        uy = dy / length
        tip_x = float(x1) + dx * 0.68
        tip_y = float(y1) + dy * 0.68
        base_x = tip_x - ux * size
        base_y = tip_y - uy * size
        perp_x = -uy
        perp_y = ux
        polygon = [
            (int(round(tip_x)), int(round(tip_y))),
            (int(round(base_x + perp_x * wing)), int(round(base_y + perp_y * wing))),
            (int(round(base_x - perp_x * wing)), int(round(base_y - perp_y * wing))),
        ]
        outline = _expand_polygon_from_centroid(polygon, amount=max(1.5, line_width * 0.55))
        draw.polygon(outline, fill=(0, 0, 0))
        draw.polygon(polygon, fill=color)


def _expand_polygon_from_centroid(
    polygon: list[tuple[int, int]],
    *,
    amount: float,
) -> list[tuple[int, int]]:
    cx = sum(x for x, _ in polygon) / float(len(polygon))
    cy = sum(y for _, y in polygon) / float(len(polygon))
    expanded: list[tuple[int, int]] = []
    for x, y in polygon:
        dx = float(x) - cx
        dy = float(y) - cy
        length = math.hypot(dx, dy) or 1.0
        expanded.append(
            (
                int(round(float(x) + dx / length * amount)),
                int(round(float(y) + dy / length * amount)),
            )
        )
    return expanded


def _expand_rect(
    bbox: PixelBBox,
    padding: int,
    image_size: tuple[int, int],
) -> PixelBBox:
    image_width, image_height = image_size
    x1, y1, x2, y2 = bbox
    return (
        max(0, int(x1) - int(padding)),
        max(0, int(y1) - int(padding)),
        min(max(0, image_width - 1), int(x2) + int(padding)),
        min(max(0, image_height - 1), int(y2) + int(padding)),
    )


def _union_rects(rects: list[PixelBBox]) -> PixelBBox:
    return (
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    )


def _rects_touch(a: PixelBBox, b: PixelBBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _rect_center(rect: PixelBBox) -> tuple[float, float]:
    return (float(rect[0] + rect[2]) / 2.0, float(rect[1] + rect[3]) / 2.0)


def _rect_inside(inner: PixelBBox, outer: PixelBBox) -> bool:
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _rect_near_boundary(inner: PixelBBox, outer: PixelBBox, *, boundary_margin: int) -> bool:
    return (
        inner[0] - outer[0] <= boundary_margin
        or inner[1] - outer[1] <= boundary_margin
        or outer[2] - inner[2] <= boundary_margin
        or outer[3] - inner[3] <= boundary_margin
    )


def _point_inside_rect(point: tuple[int, int], rect: PixelBBox) -> bool:
    x, y = point
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _point_near_boundary(point: tuple[int, int], rect: PixelBBox, *, boundary_margin: int) -> bool:
    x, y = point
    return (
        x - rect[0] <= boundary_margin
        or y - rect[1] <= boundary_margin
        or rect[2] - x <= boundary_margin
        or rect[3] - y <= boundary_margin
    )


def _bbox_inside_zoom_interior(
    bbox: PixelBBox,
    *,
    regions: list[_DenseRegion],
    boundary_margin: int,
) -> bool:
    return any(
        _rect_inside(bbox, region.bbox)
        and not _rect_near_boundary(bbox, region.bbox, boundary_margin=boundary_margin)
        for region in regions
    )


def _point_inside_zoom_interior(
    item: tuple[ShaftVisualPoint, tuple[int, int]] | None,
    *,
    regions: list[_DenseRegion],
    boundary_margin: int,
) -> bool:
    if item is None:
        return False
    _, point = item
    return any(
        _point_inside_rect(point, region.bbox)
        and not _point_near_boundary(point, region.bbox, boundary_margin=boundary_margin)
        for region in regions
    )


def _line_strip_inside_zoom_interior(
    strip_item: tuple[ShaftVisualLineStrip, list[tuple[ShaftVisualPoint, tuple[int, int]]]] | None,
    *,
    regions: list[_DenseRegion],
    boundary_margin: int,
) -> bool:
    if strip_item is None:
        return False
    _, strip_points = strip_item
    if not strip_points:
        return False
    xy_points = [xy for _, xy in strip_points]
    return any(
        all(_point_inside_rect(point, region.bbox) for point in xy_points)
        and not any(
            _point_near_boundary(point, region.bbox, boundary_margin=boundary_margin)
            for point in xy_points
        )
        for region in regions
    )


def _translate_scale_bbox(
    *,
    bbox: PixelBBox,
    crop_origin: tuple[int, int],
    scale: float,
    offset: tuple[int, int],
    image_size: tuple[int, int],
) -> PixelBBox:
    crop_x, crop_y = crop_origin
    offset_x, offset_y = offset
    x1 = int(round((bbox[0] - crop_x) * scale + offset_x))
    y1 = int(round((bbox[1] - crop_y) * scale + offset_y))
    x2 = int(round((bbox[2] - crop_x) * scale + offset_x))
    y2 = int(round((bbox[3] - crop_y) * scale + offset_y))
    image_width, image_height = image_size
    return (
        max(0, min(image_width - 1, x1)),
        max(0, min(image_height - 1, y1)),
        max(0, min(image_width - 1, x2)),
        max(0, min(image_height - 1, y2)),
    )


def _translate_scale_point(
    point: tuple[int, int],
    crop_bbox: PixelBBox,
    scale: float,
    offset: tuple[int, int],
) -> tuple[int, int]:
    x, y = point
    crop_x1, crop_y1, _, _ = crop_bbox
    offset_x, offset_y = offset
    return (
        int(round((x - crop_x1) * scale + offset_x)),
        int(round((y - crop_y1) * scale + offset_y)),
    )


def _clamp_rect(
    x: int,
    y: int,
    width: int,
    height: int,
    image_size: tuple[int, int],
) -> PixelBBox:
    image_width, image_height = image_size
    max_x = max(0, image_width - width)
    max_y = max(0, image_height - height)
    x1 = int(max(0, min(max_x, int(round(x)))))
    y1 = int(max(0, min(max_y, int(round(y)))))
    return x1, y1, min(image_width, x1 + width), min(image_height, y1 + height)


def _fit_label_text(
    *,
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    if _text_width(draw, text, font) <= max_width:
        return text
    if max_width <= _text_width(draw, "...", font):
        return ""

    lo = 1
    hi = len(text)
    best = "..."
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + "..."
        if _text_width(draw, candidate, font) <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _wrap_footer_lines(
    *,
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if _text_width(draw, line, font) <= max_width:
            wrapped.append(line)
            continue
        words = line.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if current and _text_width(draw, candidate, font) > max_width:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        if current:
            wrapped.append(current)
    return wrapped


def _text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> float:
    return float(draw.textbbox((0, 0), text, font=font)[2])


def _font_height(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> int:
    bbox = font.getbbox("Hg") if hasattr(font, "getbbox") else (0, 0, 0, 12)
    return max(1, int(bbox[3] - bbox[1]))


def _format_box_label(box: ShaftVisualBox) -> str:
    label = str(box.label or "box").strip().lower() or "box"
    if box.index is None:
        return label
    return f"{int(box.index)}:{label}"


def _format_point_label(point: ShaftVisualPoint) -> str:
    text = str(point.label or "").strip().lower()
    if point.index is None:
        return text
    prefix = str(int(point.index))
    return f"{prefix}:{text}" if text else prefix


def _pixel_bbox_area(bbox: PixelBBox | None) -> int:
    if bbox is None:
        return 0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _intersection_area(a: PixelBBox, b: PixelBBox) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)
