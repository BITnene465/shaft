from __future__ import annotations

from dataclasses import dataclass
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
class ShaftVisualizationStyle:
    base_max_side: int = 2400
    medium_density_max_side: int = 3200
    high_density_max_side: int = 3800
    medium_density_shapes: int = 40
    high_density_shapes: int = 80
    jpeg_quality: int = 92


DEFAULT_VISUALIZATION_STYLE = ShaftVisualizationStyle()


def resolve_box_line_width(image_width: int, image_height: int) -> int:
    short_edge = max(1, min(int(image_width), int(image_height)))
    return max(3, int(round(short_edge / 200.0)))


def resolve_point_radius(image_width: int, image_height: int) -> int:
    short_edge = max(1, min(int(image_width), int(image_height)))
    return max(4, int(round(short_edge / 145.0)))


def resolve_annotation_font_size(image_width: int, image_height: int) -> int:
    short_edge = max(1, min(int(image_width), int(image_height)))
    return max(14, min(38, int(round(short_edge / 46.0))))


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
    footer_lines: Iterable[str] = (),
    style: ShaftVisualizationStyle = DEFAULT_VISUALIZATION_STYLE,
) -> Image.Image:
    source = ImageOps.exif_transpose(image).convert("RGB")
    box_list = list(boxes)
    point_list = list(points)
    render_scale = _resolve_render_scale(
        image_width=source.width,
        image_height=source.height,
        shape_count=len(box_list) + len(point_list),
        style=style,
    )
    if render_scale < 1.0:
        width = max(1, int(round(source.width * render_scale)))
        height = max(1, int(round(source.height * render_scale)))
        canvas = source.resize((width, height), Image.Resampling.LANCZOS)
    else:
        canvas = source.copy()

    draw = ImageDraw.Draw(canvas)
    font = load_annotation_font(resolve_annotation_font_size(canvas.width, canvas.height))
    line_width = resolve_box_line_width(canvas.width, canvas.height)
    label_rects: list[PixelBBox] = []

    scaled_boxes = [
        (idx, box, _scale_bbox(box.bbox, render_scale, canvas.width, canvas.height))
        for idx, box in enumerate(box_list)
    ]
    scaled_boxes = [item for item in scaled_boxes if item[2] is not None]
    scaled_boxes.sort(key=lambda item: (-_pixel_bbox_area(item[2]), item[0]))
    for _, box, pixel_bbox in scaled_boxes:
        if pixel_bbox is None:
            continue
        _draw_labeled_box(
            draw=draw,
            image_size=(canvas.width, canvas.height),
            bbox=pixel_bbox,
            label=_format_box_label(box),
            color=resolve_label_color(box.label, box.index or 1),
            line_width=line_width,
            font=font,
            occupied_labels=label_rects,
        )

    scaled_points = [
        _scale_point(point, render_scale, canvas.width, canvas.height) for point in point_list
    ]
    _draw_points(
        draw=draw,
        image_size=(canvas.width, canvas.height),
        points=scaled_points,
        line_width=line_width,
        font=font,
        occupied_labels=label_rects,
    )

    return append_footer(canvas, footer_lines)


def save_labeled_visualization(
    *,
    image_path: str | Path,
    output_path: str | Path,
    boxes: Iterable[ShaftVisualBox] = (),
    points: Iterable[ShaftVisualPoint] = (),
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
    outline_width = line_width + max(3, line_width // 2)
    draw.rectangle(bbox, outline=(0, 0, 0), width=outline_width)
    draw.rectangle(bbox, outline=color, width=line_width)
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


def _draw_points(
    *,
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    points: list[tuple[ShaftVisualPoint, tuple[int, int]] | None],
    line_width: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    occupied_labels: list[PixelBBox],
) -> None:
    valid_points = [item for item in points if item is not None]
    if not valid_points:
        return

    xy_points = [xy for _, xy in valid_points]
    point_color = resolve_label_color("keypoint", 1)
    if len(xy_points) >= 2:
        draw.line(xy_points, fill=(0, 0, 0), width=line_width + 3, joint="curve")
        draw.line(xy_points, fill=point_color, width=line_width, joint="curve")

    radius = resolve_point_radius(*image_size)
    for point, (cx, cy) in valid_points:
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(0, 0, 0),
            outline=(0, 0, 0),
            width=line_width,
        )
        draw.ellipse(
            [cx - radius + 2, cy - radius + 2, cx + radius - 2, cy + radius - 2],
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
