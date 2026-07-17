from __future__ import annotations

import re
from typing import Any


HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
SHAPE_TYPES = {
    "rectangle",
    "oval",
    "triangle",
    "trapezoid",
    "parallelogram",
    "diamond",
    "step",
    "regular_pentagon",
    "regular_hexagon",
    "arrow_pentagon",
    "other_polygon",
    "callout",
    "other",
}
FORBIDDEN_SHAPE_GEOMETRY_FIELDS = {
    "bbox",
    "bbox_2d",
    "points",
    "corners",
    "body_corners",
    "body_bbox",
    "tail",
}


def _unexpected_fields(
    value: dict[str, Any],
    *,
    allowed: set[str],
    field: str,
    errors: list[str],
) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        errors.append(f"{field}:unexpected_fields:{','.join(unexpected)}")


def _validate_color(value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(value, str) or HEX_COLOR.fullmatch(value) is None:
        errors.append(f"{field}:invalid_hex_color")


def _validate_border(value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append("border:missing_or_not_object")
        return
    border_type = value.get("type")
    if border_type not in {"none", "uniform", "complex"}:
        errors.append("border.type:invalid")
        _unexpected_fields(value, allowed={"type"}, field="border", errors=errors)
        return
    allowed = {"type"}
    if border_type == "uniform":
        allowed.update({"style", "color"})
        if value.get("style") not in {"solid", "dash", "dot"}:
            errors.append("border.style:invalid")
        _validate_color(value.get("color"), "border.color", errors)
    _unexpected_fields(value, allowed=allowed, field="border", errors=errors)


def _validate_fill(value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append("fill:missing_or_not_object")
        return
    fill_type = value.get("type")
    if fill_type not in {"none", "solid", "linear_gradient", "radial_gradient", "complex"}:
        errors.append("fill.type:invalid")
        _unexpected_fields(value, allowed={"type"}, field="fill", errors=errors)
        return
    allowed = {"type"}
    if fill_type == "solid":
        allowed.add("color")
        _validate_color(value.get("color"), "fill.color", errors)
    if fill_type in {"linear_gradient", "radial_gradient"}:
        allowed.update({"colors", "direction"})
        colors = value.get("colors")
        if not isinstance(colors, list) or len(colors) != 2:
            errors.append("fill.colors:requires_two_colors")
        else:
            for index, color in enumerate(colors):
                _validate_color(color, f"fill.colors[{index}]", errors)
    if fill_type == "linear_gradient" and value.get("direction") not in {
        "bottom_to_top",
        "bottom_left_to_top_right",
        "left_to_right",
        "top_left_to_bottom_right",
    }:
        errors.append("fill.direction:invalid_linear")
    if fill_type == "radial_gradient" and value.get("direction") != "center_to_edge":
        errors.append("fill.direction:invalid_radial")
    _unexpected_fields(value, allowed=allowed, field="fill", errors=errors)


def _validate_effect(value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append("effect:invalid")
        return
    if value.get("type") not in {"none", "shadow", "glow"}:
        errors.append("effect:invalid")
    _unexpected_fields(value, allowed={"type"}, field="effect", errors=errors)


def validate_shape_parameters(parameters: Any) -> list[str]:
    """Validate the exact non-geometric shape-attribute training contract."""

    if not isinstance(parameters, dict):
        return ["parameters:missing_or_not_object"]
    errors: list[str] = []
    forbidden = sorted(FORBIDDEN_SHAPE_GEOMETRY_FIELDS.intersection(parameters))
    if forbidden:
        errors.append(f"parameters:forbidden_geometry:{','.join(forbidden)}")
    shape_type = parameters.get("shape_type")
    if shape_type not in SHAPE_TYPES:
        errors.append("shape_type:invalid")
        return errors
    if shape_type == "other":
        if set(parameters) != {"shape_type"}:
            errors.append("other:must_only_contain_shape_type")
        return errors

    allowed = {"shape_type", "border", "fill", "effect"}
    if shape_type == "callout":
        allowed.add("body_type")
    _unexpected_fields(parameters, allowed=allowed, field="parameters", errors=errors)
    _validate_border(parameters.get("border"), errors)
    _validate_fill(parameters.get("fill"), errors)
    _validate_effect(parameters.get("effect"), errors)
    if shape_type == "callout" and parameters.get("body_type") not in {"rectangle", "oval"}:
        errors.append("callout.body_type:invalid")
    return errors


__all__ = ["validate_shape_parameters"]
