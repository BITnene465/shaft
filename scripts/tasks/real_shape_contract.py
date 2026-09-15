"""Lossless field projection and conservative geometry gates for compact real shapes."""

from __future__ import annotations

from copy import deepcopy

import prepare_gt_standard_v5_7 as V


def corner_points(corners: list[dict]) -> list[list]:
    return [
        c[k]
        for c in corners
        for k in (("point",) if c["type"] == "sharp" else ("start", "mid", "end"))
    ]


def cross(a: list, b: list, c: list) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def area(points: list[list]) -> float:
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))


def intersects(a: list, b: list, c: list, d: list) -> bool:
    def on(p, q, r):
        return (
            cross(p, q, r) == 0
            and min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
            and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
        )

    return (
        (cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0)
        or on(a, b, c)
        or on(a, b, d)
        or on(c, d, a)
        or on(c, d, b)
    )


def validate_ring(points: list[list]) -> None:
    if len(points) < 3 or len({tuple(p) for p in points}) != len(points) or area(points) == 0:
        raise ValueError("geometry.degenerate_or_repeated_contour")
    edges = list(zip(points, points[1:] + points[:1]))
    for i, (a, b) in enumerate(edges):
        for j in range(i + 2, len(edges)):
            if i == 0 and j == len(edges) - 1:
                continue
            if intersects(a, b, *edges[j]):
                raise ValueError("geometry.self_intersecting_sampled_contour")


def canonical_corners(corners: list[dict]) -> list[dict]:
    result = deepcopy(corners)
    points = corner_points(result)
    validate_ring(points)
    if area(points) < 0:
        result.reverse()
        for c in result:
            if c["type"] == "round":
                c["start"], c["end"] = c["end"], c["start"]
    anchors = [c.get("point", c.get("mid")) for c in result]
    left, top = min(p[0] for p in anchors), min(p[1] for p in anchors)
    width = max(p[0] for p in anchors) - left
    height = max(p[1] for p in anchors) - top
    if not width or not height:
        raise ValueError("geometry.degenerate_corner_anchors")
    first = min(
        range(len(result)),
        key=lambda i: (
            (anchors[i][0] - left) / width + (anchors[i][1] - top) / height,
            anchors[i][1],
            anchors[i][0],
        ),
    )
    return result[first:] + result[:first]


def geometry_points(p: dict) -> list[list]:
    points = []
    for key in ("corners", "body_corners"):
        points.extend(corner_points(p.get(key, [])))
    for split in p.get("splits", []):
        points.extend(corner_points(split["split_corners"]))
    if "body_bbox" in p:
        x1, y1, x2, y2 = p["body_bbox"]
        points.extend(([x1, y1], [x2, y2]))
    points.extend(p.get("tail", {}).get("points", []))
    return points


def validate_geometry(p: dict) -> None:
    for key in ("corners", "body_corners"):
        if key in p:
            validate_ring(corner_points(p[key]))
            if area(corner_points(p[key])) <= 0:
                raise ValueError("geometry.not_clockwise")
    for split in p.get("splits", []):
        points = corner_points(split["split_corners"])
        if len({tuple(p) for p in points}) != len(points):
            raise ValueError("geometry.degenerate_split")
    if "tail" in p:
        tail = p["tail"]["points"]
        if len({tuple(p) for p in tail}) != 3 or area(tail) == 0:
            raise ValueError("geometry.degenerate_tail")


def normalize(parameters: dict, width: int, height: int) -> dict:
    if not isinstance(parameters, dict) or parameters.get("shape_type") not in V.SHAPE_TYPES:
        raise ValueError("shape.missing_or_unknown_type")
    p = deepcopy(parameters)
    t = p["shape_type"]
    known = {
        "shape_type",
        "border",
        "fill",
        "effect",
        "corners",
        "splits",
        "body_type",
        "body_corners",
        "body_bbox",
        "tail",
        "subbbox",
    }
    if set(p) - known:
        raise ValueError("shape.unknown_fields")
    keys = {"shape_type"} if t == "other" else {"shape_type", "border", "fill", "effect"}
    if t == "card":
        keys |= {"corners", "splits"}
    elif t == "callout":
        if p.get("body_type") not in ("rectangle", "oval"):
            raise ValueError("shape.callout.missing_body_type")
        keys |= {
            "body_type",
            "tail",
            "body_corners" if p["body_type"] == "rectangle" else "body_bbox",
        }
    elif t not in ("other", "oval"):
        keys.add("corners")
    p = {k: v for k, v in p.items() if k in keys}
    # Remove inactive style/color placeholders, not required active attributes.
    styles = [p.get("border")]
    fills = p.get("fill")
    styles += fills if isinstance(fills, list) else [fills]
    styles += p.get("splits", [])
    for value in styles:
        if isinstance(value, dict) and value.get("type") in ("none", "complex"):
            for key in ("style", "color"):
                if key in value:
                    if value[key] not in ("", None):
                        raise ValueError("appearance.inactive_field_has_value")
                    del value[key]
    issues = V._validate_shape(p, width=width, height=height)
    if issues:
        raise ValueError(";".join(issues))
    for key in ("corners", "body_corners"):
        if key in p:
            p[key] = canonical_corners(p[key])
    # Preserve card region/split order and all points. Tail reversal preserves its middle tip.
    if "tail" in p and area(p["tail"]["points"]) < 0:
        p["tail"]["points"].reverse()
    validate_geometry(p)
    return p
