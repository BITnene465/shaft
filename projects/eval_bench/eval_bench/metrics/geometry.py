from __future__ import annotations


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return safe_div(intersection, union)


def keypoint_distance(gt_item: dict, pred_item: dict) -> float | None:
    gt_points = gt_item["points"]
    pred_points = pred_item["points"]
    if len(gt_points) < 2 or len(pred_points) < 2:
        return None
    return (point_distance(gt_points[0], pred_points[0]) + point_distance(gt_points[-1], pred_points[-1])) / 2


def point_distance(point_a: list[float], point_b: list[float]) -> float:
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


def safe_div(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if float(denominator) else 0.0
