from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..metric_profiles import MetricProfile
from .geometry import bbox_iou, keypoint_distance, safe_div


@dataclass(frozen=True)
class LabelMetric:
    label: str
    gt_count: int = 0
    pred_count: int = 0
    matched_count: int = 0
    precision_iou50: float = 0.0
    recall_iou50: float = 0.0
    mean_iou: float = 0.0
    keypoint_pair_count: int = 0
    mean_keypoint_distance: float | None = None


@dataclass(frozen=True)
class MetricSample:
    sample_index: int
    json_relative: Path
    image: Path
    gt_instances: list[dict[str, Any]]
    pred_instances: list[dict[str, Any]]
    has_prediction: bool


@dataclass(frozen=True)
class MetricEvaluation:
    labels: list[LabelMetric]
    samples: list[dict[str, Any]]
    gt_instance_count: int
    pred_instance_count: int
    matched_count: int
    precision_iou50: float
    recall_iou50: float
    mean_iou: float
    keypoint_pair_count: int = 0
    mean_keypoint_distance: float | None = None


@dataclass(frozen=True)
class _CandidateMatch:
    gt_index: int
    pred_index: int
    sort_key: tuple[float, float]
    payload: dict[str, Any]


@dataclass
class _FlowEdge:
    to: int
    rev: int
    cap: int
    cost: int
    candidate: _CandidateMatch | None = None


@dataclass
class _LabelStats:
    gt_count: int = 0
    pred_count: int = 0
    matched_count: int = 0
    iou_sum: float = 0.0
    keypoint_pair_count: int = 0
    keypoint_distance_sum: float = 0.0


def evaluate_metric_samples(
    samples: list[MetricSample],
    *,
    profile: MetricProfile,
    iou_threshold: float | None = None,
) -> MetricEvaluation:
    label_stats: dict[str, _LabelStats] = defaultdict(_LabelStats)
    sample_reports: list[dict[str, Any]] = []
    for sample in samples:
        sample_report = _sample_diagnostic(
            sample,
            profile=profile,
            iou_threshold=iou_threshold if iou_threshold is not None else profile.iou_threshold,
        )
        sample_reports.append(sample_report)
        _accumulate_sample(label_stats=label_stats, sample_report=sample_report, profile=profile)

    label_metrics = [_finalize_label(label, stats) for label, stats in sorted(label_stats.items())]
    gt_count = sum(item.gt_count for item in label_metrics)
    pred_count = sum(item.pred_count for item in label_metrics)
    matched_count = sum(item.matched_count for item in label_metrics)
    weighted_iou_sum = sum(
        float(item.mean_iou) * int(item.matched_count)
        for item in label_metrics
        if item.matched_count > 0
    )
    keypoint_pair_count = sum(item.keypoint_pair_count for item in label_metrics)
    keypoint_distance_sum = sum(
        float(item.mean_keypoint_distance) * int(item.keypoint_pair_count)
        for item in label_metrics
        if item.mean_keypoint_distance is not None and item.keypoint_pair_count > 0
    )
    return MetricEvaluation(
        labels=label_metrics,
        samples=sample_reports,
        gt_instance_count=gt_count,
        pred_instance_count=pred_count,
        matched_count=matched_count,
        precision_iou50=safe_div(matched_count, pred_count),
        recall_iou50=safe_div(matched_count, gt_count),
        mean_iou=safe_div(weighted_iou_sum, matched_count),
        keypoint_pair_count=keypoint_pair_count,
        mean_keypoint_distance=(
            safe_div(keypoint_distance_sum, keypoint_pair_count)
            if keypoint_pair_count > 0
            else None
        ),
    )


def _sample_diagnostic(
    sample: MetricSample,
    *,
    profile: MetricProfile,
    iou_threshold: float,
) -> dict[str, Any]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
    labels: dict[str, dict[str, Any]] = {}

    gt_instances = sample.gt_instances
    pred_instances = sample.pred_instances
    for label in sorted({item["label"] for item in gt_instances + pred_instances}):
        gt_indices = [index for index, item in enumerate(gt_instances) if item["label"] == label]
        pred_indices = [index for index, item in enumerate(pred_instances) if item["label"] == label]
        gt_for_label = [gt_instances[index] for index in gt_indices]
        pred_for_label = [pred_instances[index] for index in pred_indices]
        label_matches = _match_label_instances(
            gt_for_label,
            pred_for_label,
            profile=profile,
            iou_threshold=iou_threshold,
        )
        iou_sum = 0.0
        keypoint_distance_sum = 0.0
        keypoint_pair_count = 0
        for local_match in label_matches:
            gt_index = gt_indices[local_match.gt_index]
            pred_index = pred_indices[local_match.pred_index]
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            iou = float(local_match.payload.get("iou") or 0.0)
            iou_sum += iou
            match = {
                "label": label,
                "gt_index": gt_index,
                "pred_index": pred_index,
                **local_match.payload,
            }
            distance = local_match.payload.get("keypoint_distance")
            if distance is not None:
                keypoint_pair_count += 1
                keypoint_distance_sum += float(distance)
            matches.append(match)
        labels[label] = {
            "gt_count": len(gt_indices),
            "pred_count": len(pred_indices),
            "matched_count": len(label_matches),
            "false_negative_count": len(gt_indices) - len(label_matches),
            "false_positive_count": len(pred_indices) - len(label_matches),
            "mean_iou": safe_div(iou_sum, len(label_matches)),
            "keypoint_pair_count": keypoint_pair_count,
            "mean_keypoint_distance": (
                safe_div(keypoint_distance_sum, keypoint_pair_count)
                if keypoint_pair_count > 0
                else None
            ),
        }

    matches.sort(key=lambda item: (str(item["label"]), int(item["gt_index"]), int(item["pred_index"])))
    false_negatives = [
        _instance_reference(index=index, instance=instance)
        for index, instance in enumerate(gt_instances)
        if index not in matched_gt
    ]
    false_positives = [
        _instance_reference(index=index, instance=instance)
        for index, instance in enumerate(pred_instances)
        if index not in matched_pred
    ]
    matched_count = len(matches)
    keypoint_distances = [
        float(item["keypoint_distance"])
        for item in matches
        if item.get("keypoint_distance") is not None
    ]
    return {
        "index": sample.sample_index,
        "json_path": str(sample.json_relative),
        "image": str(sample.image),
        "has_prediction": sample.has_prediction,
        "gt_instance_count": len(gt_instances),
        "pred_instance_count": len(pred_instances),
        "matched_count": matched_count,
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "mean_iou": safe_div(sum(float(item["iou"]) for item in matches), matched_count),
        "keypoint_pair_count": len(keypoint_distances),
        "mean_keypoint_distance": (
            safe_div(sum(keypoint_distances), len(keypoint_distances))
            if keypoint_distances
            else None
        ),
        "matches": matches,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "labels": labels,
    }


def _match_label_instances(
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    *,
    profile: MetricProfile,
    iou_threshold: float,
) -> list[_CandidateMatch]:
    if profile.matcher == "bbox_iou":
        candidates = _bbox_iou_candidates(gt_instances, pred_instances, iou_threshold=iou_threshold)
        return _greedy_assign(candidates)
    if profile.matcher == "bbox_iou_maxmatch":
        candidates = _bbox_iou_candidates(gt_instances, pred_instances, iou_threshold=iou_threshold)
        return _max_cardinality_max_iou_assign(candidates)
    elif profile.matcher == "ordered_endpoint_distance":
        candidates = _endpoint_distance_candidates(
            gt_instances,
            pred_instances,
            endpoint_threshold_px=profile.endpoint_threshold_px,
        )
        return _greedy_assign(candidates)
    else:
        raise ValueError(f"unsupported metric matcher: {profile.matcher!r}")


def _bbox_iou_candidates(
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> list[_CandidateMatch]:
    candidates: list[_CandidateMatch] = []
    for gt_index, gt_item in enumerate(gt_instances):
        for pred_index, pred_item in enumerate(pred_instances):
            iou = bbox_iou(gt_item["bbox"], pred_item["bbox"])
            if iou >= iou_threshold:
                candidates.append(
                    _CandidateMatch(
                        gt_index=gt_index,
                        pred_index=pred_index,
                        sort_key=(-iou, 0.0),
                        payload={"iou": iou},
                    )
                )
    return candidates


def _endpoint_distance_candidates(
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    *,
    endpoint_threshold_px: float | None,
) -> list[_CandidateMatch]:
    threshold = endpoint_threshold_px if endpoint_threshold_px is not None else 20.0
    candidates: list[_CandidateMatch] = []
    for gt_index, gt_item in enumerate(gt_instances):
        for pred_index, pred_item in enumerate(pred_instances):
            distance = keypoint_distance(gt_item, pred_item)
            if distance is None or distance > threshold:
                continue
            iou = bbox_iou(gt_item["bbox"], pred_item["bbox"])
            candidates.append(
                _CandidateMatch(
                    gt_index=gt_index,
                    pred_index=pred_index,
                    sort_key=(distance, -iou),
                    payload={"iou": iou, "keypoint_distance": distance},
                )
            )
    return candidates


def _greedy_assign(candidates: list[_CandidateMatch]) -> list[_CandidateMatch]:
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[_CandidateMatch] = []
    for candidate in sorted(candidates, key=lambda item: item.sort_key):
        if candidate.gt_index in used_gt or candidate.pred_index in used_pred:
            continue
        used_gt.add(candidate.gt_index)
        used_pred.add(candidate.pred_index)
        matches.append(candidate)
    return matches


def _max_cardinality_max_iou_assign(candidates: list[_CandidateMatch]) -> list[_CandidateMatch]:
    """Exact bipartite assignment: maximize match count, then matched IoU sum."""
    if not candidates:
        return []

    gt_ids = sorted({candidate.gt_index for candidate in candidates})
    pred_ids = sorted({candidate.pred_index for candidate in candidates})
    gt_to_local = {gt_index: index for index, gt_index in enumerate(gt_ids)}
    pred_to_local = {pred_index: index for index, pred_index in enumerate(pred_ids)}

    source = 0
    gt_offset = 1
    pred_offset = gt_offset + len(gt_ids)
    sink = pred_offset + len(pred_ids)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]

    def add_edge(frm: int, to: int, cap: int, cost: int, candidate: _CandidateMatch | None = None) -> None:
        forward = _FlowEdge(to=to, rev=len(graph[to]), cap=cap, cost=cost, candidate=candidate)
        backward = _FlowEdge(to=frm, rev=len(graph[frm]), cap=0, cost=-cost)
        graph[frm].append(forward)
        graph[to].append(backward)

    for gt_local in range(len(gt_ids)):
        add_edge(source, gt_offset + gt_local, 1, 0)
    for pred_local in range(len(pred_ids)):
        add_edge(pred_offset + pred_local, sink, 1, 0)
    for candidate in candidates:
        gt_node = gt_offset + gt_to_local[candidate.gt_index]
        pred_node = pred_offset + pred_to_local[candidate.pred_index]
        iou_cost = -int(round(float(candidate.payload.get("iou") or 0.0) * 1_000_000))
        add_edge(gt_node, pred_node, 1, iou_cost, candidate)

    while _augment_shortest_path(graph, source=source, sink=sink):
        pass

    matches: list[_CandidateMatch] = []
    for gt_local in range(len(gt_ids)):
        gt_node = gt_offset + gt_local
        for edge in graph[gt_node]:
            if edge.candidate is not None and edge.cap == 0:
                matches.append(edge.candidate)
    return sorted(matches, key=lambda item: (item.gt_index, item.pred_index))


def _augment_shortest_path(graph: list[list[_FlowEdge]], *, source: int, sink: int) -> bool:
    inf = 10**18
    dist = [inf] * len(graph)
    in_queue = [False] * len(graph)
    prev_node = [-1] * len(graph)
    prev_edge = [-1] * len(graph)
    dist[source] = 0
    queue: deque[int] = deque([source])
    in_queue[source] = True

    while queue:
        node = queue.popleft()
        in_queue[node] = False
        for edge_index, edge in enumerate(graph[node]):
            if edge.cap <= 0:
                continue
            next_dist = dist[node] + edge.cost
            if next_dist >= dist[edge.to]:
                continue
            dist[edge.to] = next_dist
            prev_node[edge.to] = node
            prev_edge[edge.to] = edge_index
            if not in_queue[edge.to]:
                queue.append(edge.to)
                in_queue[edge.to] = True

    if prev_node[sink] < 0:
        return False

    node = sink
    while node != source:
        parent = prev_node[node]
        edge_index = prev_edge[node]
        edge = graph[parent][edge_index]
        edge.cap -= 1
        graph[node][edge.rev].cap += 1
        node = parent
    return True


def _instance_reference(*, index: int, instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "label": instance["label"],
        "bbox": instance["bbox"],
    }


def _accumulate_sample(
    *,
    label_stats: dict[str, _LabelStats],
    sample_report: dict[str, Any],
    profile: MetricProfile,
) -> None:
    labels = sample_report.get("labels") or {}
    if not isinstance(labels, dict):
        return
    for label, sample_label_stats in labels.items():
        if not isinstance(sample_label_stats, dict):
            continue
        stats = label_stats[label]
        stats.gt_count += int(sample_label_stats.get("gt_count") or 0)
        stats.pred_count += int(sample_label_stats.get("pred_count") or 0)
        stats.matched_count += int(sample_label_stats.get("matched_count") or 0)
        stats.iou_sum += float(sample_label_stats.get("mean_iou") or 0.0) * int(
            sample_label_stats.get("matched_count") or 0
        )
        distance = sample_label_stats.get("mean_keypoint_distance")
        pair_count = int(sample_label_stats.get("keypoint_pair_count") or 0)
        if profile.task == "keypoint" and distance is not None and pair_count > 0:
            stats.keypoint_pair_count += pair_count
            stats.keypoint_distance_sum += float(distance) * pair_count


def _finalize_label(label: str, stats: _LabelStats) -> LabelMetric:
    mean_keypoint_distance = (
        safe_div(stats.keypoint_distance_sum, stats.keypoint_pair_count)
        if stats.keypoint_pair_count > 0
        else None
    )
    return LabelMetric(
        label=label,
        gt_count=stats.gt_count,
        pred_count=stats.pred_count,
        matched_count=stats.matched_count,
        precision_iou50=safe_div(stats.matched_count, stats.pred_count),
        recall_iou50=safe_div(stats.matched_count, stats.gt_count),
        mean_iou=safe_div(stats.iou_sum, stats.matched_count),
        keypoint_pair_count=stats.keypoint_pair_count,
        mean_keypoint_distance=mean_keypoint_distance,
    )
