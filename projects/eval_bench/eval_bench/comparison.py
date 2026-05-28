from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .artifacts import DEFAULT_STORE_ROOT, StoreLayout, atomic_write_json, read_json
from .schema import utc_now_iso


def compare_runs(
    *,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    baseline_run_id: str,
    candidate_run_id: str,
) -> Path:
    layout = StoreLayout(store_root)
    baseline = _load_report(layout, baseline_run_id)
    candidate = _load_report(layout, candidate_run_id)
    comparison_id = comparison_id_for_runs(baseline_run_id, candidate_run_id)
    report = compare_report_payloads(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        baseline=baseline,
        candidate=candidate,
    )
    report["comparison_id"] = comparison_id
    report["created_at"] = utc_now_iso()
    output_path = layout.exports_dir / "comparisons" / f"{comparison_id}.json"
    atomic_write_json(output_path, report)
    return output_path


def comparison_id_for_runs(baseline_run_id: str, candidate_run_id: str) -> str:
    return f"{_safe_name(baseline_run_id)}__vs__{_safe_name(candidate_run_id)}"


def load_comparison_report(
    *,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    comparison_id: str | None = None,
    baseline_run_id: str | None = None,
    candidate_run_id: str | None = None,
) -> dict[str, Any]:
    layout = StoreLayout(store_root)
    resolved_id = _resolve_comparison_id(
        comparison_id=comparison_id,
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
    )
    path = layout.exports_dir / "comparisons" / f"{_safe_name(resolved_id)}.json"
    if not path.exists():
        raise FileNotFoundError(f"comparison report does not exist: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"comparison report must be a JSON object: {path}")
    payload.setdefault("comparison_id", path.stem)
    payload.setdefault("path", str(path))
    return payload


def comparison_sample_detail_payload(
    *,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    baseline_run_id: str,
    candidate_run_id: str,
    sample_index: int,
    baseline_sample_index: int | None = None,
    candidate_sample_index: int | None = None,
) -> dict[str, Any]:
    from .store import EvalBenchStore

    store = EvalBenchStore(store_root)
    resolved_baseline_index = sample_index if baseline_sample_index is None else baseline_sample_index
    resolved_candidate_index = sample_index if candidate_sample_index is None else candidate_sample_index
    baseline = store.run_sample_detail(baseline_run_id, sample_index=resolved_baseline_index)
    candidate = store.run_sample_detail(candidate_run_id, sample_index=resolved_candidate_index)
    return {
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "sample_index": sample_index,
        "baseline_index": resolved_baseline_index,
        "candidate_index": resolved_candidate_index,
        "baseline": run_sample_detail_payload(baseline_run_id, baseline),
        "candidate": run_sample_detail_payload(candidate_run_id, candidate),
    }


def run_sample_detail_payload(
    run_id: str,
    detail: Any,
    sample_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample = asdict(detail.sample)
    if sample_extra:
        sample.update(sample_extra)
    return {
        "run_id": run_id,
        "sample": sample,
        "gt_instances": detail.gt_instances,
        "pred_instances": detail.pred_instances,
        "raw_payload": detail.raw_payload,
        "prediction_payload": detail.prediction_payload,
        "diagnostics": detail.diagnostics,
    }


def list_comparison_reports(
    *,
    store_root: str | Path = DEFAULT_STORE_ROOT,
) -> list[dict[str, Any]]:
    layout = StoreLayout(store_root)
    items: list[dict[str, Any]] = []
    for path in sorted((layout.exports_dir / "comparisons").glob("*.json")):
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        items.append(
            {
                "comparison_id": str(payload.get("comparison_id") or path.stem),
                "baseline_run_id": str(payload.get("baseline_run_id") or ""),
                "candidate_run_id": str(payload.get("candidate_run_id") or ""),
                "benchmark_id": str(payload.get("benchmark_id") or ""),
                "benchmark_split": str(payload.get("benchmark_split") or ""),
                "task": str(payload.get("task") or ""),
                "metric_profile": str(payload.get("metric_profile") or ""),
                "target_labels": _target_labels(payload),
                "target_labels_source": payload.get("target_labels_source"),
                "sample_count": int(payload.get("sample_count") or 0),
                "created_at": payload.get("created_at"),
                "path": str(path),
                "delta": dict(payload.get("delta") or {}),
                "summary": dict(payload.get("summary") or {}),
            }
        )
    return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def filter_comparison_reports(
    reports: list[dict[str, Any]],
    *,
    task: str | None = None,
    baseline_run_id: str | None = None,
    candidate_run_id: str | None = None,
    label: str | None = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    task_filter = _filter_value(task)
    baseline_filter = _filter_value(baseline_run_id)
    candidate_filter = _filter_value(candidate_run_id)
    label_filter = _filter_value(label)
    query_filter = _filter_value(query).lower()
    items: list[dict[str, Any]] = []
    for report in reports:
        target_labels = _target_labels(report)
        if task_filter and str(report.get("task") or "") != task_filter:
            continue
        if baseline_filter and str(report.get("baseline_run_id") or "") != baseline_filter:
            continue
        if candidate_filter and str(report.get("candidate_run_id") or "") != candidate_filter:
            continue
        if label_filter and label_filter not in target_labels:
            continue
        if query_filter and not _comparison_query_matches(report, target_labels, query_filter):
            continue
        items.append(report)
    return items


def _resolve_comparison_id(
    *,
    comparison_id: str | None,
    baseline_run_id: str | None,
    candidate_run_id: str | None,
) -> str:
    value = _filter_value(comparison_id)
    baseline = _filter_value(baseline_run_id)
    candidate = _filter_value(candidate_run_id)
    if value and (baseline or candidate):
        raise ValueError("use either comparison_id or baseline/candidate run ids, not both.")
    if value:
        return value
    if baseline and candidate:
        return comparison_id_for_runs(baseline, candidate)
    raise ValueError("comparison_id or both baseline_run_id and candidate_run_id are required.")


def compare_report_payloads(
    *,
    baseline_run_id: str,
    candidate_run_id: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_samples = _samples_by_key(baseline)
    candidate_samples = _samples_by_key(candidate)
    keys = sorted(set(baseline_samples) | set(candidate_samples))
    sample_deltas = [
        _compare_sample(key, baseline_samples.get(key), candidate_samples.get(key)) for key in keys
    ]
    improved = [item for item in sample_deltas if item["status"] == "improved"]
    regressed = [item for item in sample_deltas if item["status"] == "regressed"]
    changed = [item for item in sample_deltas if item["status"] != "unchanged"]
    label_deltas = _compare_run_labels(baseline, candidate)
    return {
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "benchmark_id": candidate.get("benchmark_id") or baseline.get("benchmark_id"),
        "benchmark_split": candidate.get("benchmark_split") or baseline.get("benchmark_split"),
        "task": candidate.get("task") or baseline.get("task"),
        "metric_profile": candidate.get("metric_profile") or baseline.get("metric_profile"),
        "target_labels": _target_labels(candidate) or _target_labels(baseline),
        "target_labels_source": candidate.get("target_labels_source")
        or baseline.get("target_labels_source"),
        "warnings": _comparison_warnings(baseline, candidate),
        "sample_count": len(keys),
        "baseline": _run_metrics(baseline),
        "candidate": _run_metrics(candidate),
        "delta": {
            "precision_iou50": _metric_delta(candidate, baseline, "precision_iou50"),
            "recall_iou50": _metric_delta(candidate, baseline, "recall_iou50"),
            "mean_iou": _metric_delta(candidate, baseline, "mean_iou"),
            "mean_keypoint_distance": _metric_delta(
                candidate,
                baseline,
                "mean_keypoint_distance",
            ),
            "matched_count": int(candidate.get("matched_count") or 0)
            - int(baseline.get("matched_count") or 0),
            "keypoint_pair_count": int(candidate.get("keypoint_pair_count") or 0)
            - int(baseline.get("keypoint_pair_count") or 0),
            "false_positive_count": sum(int(item["delta"]["false_positive_count"]) for item in sample_deltas),
            "false_negative_count": sum(int(item["delta"]["false_negative_count"]) for item in sample_deltas),
        },
        "summary": {
            "improved_samples": len(improved),
            "regressed_samples": len(regressed),
            "changed_samples": len(changed),
            "unchanged_samples": len(sample_deltas) - len(changed),
            "missing_in_baseline": sum(1 for item in sample_deltas if item["baseline"] is None),
            "missing_in_candidate": sum(1 for item in sample_deltas if item["candidate"] is None),
            "improved_labels": sum(1 for item in label_deltas if float(item["delta_score"]) > 0),
            "regressed_labels": sum(1 for item in label_deltas if float(item["delta_score"]) < 0),
        },
        "labels": label_deltas,
        "samples": sample_deltas,
        "top_improvements": sorted(improved, key=lambda item: float(item["delta_score"]), reverse=True)[:50],
        "top_regressions": sorted(regressed, key=lambda item: float(item["delta_score"]))[:50],
    }


def _load_report(layout: StoreLayout, run_id: str) -> dict[str, Any]:
    path = layout.runs_dir / run_id / "reports" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"run report does not exist: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"run report must be a JSON object: {path}")
    return payload


def _samples_by_key(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    samples = report.get("samples") or []
    if not isinstance(samples, list):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        key = str(sample.get("json_path") or sample.get("image") or sample.get("index"))
        indexed[key] = sample
    return indexed


def _compare_sample(
    key: str,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_metrics = _sample_metrics(baseline)
    candidate_metrics = _sample_metrics(candidate)
    baseline_index = _sample_index(baseline)
    candidate_index = _sample_index(candidate)
    delta = {
        "matched_count": candidate_metrics["matched_count"] - baseline_metrics["matched_count"],
        "false_positive_count": candidate_metrics["false_positive_count"]
        - baseline_metrics["false_positive_count"],
        "false_negative_count": candidate_metrics["false_negative_count"]
        - baseline_metrics["false_negative_count"],
        "mean_iou": candidate_metrics["mean_iou"] - baseline_metrics["mean_iou"],
        "keypoint_pair_count": candidate_metrics["keypoint_pair_count"]
        - baseline_metrics["keypoint_pair_count"],
        "mean_keypoint_distance": candidate_metrics["mean_keypoint_distance"]
        - baseline_metrics["mean_keypoint_distance"],
    }
    delta_score = (
        delta["matched_count"]
        - delta["false_positive_count"]
        - delta["false_negative_count"]
        + delta["mean_iou"] * 0.25
        + delta["keypoint_pair_count"]
        - delta["mean_keypoint_distance"] * 0.05
    )
    if delta_score > 0:
        status = "improved"
    elif delta_score < 0:
        status = "regressed"
    else:
        status = "unchanged"
    return {
        "key": key,
        "image": _sample_image(candidate) or _sample_image(baseline),
        "sample_index": candidate_index if candidate_index is not None else baseline_index,
        "baseline_index": baseline_index,
        "candidate_index": candidate_index,
        "status": status,
        "delta_score": delta_score,
        "delta": delta,
        "baseline": baseline_metrics if baseline is not None else None,
        "candidate": candidate_metrics if candidate is not None else None,
        "labels": _compare_sample_labels(baseline, candidate),
    }


def _sample_metrics(sample: dict[str, Any] | None) -> dict[str, Any]:
    if sample is None:
        return {
            "matched_count": 0,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "mean_iou": 0.0,
            "keypoint_pair_count": 0,
            "mean_keypoint_distance": 0.0,
        }
    return {
        "matched_count": int(sample.get("matched_count") or 0),
        "false_positive_count": int(sample.get("false_positive_count") or 0),
        "false_negative_count": int(sample.get("false_negative_count") or 0),
        "mean_iou": float(sample.get("mean_iou") or 0.0),
        "keypoint_pair_count": int(sample.get("keypoint_pair_count") or 0),
        "mean_keypoint_distance": float(sample.get("mean_keypoint_distance") or 0.0),
    }


def _compare_sample_labels(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_labels = baseline.get("labels") if isinstance(baseline, dict) else {}
    candidate_labels = candidate.get("labels") if isinstance(candidate, dict) else {}
    if not isinstance(baseline_labels, dict):
        baseline_labels = {}
    if not isinstance(candidate_labels, dict):
        candidate_labels = {}
    labels: dict[str, Any] = {}
    for label in sorted(set(baseline_labels) | set(candidate_labels)):
        base = _sample_metrics(baseline_labels.get(label))
        cand = _sample_metrics(candidate_labels.get(label))
        labels[label] = {
            "matched_count": cand["matched_count"] - base["matched_count"],
            "false_positive_count": cand["false_positive_count"] - base["false_positive_count"],
            "false_negative_count": cand["false_negative_count"] - base["false_negative_count"],
            "mean_iou": cand["mean_iou"] - base["mean_iou"],
            "keypoint_pair_count": cand["keypoint_pair_count"] - base["keypoint_pair_count"],
            "mean_keypoint_distance": cand["mean_keypoint_distance"]
            - base["mean_keypoint_distance"],
        }
    return labels


def _compare_run_labels(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_labels = _label_metrics_by_label(baseline)
    candidate_labels = _label_metrics_by_label(candidate)
    rows: list[dict[str, Any]] = []
    for label in sorted(set(baseline_labels) | set(candidate_labels)):
        base = _run_label_metrics(baseline_labels.get(label))
        cand = _run_label_metrics(candidate_labels.get(label))
        delta = {
            "precision_iou50": cand["precision_iou50"] - base["precision_iou50"],
            "recall_iou50": cand["recall_iou50"] - base["recall_iou50"],
            "mean_iou": cand["mean_iou"] - base["mean_iou"],
            "mean_keypoint_distance": cand["mean_keypoint_distance"]
            - base["mean_keypoint_distance"],
            "matched_count": cand["matched_count"] - base["matched_count"],
            "keypoint_pair_count": cand["keypoint_pair_count"] - base["keypoint_pair_count"],
            "false_positive_count": cand["false_positive_count"] - base["false_positive_count"],
            "false_negative_count": cand["false_negative_count"] - base["false_negative_count"],
        }
        delta_score = (
            delta["matched_count"]
            - delta["false_positive_count"]
            - delta["false_negative_count"]
            + delta["mean_iou"] * 0.25
            + delta["keypoint_pair_count"]
            - delta["mean_keypoint_distance"] * 0.05
        )
        rows.append(
            {
                "label": label,
                "baseline": base,
                "candidate": cand,
                "delta": delta,
                "delta_score": delta_score,
            }
        )
    return sorted(rows, key=lambda item: abs(float(item["delta_score"])), reverse=True)


def _label_metrics_by_label(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    labels = report.get("labels") or []
    if not isinstance(labels, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in labels:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if label:
            result[label] = item
    return result


def _run_label_metrics(label: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(label, dict):
        return {
            "gt_count": 0,
            "pred_count": 0,
            "matched_count": 0,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "precision_iou50": 0.0,
            "recall_iou50": 0.0,
            "mean_iou": 0.0,
            "keypoint_pair_count": 0,
            "mean_keypoint_distance": 0.0,
        }
    gt_count = int(label.get("gt_count") or 0)
    pred_count = int(label.get("pred_count") or 0)
    matched_count = int(label.get("matched_count") or 0)
    return {
        "gt_count": gt_count,
        "pred_count": pred_count,
        "matched_count": matched_count,
        "false_positive_count": max(0, pred_count - matched_count),
        "false_negative_count": max(0, gt_count - matched_count),
        "precision_iou50": float(label.get("precision_iou50") or 0.0),
        "recall_iou50": float(label.get("recall_iou50") or 0.0),
        "mean_iou": float(label.get("mean_iou") or 0.0),
        "keypoint_pair_count": int(label.get("keypoint_pair_count") or 0),
        "mean_keypoint_distance": float(label.get("mean_keypoint_distance") or 0.0),
    }


def _run_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "precision_iou50": float(report.get("precision_iou50") or 0.0),
        "recall_iou50": float(report.get("recall_iou50") or 0.0),
        "mean_iou": float(report.get("mean_iou") or 0.0),
        "keypoint_pair_count": int(report.get("keypoint_pair_count") or 0),
        "mean_keypoint_distance": float(report.get("mean_keypoint_distance") or 0.0),
        "matched_count": int(report.get("matched_count") or 0),
        "gt_instance_count": int(report.get("gt_instance_count") or 0),
        "pred_instance_count": int(report.get("pred_instance_count") or 0),
    }


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return float(candidate.get(key) or 0.0) - float(baseline.get(key) or 0.0)


def _target_labels(report: dict[str, Any]) -> list[str]:
    labels = report.get("target_labels") or []
    if not isinstance(labels, list):
        return []
    return [str(item) for item in labels if str(item).strip()]


def _filter_value(value: str | None) -> str:
    return str(value).strip() if value is not None else ""


def _comparison_query_matches(
    report: dict[str, Any],
    target_labels: list[str],
    query: str,
) -> bool:
    fields = [
        report.get("comparison_id"),
        report.get("baseline_run_id"),
        report.get("candidate_run_id"),
        report.get("benchmark_id"),
        report.get("benchmark_split"),
        report.get("task"),
        report.get("metric_profile"),
        " ".join(target_labels),
    ]
    return any(query in str(field or "").lower() for field in fields)


def _comparison_warnings(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if str(baseline.get("benchmark_id") or "") != str(candidate.get("benchmark_id") or ""):
        warnings.append("baseline and candidate benchmarks differ")
    if str(baseline.get("benchmark_split") or "") != str(candidate.get("benchmark_split") or ""):
        warnings.append("baseline and candidate benchmark splits differ")
    if str(baseline.get("task") or "") != str(candidate.get("task") or ""):
        warnings.append("baseline and candidate tasks differ")
    if str(baseline.get("metric_profile") or "") != str(candidate.get("metric_profile") or ""):
        warnings.append("baseline and candidate metric profiles differ")
    baseline_labels = _target_labels(baseline)
    candidate_labels = _target_labels(candidate)
    if baseline_labels != candidate_labels:
        warnings.append("baseline and candidate target labels differ")
    return warnings


def _sample_image(sample: dict[str, Any] | None) -> str | None:
    if not isinstance(sample, dict):
        return None
    image = sample.get("image")
    return str(image) if image else None


def _sample_index(sample: dict[str, Any] | None) -> int | None:
    if not isinstance(sample, dict):
        return None
    try:
        return int(sample.get("index"))
    except (TypeError, ValueError):
        return None


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
