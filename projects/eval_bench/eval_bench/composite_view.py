from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping


VIEW_MODES = ["gt", "prediction", "diff"]
COMPOSITE_SAMPLE_PAGE_SIZE = 500


def build_composite_sample_view(
    store: Any,
    *,
    layer_runs: Mapping[str, str],
    sample_index: int,
    layer_sample_indices: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    normalized_layers = {
        str(layer).strip(): str(run_id).strip()
        for layer, run_id in layer_runs.items()
        if str(layer).strip() and str(run_id).strip()
    }
    if len(normalized_layers) < 2:
        raise ValueError("composite sample view requires at least two layer run ids.")

    run_context = {run.run_id: run for run in store.runs()}
    layer_indexes = {
        layer: _run_image_index(store, run_id)
        for layer, run_id in normalized_layers.items()
    }
    image_keys = _union_image_keys(layer_indexes)
    if not image_keys:
        raise ValueError("composite sample view requires at least one sample image.")
    if sample_index < 0 or sample_index >= len(image_keys):
        raise IndexError(f"sample_index={sample_index} is outside composite image union range.")

    selected_image = image_keys[sample_index]
    layer_indices = dict(layer_sample_indices or {})
    layers: list[dict[str, Any]] = []
    layer_statuses: list[dict[str, Any]] = []
    warnings: list[str] = []
    primary_image: str | None = selected_image
    primary_benchmark_id: str | None = None

    for layer, run_id in normalized_layers.items():
        run = run_context.get(run_id)
        run_index = layer_indexes[layer]
        summary = run_index.by_image.get(selected_image)
        if layer in layer_indices:
            detail = store.run_sample_detail(run_id, sample_index=int(layer_indices[layer]))
            summary = detail.sample
        else:
            detail = (
                store.run_sample_detail(run_id, sample_index=summary.index)
                if summary is not None and summary.has_prediction
                else None
            )
        benchmark_id = run.benchmark_id if run is not None else ""
        if primary_benchmark_id is None and benchmark_id:
            primary_benchmark_id = benchmark_id
        elif benchmark_id and benchmark_id != primary_benchmark_id:
            warnings.append(f"layer {layer!r} uses a different benchmark.")

        status = _layer_status(
            layer=layer,
            run_id=run_id,
            run=run,
            selected_image=selected_image,
            summary=summary,
        )
        layer_statuses.append(status)
        if detail is None:
            continue
        sample = asdict(detail.sample)
        layers.append(
            {
                "layer": layer,
                "run_id": run_id,
                "sample_index": detail.sample.index,
                "status": "ready",
                "available": True,
                "missing_reason": "",
                "image_key": selected_image,
                "benchmark_id": benchmark_id,
                "benchmark_split": run.benchmark_split if run is not None else "",
                "task": run.spec_task if run is not None else "",
                "target_labels": run.target_labels if run is not None else [],
                "sample": sample,
                "gt_instances": detail.gt_instances,
                "pred_instances": detail.pred_instances,
                "raw_payload": detail.raw_payload,
                "prediction_payload": detail.prediction_payload,
                "diagnostics": detail.diagnostics,
                "diagnostic_summary": _diagnostic_summary(detail.diagnostics),
            }
        )

    if not layers:
        warnings.append(f"no selected layer has prediction for image {selected_image!r}.")

    return {
        "kind": "composite_sample_view",
        "sample_index": sample_index,
        "image_index": sample_index,
        "image_count": len(image_keys),
        "image_key": selected_image,
        "image_keys": image_keys,
        "image": primary_image or "",
        "benchmark_id": primary_benchmark_id or "",
        "layer_options": [item["layer"] for item in layer_statuses],
        "view_modes": VIEW_MODES,
        "layers": layers,
        "layer_statuses": layer_statuses,
        "diagnostics": {
            "warnings": sorted(set(warnings)),
            "per_layer": {
                item["layer"]: item["diagnostic_summary"]
                for item in layers
            },
        },
    }


class _RunImageIndex:
    def __init__(self) -> None:
        self.image_order: list[str] = []
        self.by_image: dict[str, Any] = {}

    def add(self, summary: Any) -> None:
        image = str(summary.image or "").strip()
        if not image or image in self.by_image:
            return
        self.image_order.append(image)
        self.by_image[image] = summary


def _run_image_index(store: Any, run_id: str) -> _RunImageIndex:
    output = _RunImageIndex()
    offset = 0
    while True:
        page = store.run_sample_page(
            run_id,
            offset=offset,
            limit=COMPOSITE_SAMPLE_PAGE_SIZE,
        )
        for sample in page.samples:
            output.add(sample)
        offset += len(page.samples)
        if offset >= page.total or not page.samples:
            break
    return output


def _union_image_keys(layer_indexes: Mapping[str, _RunImageIndex]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for index in layer_indexes.values():
        for image in index.image_order:
            if not index.by_image[image].has_prediction:
                continue
            if image in seen:
                continue
            seen.add(image)
            output.append(image)
    return output


def _layer_status(
    *,
    layer: str,
    run_id: str,
    run: Any,
    selected_image: str,
    summary: Any | None,
) -> dict[str, Any]:
    if summary is None:
        return {
            "layer": layer,
            "run_id": run_id,
            "status": "image_missing",
            "available": False,
            "missing_reason": "image_missing",
            "image_key": selected_image,
            "sample_index": None,
            "sample": None,
            "benchmark_id": run.benchmark_id if run is not None else "",
            "benchmark_split": run.benchmark_split if run is not None else "",
            "task": run.spec_task if run is not None else "",
            "target_labels": run.target_labels if run is not None else [],
            "diagnostic_summary": _diagnostic_summary(None),
        }
    status = "ready" if summary.has_prediction else "prediction_missing"
    return {
        "layer": layer,
        "run_id": run_id,
        "status": status,
        "available": summary.has_prediction,
        "missing_reason": "" if summary.has_prediction else "prediction_missing",
        "image_key": selected_image,
        "sample_index": summary.index,
        "sample": asdict(summary),
        "benchmark_id": run.benchmark_id if run is not None else "",
        "benchmark_split": run.benchmark_split if run is not None else "",
        "task": run.spec_task if run is not None else "",
        "target_labels": run.target_labels if run is not None else [],
        "diagnostic_summary": _diagnostic_summary(summary.diagnostics),
    }


def _diagnostic_summary(diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    if not diagnostics:
        return {
            "matched_count": 0,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "labels": [],
        }
    return {
        "matched_count": int(diagnostics.get("matched_count") or 0),
        "false_positive_count": int(diagnostics.get("false_positive_count") or 0),
        "false_negative_count": int(diagnostics.get("false_negative_count") or 0),
        "labels": sorted(str(label) for label in (diagnostics.get("labels") or [])),
    }
