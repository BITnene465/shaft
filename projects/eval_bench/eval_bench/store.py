from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import DEFAULT_STORE_ROOT, RunArtifacts, StoreLayout
from .sample_paths import sample_image_string
from .sample_scope import (
    filter_instances_by_labels,
    filter_payload_instances,
    run_target_label_set,
    run_target_labels,
    scope_sample_diagnostics,
)


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _line_count(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


@dataclass(frozen=True)
class BenchmarkSummary:
    benchmark_id: str
    tasks: list[str]
    layers: list[str]
    split: str
    sample_count: int
    root: str
    manifest_path: str
    created_at: str | None = None
    source_manifest_path: str | None = None


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    status: str
    benchmark_id: str
    tasks: list[str]
    spec_task: str
    model_id: str
    model_path: str
    prompt_id: str
    prompt_path: str | None
    prompt_hash: str | None
    prompt_metadata: dict[str, Any]
    parser: str
    metric_profile: str
    visualization_profile: str
    inference: dict[str, Any]
    created_at: str | None
    prediction_count: int
    report_count: int
    manifest_path: str
    report_path: str | None = None
    precision_iou50: float | None = None
    recall_iou50: float | None = None
    mean_iou: float | None = None


@dataclass(frozen=True)
class RunSampleSummary:
    index: int
    image: str
    json_path: str
    image_width: int | None
    image_height: int | None
    gt_instance_count: int
    pred_instance_count: int
    labels: list[str]
    has_prediction: bool
    prediction_path: str | None = None
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunSamplePage:
    offset: int
    limit: int
    total: int
    labels: list[str]
    samples: list[RunSampleSummary]


@dataclass(frozen=True)
class BenchmarkSampleSummary:
    index: int
    image: str
    json_path: str
    image_width: int | None
    image_height: int | None
    instance_count: int
    labels: list[str]


@dataclass(frozen=True)
class BenchmarkSamplePage:
    offset: int
    limit: int
    total: int
    labels: list[str]
    samples: list[BenchmarkSampleSummary]


@dataclass(frozen=True)
class RunSampleDetail:
    sample: RunSampleSummary
    gt_instances: list[dict[str, Any]]
    pred_instances: list[dict[str, Any]]
    raw_payload: dict[str, Any]
    prediction_payload: dict[str, Any] | None
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkSampleDetail:
    sample: BenchmarkSampleSummary
    gt_instances: list[dict[str, Any]]
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class DashboardState:
    store_root: str
    benchmark_count: int
    run_count: int
    total_benchmark_samples: int
    prediction_count: int
    benchmarks: list[BenchmarkSummary]
    runs: list[RunSummary]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvalBenchStore:
    def __init__(self, root: str | Path = DEFAULT_STORE_ROOT) -> None:
        self.layout = StoreLayout(root)
        self._benchmark_label_cache: dict[str, list[str]] = {}
        self._benchmark_sample_paths_cache: dict[str, list[Path]] = {}
        self._run_label_cache: dict[str, list[str]] = {}
        self._run_metrics_cache: dict[str, dict[int, dict[str, Any]]] = {}
        self._run_sample_paths_cache: dict[str, list[Path]] = {}

    def benchmarks(self) -> list[BenchmarkSummary]:
        items: list[BenchmarkSummary] = []
        for manifest_path in sorted(self.layout.benchmarks_dir.glob("*/benchmark.json")):
            payload = _read_json_or_none(manifest_path)
            if payload is None:
                continue
            split_path = Path(str(payload.get("manifest_path") or ""))
            sample_count = int(payload.get("sample_count") or 0)
            if sample_count <= 0 and split_path.exists():
                sample_count = _line_count(split_path)
            items.append(
                BenchmarkSummary(
                    benchmark_id=str(payload.get("benchmark_id") or manifest_path.parent.name),
                    tasks=[str(item) for item in payload.get("tasks") or []],
                    layers=[str(item) for item in payload.get("layers") or []],
                    split=str(payload.get("split") or ""),
                    sample_count=sample_count,
                    root=str(payload.get("root") or ""),
                    manifest_path=str(payload.get("manifest_path") or ""),
                    created_at=payload.get("created_at"),
                    source_manifest_path=payload.get("source_manifest_path"),
                )
            )
        return sorted(items, key=lambda item: item.benchmark_id)

    def runs(self) -> list[RunSummary]:
        items: list[RunSummary] = []
        for manifest_path in sorted(self.layout.runs_dir.glob("*/run.json")):
            payload = _read_json_or_none(manifest_path)
            if payload is None:
                continue
            benchmark = payload.get("benchmark") or payload.get("dataset") or {}
            spec = payload.get("spec") or {}
            model = payload.get("model") or {}
            prompt = spec.get("prompt") or {}
            inference = spec.get("inference") or {}
            run_dir = manifest_path.parent
            report_count = len(list((run_dir / "reports").rglob("*")))
            report_path = run_dir / "reports" / "metrics.json"
            summary_path = run_dir / "reports" / "summary.json"
            report_payload = _read_json_or_none(summary_path) or _read_json_or_none(report_path)
            prediction_count = _optional_int(report_payload, "prediction_file_count")
            if prediction_count is None:
                prediction_count = len(list((run_dir / "predictions").rglob("*.json")))
            items.append(
                RunSummary(
                    run_id=str(payload.get("run_id") or run_dir.name),
                    status=str(payload.get("status") or "unknown"),
                    benchmark_id=str(
                        benchmark.get("benchmark_id")
                        or benchmark.get("dataset_id")
                        or benchmark.get("id")
                        or ""
                    ),
                    tasks=[str(item) for item in benchmark.get("tasks") or []],
                    spec_task=str(spec.get("task") or ""),
                    model_id=str(model.get("model_id") or ""),
                    model_path=str(model.get("path") or ""),
                    prompt_id=str(prompt.get("prompt_id") or ""),
                    prompt_path=prompt.get("path"),
                    prompt_hash=prompt.get("text_hash"),
                    prompt_metadata=dict(prompt.get("metadata") or {}),
                    parser=str(spec.get("parser") or ""),
                    metric_profile=str(spec.get("metric_profile") or ""),
                    visualization_profile=str(spec.get("visualization_profile") or ""),
                    inference=dict(inference) if isinstance(inference, dict) else {},
                    created_at=payload.get("created_at"),
                    prediction_count=prediction_count,
                    report_count=report_count,
                    manifest_path=str(manifest_path),
                    report_path=str(report_path) if report_path.exists() else None,
                    precision_iou50=_optional_float(report_payload, "precision_iou50"),
                    recall_iou50=_optional_float(report_payload, "recall_iou50"),
                    mean_iou=_optional_float(report_payload, "mean_iou"),
                )
            )
        return sorted(items, key=lambda item: item.created_at or "", reverse=True)

    def state(self) -> DashboardState:
        benchmarks = self.benchmarks()
        runs = self.runs()
        return DashboardState(
            store_root=str(self.layout.root),
            benchmark_count=len(benchmarks),
            run_count=len(runs),
            total_benchmark_samples=sum(item.sample_count for item in benchmarks),
            prediction_count=sum(item.prediction_count for item in runs),
            benchmarks=benchmarks,
            runs=runs,
        )

    def run_samples(self, run_id: str, *, offset: int = 0, limit: int = 100) -> list[RunSampleSummary]:
        return self.run_sample_page(run_id, offset=offset, limit=limit).samples

    def run_sample_page(
        self,
        run_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
        label: str | None = None,
        error_filter: str = "all",
    ) -> RunSamplePage:
        payload = self._run_manifest(run_id)
        sample_paths = self._run_sample_json_paths(payload)
        sample_metrics = self._run_sample_metrics_by_index(run_id)
        labels = self._run_label_options(run_id, payload, sample_paths)
        start = max(0, offset)
        page_limit = max(1, limit)
        normalized_label = _normalize_filter_value(label)
        normalized_error = error_filter if error_filter in {"all", "fn", "fp", "missing", "clean"} else "all"
        if normalized_label is None and normalized_error == "all":
            stop = min(len(sample_paths), start + page_limit)
            samples = [
                self._sample_summary(
                    run_id=run_id,
                    run_payload=payload,
                    index=index,
                    json_path=json_path,
                    diagnostics=sample_metrics.get(index),
                )
                for index, json_path in enumerate(sample_paths[start:stop], start=start)
            ]
            return RunSamplePage(
                offset=start,
                limit=page_limit,
                total=len(sample_paths),
                labels=labels,
                samples=samples,
            )
        filtered: list[RunSampleSummary] = []
        for index, json_path in enumerate(sample_paths):
            summary = self._sample_summary(
                run_id=run_id,
                run_payload=payload,
                index=index,
                json_path=json_path,
                diagnostics=sample_metrics.get(index),
            )
            if _run_sample_matches(summary, label=normalized_label, error_filter=normalized_error):
                filtered.append(summary)
        return RunSamplePage(
            offset=start,
            limit=page_limit,
            total=len(filtered),
            labels=labels,
            samples=filtered[start : start + page_limit],
        )

    def run_sample_detail(self, run_id: str, *, sample_index: int) -> RunSampleDetail:
        payload = self._run_manifest(run_id)
        sample_paths = self._run_sample_json_paths(payload)
        if sample_index < 0 or sample_index >= len(sample_paths):
            raise IndexError(f"sample_index={sample_index} is outside run sample range.")
        json_path = sample_paths[sample_index]
        raw_payload = _read_json_or_none(json_path) or {}
        diagnostics = self._run_sample_metrics_by_index(run_id).get(sample_index)
        summary = self._sample_summary(
            run_id=run_id,
            run_payload=payload,
            index=sample_index,
            json_path=json_path,
            raw_payload=raw_payload,
            diagnostics=diagnostics,
        )
        prediction_payload = _read_json_or_none(Path(summary.prediction_path or ""))
        target_labels = run_target_label_set(payload)
        filtered_raw_payload = filter_payload_instances(raw_payload, target_labels)
        filtered_prediction_payload = filter_payload_instances(prediction_payload, target_labels)
        return RunSampleDetail(
            sample=summary,
            gt_instances=filter_instances_by_labels(_raw_instances(raw_payload), target_labels),
            pred_instances=filter_instances_by_labels(
                _prediction_instances(prediction_payload),
                target_labels,
            ),
            raw_payload=filtered_raw_payload,
            prediction_payload=filtered_prediction_payload,
            diagnostics=summary.diagnostics,
        )

    def run_sample_image_path(self, run_id: str, *, sample_index: int) -> Path:
        payload = self._run_manifest(run_id)
        sample_paths = self._run_sample_json_paths(payload)
        if sample_index < 0 or sample_index >= len(sample_paths):
            raise IndexError(f"sample_index={sample_index} is outside run sample range.")
        root = _benchmark_root(payload)
        raw_payload = _read_json_or_none(sample_paths[sample_index]) or {}
        image = _sample_image(raw_payload, sample_paths[sample_index], root)
        return root / image

    def benchmark_samples(
        self,
        benchmark_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[BenchmarkSampleSummary]:
        return self.benchmark_sample_page(benchmark_id, offset=offset, limit=limit).samples

    def benchmark_sample_page(
        self,
        benchmark_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
        label: str | None = None,
    ) -> BenchmarkSamplePage:
        payload = self._benchmark_manifest(benchmark_id)
        sample_paths = self._benchmark_sample_json_paths(payload)
        labels = self._benchmark_label_options(benchmark_id, payload, sample_paths)
        start = max(0, offset)
        page_limit = max(1, limit)
        normalized_label = _normalize_filter_value(label)
        if normalized_label is None:
            stop = min(len(sample_paths), start + page_limit)
            samples = [
                self._benchmark_sample_summary(
                    benchmark_payload=payload,
                    index=index,
                    json_path=json_path,
                )
                for index, json_path in enumerate(sample_paths[start:stop], start=start)
            ]
            return BenchmarkSamplePage(
                offset=start,
                limit=page_limit,
                total=len(sample_paths),
                labels=labels,
                samples=samples,
            )
        filtered: list[BenchmarkSampleSummary] = []
        for index, json_path in enumerate(sample_paths):
            summary = self._benchmark_sample_summary(
                benchmark_payload=payload,
                index=index,
                json_path=json_path,
            )
            if normalized_label in summary.labels:
                filtered.append(summary)
        return BenchmarkSamplePage(
            offset=start,
            limit=page_limit,
            total=len(filtered),
            labels=labels,
            samples=filtered[start : start + page_limit],
        )

    def benchmark_sample_detail(
        self,
        benchmark_id: str,
        *,
        sample_index: int,
    ) -> BenchmarkSampleDetail:
        payload = self._benchmark_manifest(benchmark_id)
        sample_paths = self._benchmark_sample_json_paths(payload)
        if sample_index < 0 or sample_index >= len(sample_paths):
            raise IndexError(f"sample_index={sample_index} is outside benchmark sample range.")
        json_path = sample_paths[sample_index]
        raw_payload = _read_json_or_none(json_path) or {}
        summary = self._benchmark_sample_summary(
            benchmark_payload=payload,
            index=sample_index,
            json_path=json_path,
            raw_payload=raw_payload,
        )
        return BenchmarkSampleDetail(
            sample=summary,
            gt_instances=_raw_instances(raw_payload),
            raw_payload=raw_payload,
        )

    def benchmark_preview_sample(
        self,
        benchmark_id: str | None = None,
    ) -> tuple[str, BenchmarkSampleDetail]:
        benchmark_ids = (
            [benchmark_id]
            if benchmark_id
            else [item.benchmark_id for item in self.benchmarks() if item.sample_count > 0]
        )
        best: tuple[int, str, BenchmarkSampleDetail] | None = None
        for item_id in benchmark_ids:
            payload = self._benchmark_manifest(item_id)
            sample_paths = self._benchmark_sample_json_paths(payload)
            for index, json_path in enumerate(sample_paths):
                raw_payload = _read_json_or_none(json_path) or {}
                instances = _raw_instances(raw_payload)
                if not instances:
                    continue
                score = _preview_instance_score(instances)
                if best is None or score > best[0]:
                    detail = self.benchmark_sample_detail(item_id, sample_index=index)
                    best = (score, item_id, detail)
                    if score >= 32:
                        return item_id, detail
        if best is None:
            raise FileNotFoundError("no benchmark sample with drawable instances was found.")
        return best[1], best[2]

    def benchmark_sample_image_path(self, benchmark_id: str, *, sample_index: int) -> Path:
        payload = self._benchmark_manifest(benchmark_id)
        sample_paths = self._benchmark_sample_json_paths(payload)
        if sample_index < 0 or sample_index >= len(sample_paths):
            raise IndexError(f"sample_index={sample_index} is outside benchmark sample range.")
        root = Path(str(payload.get("root") or ""))
        raw_payload = _read_json_or_none(sample_paths[sample_index]) or {}
        image = _sample_image(raw_payload, sample_paths[sample_index], root)
        return root / image

    def _run_manifest(self, run_id: str) -> dict[str, Any]:
        payload = _read_json_or_none(self.layout.runs_dir / run_id / "run.json")
        if payload is None:
            raise FileNotFoundError(f"run manifest does not exist for run_id={run_id!r}.")
        return payload

    def _benchmark_manifest(self, benchmark_id: str) -> dict[str, Any]:
        payload = _read_json_or_none(self.layout.benchmarks_dir / benchmark_id / "benchmark.json")
        if payload is None:
            raise FileNotFoundError(
                f"benchmark manifest does not exist for benchmark_id={benchmark_id!r}."
            )
        return payload

    def _run_sample_json_paths(self, run_payload: dict[str, Any]) -> list[Path]:
        benchmark = run_payload.get("benchmark") or {}
        root = _benchmark_root(run_payload)
        manifest_path = Path(str(benchmark.get("manifest_path") or ""))
        cache_key = f"{root}::{manifest_path}"
        if cache_key in self._run_sample_paths_cache:
            return self._run_sample_paths_cache[cache_key]
        if not manifest_path.exists():
            raise FileNotFoundError(f"benchmark split manifest does not exist: {manifest_path}")
        paths: list[Path] = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            relative = line.strip()
            if relative:
                paths.append(root / relative)
        self._run_sample_paths_cache[cache_key] = paths
        return paths

    def _benchmark_sample_json_paths(self, benchmark_payload: dict[str, Any]) -> list[Path]:
        root = Path(str(benchmark_payload.get("root") or ""))
        manifest_path = Path(str(benchmark_payload.get("manifest_path") or ""))
        cache_key = f"{root}::{manifest_path}"
        if cache_key in self._benchmark_sample_paths_cache:
            return self._benchmark_sample_paths_cache[cache_key]
        if not manifest_path.exists():
            raise FileNotFoundError(f"benchmark split manifest does not exist: {manifest_path}")
        paths: list[Path] = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            relative = line.strip()
            if relative:
                paths.append(root / relative)
        self._benchmark_sample_paths_cache[cache_key] = paths
        return paths

    def _run_sample_metrics_by_index(self, run_id: str) -> dict[int, dict[str, Any]]:
        if run_id in self._run_metrics_cache:
            return self._run_metrics_cache[run_id]
        payload = _read_json_or_none(self.layout.runs_dir / run_id / "reports" / "metrics.json")
        if payload is None:
            self._run_metrics_cache[run_id] = {}
            return {}
        samples = payload.get("samples") or []
        if not isinstance(samples, list):
            self._run_metrics_cache[run_id] = {}
            return {}
        indexed: dict[int, dict[str, Any]] = {}
        for item in samples:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            indexed[index] = item
        self._run_metrics_cache[run_id] = indexed
        return indexed

    def _run_label_options(
        self,
        run_id: str,
        run_payload: dict[str, Any],
        sample_paths: list[Path],
    ) -> list[str]:
        if run_id in self._run_label_cache:
            return self._run_label_cache[run_id]
        target_labels = run_target_labels(run_payload)
        if target_labels:
            self._run_label_cache[run_id] = target_labels
            return target_labels
        summary = _read_json_or_none(self.layout.runs_dir / run_id / "reports" / "summary.json")
        labels = _labels_from_summary(summary)
        if labels:
            self._run_label_cache[run_id] = labels
            return labels
        report = _read_json_or_none(self.layout.runs_dir / run_id / "reports" / "metrics.json")
        labels = _labels_from_report(report)
        if labels:
            self._run_label_cache[run_id] = labels
            return labels
        labels = self._labels_from_run_samples(run_id, run_payload, sample_paths)
        self._run_label_cache[run_id] = labels
        return labels

    def _labels_from_run_samples(
        self,
        run_id: str,
        run_payload: dict[str, Any],
        sample_paths: list[Path],
    ) -> list[str]:
        labels: set[str] = set()
        for index, json_path in enumerate(sample_paths):
            summary = self._sample_summary(
                run_id=run_id,
                run_payload=run_payload,
                index=index,
                json_path=json_path,
            )
            labels.update(summary.labels)
        return sorted(labels)

    def _benchmark_label_options(
        self,
        benchmark_id: str,
        benchmark_payload: dict[str, Any],
        sample_paths: list[Path],
    ) -> list[str]:
        if benchmark_id in self._benchmark_label_cache:
            return self._benchmark_label_cache[benchmark_id]
        labels = _labels_from_summary(benchmark_payload)
        if labels:
            self._benchmark_label_cache[benchmark_id] = labels
            return labels
        labels: set[str] = set()
        for index, json_path in enumerate(sample_paths):
            summary = self._benchmark_sample_summary(
                benchmark_payload=benchmark_payload,
                index=index,
                json_path=json_path,
            )
            labels.update(summary.labels)
        result = sorted(labels)
        self._benchmark_label_cache[benchmark_id] = result
        return result

    def _sample_summary(
        self,
        *,
        run_id: str,
        run_payload: dict[str, Any],
        index: int,
        json_path: Path,
        raw_payload: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> RunSampleSummary:
        root = _benchmark_root(run_payload)
        raw_payload = raw_payload if raw_payload is not None else (_read_json_or_none(json_path) or {})
        image = _sample_image(raw_payload, json_path, root)
        prediction_path = RunArtifacts(self.layout.root, run_id).prediction_path(image)
        prediction_payload = _read_json_or_none(prediction_path)
        target_labels = run_target_label_set(run_payload)
        all_gt_instances = _raw_instances(raw_payload)
        all_pred_instances = _prediction_instances(prediction_payload)
        gt_instances = filter_instances_by_labels(all_gt_instances, target_labels)
        pred_instances = filter_instances_by_labels(
            all_pred_instances,
            target_labels,
        )
        scoped_diagnostics = scope_sample_diagnostics(
            diagnostics,
            gt_instances=all_gt_instances,
            pred_instances=all_pred_instances,
            labels=target_labels,
        )
        labels = sorted(
            {str(item.get("label") or "") for item in gt_instances + pred_instances if item.get("label")}
        )
        return RunSampleSummary(
            index=index,
            image=image,
            json_path=str(json_path),
            image_width=_optional_int(raw_payload, "image_width"),
            image_height=_optional_int(raw_payload, "image_height"),
            gt_instance_count=len(gt_instances),
            pred_instance_count=len(pred_instances),
            labels=labels,
            has_prediction=prediction_payload is not None,
            prediction_path=str(prediction_path) if prediction_payload is not None else None,
            diagnostics=scoped_diagnostics,
        )

    def _benchmark_sample_summary(
        self,
        *,
        benchmark_payload: dict[str, Any],
        index: int,
        json_path: Path,
        raw_payload: dict[str, Any] | None = None,
    ) -> BenchmarkSampleSummary:
        root = Path(str(benchmark_payload.get("root") or ""))
        raw_payload = raw_payload if raw_payload is not None else (_read_json_or_none(json_path) or {})
        image = _sample_image(raw_payload, json_path, root)
        instances = _raw_instances(raw_payload)
        labels = sorted({str(item.get("label") or "") for item in instances if item.get("label")})
        return BenchmarkSampleSummary(
            index=index,
            image=image,
            json_path=str(json_path),
            image_width=_optional_int(raw_payload, "image_width"),
            image_height=_optional_int(raw_payload, "image_height"),
            instance_count=len(instances),
            labels=labels,
        )


def _optional_float(payload: dict[str, Any] | None, key: str) -> float | None:
    if payload is None or payload.get(key) is None:
        return None
    try:
        return float(payload[key])
    except (TypeError, ValueError):
        return None


def _optional_int(payload: dict[str, Any] | None, key: str) -> int | None:
    if payload is None or payload.get(key) is None:
        return None
    try:
        return int(payload[key])
    except (TypeError, ValueError):
        return None


def _labels_from_report(payload: dict[str, Any] | None) -> list[str]:
    if payload is None:
        return []
    labels = payload.get("labels") or []
    if not isinstance(labels, list):
        return []
    values: set[str] = set()
    for item in labels:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if label:
            values.add(label)
    return sorted(values)


def _labels_from_summary(payload: dict[str, Any] | None) -> list[str]:
    if payload is None:
        return []
    labels = payload.get("labels") or []
    if not isinstance(labels, list):
        return []
    values = {str(item).strip() for item in labels if str(item).strip()}
    return sorted(values)


def _normalize_filter_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized == "all":
        return None
    return normalized


def _run_sample_matches(
    sample: RunSampleSummary,
    *,
    label: str | None,
    error_filter: str,
) -> bool:
    if label is not None and label not in sample.labels:
        return False
    if error_filter == "all":
        return True
    if error_filter == "missing":
        return not sample.has_prediction
    diagnostics = sample.diagnostics
    if not diagnostics:
        return False
    false_negative_count = int(diagnostics.get("false_negative_count") or 0)
    false_positive_count = int(diagnostics.get("false_positive_count") or 0)
    if error_filter == "fn":
        return false_negative_count > 0
    if error_filter == "fp":
        return false_positive_count > 0
    if error_filter == "clean":
        return sample.has_prediction and false_negative_count == 0 and false_positive_count == 0
    return True


def _benchmark_root(run_payload: dict[str, Any]) -> Path:
    benchmark = run_payload.get("benchmark") or {}
    return Path(str(benchmark.get("root") or ""))


def _sample_image(raw_payload: dict[str, Any], json_path: Path, root: Path) -> str:
    return sample_image_string(json_path, raw_payload, root=root)


def _raw_instances(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    instances = payload.get("instances") or []
    if not isinstance(instances, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in instances:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        bbox = item.get("bbox")
        if not label or not _valid_box_like(bbox):
            continue
        normalized.append(
            {
                "label": label,
                "bbox": [float(value) for value in bbox],
                "linestrip": _points_or_none(item.get("linestrip")),
                "keypoints": _points_or_none(item.get("keypoints")),
                "extra": dict(item.get("extra") or {}),
            }
        )
    return normalized


def _prediction_instances(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    instances = payload.get("instances") or []
    if not isinstance(instances, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in instances:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        bbox = item.get("bbox")
        if not label or not _valid_box_like(bbox):
            continue
        normalized.append(
            {
                "label": label,
                "bbox": [float(value) for value in bbox],
                "linestrip": _points_or_none(item.get("linestrip")),
                "keypoints": _points_or_none(item.get("keypoints")),
                "score": item.get("score"),
                "extra": dict(item.get("extra") or {}),
            }
        )
    return normalized


def _preview_instance_score(instances: list[dict[str, Any]]) -> int:
    labels = {str(item.get("label") or "") for item in instances if item.get("label")}
    has_linestrip = any(item.get("linestrip") for item in instances)
    has_keypoints = any(item.get("keypoints") for item in instances)
    return len(instances) + len(labels) * 2 + (20 if has_linestrip else 0) + (10 if has_keypoints else 0)


def _valid_box_like(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def _points_or_none(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list):
        return None
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            continue
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            continue
    return points or None
