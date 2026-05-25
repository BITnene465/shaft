from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import DEFAULT_STORE_ROOT, RunArtifacts, StoreLayout, atomic_write_json
from .sample_paths import sample_image_string
from .sample_scope import (
    filter_instances_by_labels,
    filter_payload_instances,
    run_target_label_set,
    run_target_labels,
    scope_sample_diagnostics,
)
from .schema import utc_now_iso


MAX_RUN_NOTE_LENGTH = 20_000
DEFAULT_RANK_SORT_BY = "f1_iou50"
RANK_PRIMARY_METRIC_LABEL = "F1@.50"
WEIGHTED_RANK_SORT_BY = "weighted_score"


@dataclass(frozen=True)
class RankSchemeTerm:
    benchmark_id: str
    metric: str
    weight: float
    missing: str = "drop"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankScheme:
    name: str
    terms: list[RankSchemeTerm]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "terms": [term.to_dict() for term in self.terms]}


@dataclass(frozen=True)
class WeightedRankScore:
    score: float
    components: list[dict[str, Any]]


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
    labels: list[str]
    layers: list[str]
    split: str
    sample_count: int
    root: str
    manifest_path: str
    created_at: str | None = None
    source_manifest_path: str | None = None


@dataclass(frozen=True)
class BenchmarkListPage:
    offset: int
    limit: int
    total: int
    filters: dict[str, str]
    benchmarks: list[BenchmarkSummary]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    status: str
    benchmark_id: str
    tasks: list[str]
    spec_task: str
    target_labels: list[str]
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
    note: str = ""
    note_updated_at: str | None = None
    note_max_length: int = MAX_RUN_NOTE_LENGTH
    precision_iou50: float | None = None
    recall_iou50: float | None = None
    mean_iou: float | None = None


@dataclass(frozen=True)
class RunListPage:
    offset: int
    limit: int
    total: int
    filters: dict[str, str]
    runs: list[RunSummary]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunNote:
    run_id: str
    note: str
    updated_at: str | None
    path: str
    max_length: int = MAX_RUN_NOTE_LENGTH

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankBoardEntry:
    rank: int
    f1_iou50: float | None
    run_id: str
    score: float | None
    status: str
    benchmark_id: str
    task: str
    target_labels: list[str]
    model_id: str
    prompt_id: str
    metric_profile: str
    prediction_count: int
    precision_iou50: float | None
    recall_iou50: float | None
    mean_iou: float | None
    created_at: str | None
    note: str
    score_components: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RankBoard:
    offset: int
    limit: int
    total: int
    evaluated_count: int
    filters: dict[str, str]
    primary_metric: str
    primary_metric_label: str
    sort_by: str
    sort_order: str
    score_formula: str
    rank_scheme: dict[str, Any] | None
    facets: dict[str, list[dict[str, Any]]]
    entries: list[RankBoardEntry]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
                    labels=self._benchmark_summary_labels(
                        str(payload.get("benchmark_id") or manifest_path.parent.name),
                        payload,
                    ),
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

    def _benchmark_summary_labels(self, benchmark_id: str, payload: dict[str, Any]) -> list[str]:
        labels = _labels_from_summary(payload)
        if labels:
            return labels
        try:
            return self._benchmark_label_options(
                benchmark_id,
                payload,
                self._benchmark_sample_json_paths(payload),
            )
        except (FileNotFoundError, OSError):
            return []

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
            run_id = str(payload.get("run_id") or run_dir.name)
            note = self._run_note_for_payload(run_id, payload)
            prompt_metadata = dict(prompt.get("metadata") or {})
            target_labels = _string_list(
                spec.get("target_labels")
                or (report_payload or {}).get("target_labels")
                or prompt_metadata.get("target_labels")
                or []
            )
            items.append(
                RunSummary(
                    run_id=run_id,
                    status=str(payload.get("status") or "unknown"),
                    benchmark_id=str(
                        benchmark.get("benchmark_id")
                        or benchmark.get("dataset_id")
                        or benchmark.get("id")
                        or ""
                    ),
                    tasks=[str(item) for item in benchmark.get("tasks") or []],
                    spec_task=str(spec.get("task") or ""),
                    target_labels=target_labels,
                    model_id=str(model.get("model_id") or ""),
                    model_path=str(model.get("path") or ""),
                    prompt_id=str(prompt.get("prompt_id") or ""),
                    prompt_path=prompt.get("path"),
                    prompt_hash=prompt.get("text_hash"),
                    prompt_metadata=prompt_metadata,
                    parser=str(spec.get("parser") or ""),
                    metric_profile=str(spec.get("metric_profile") or ""),
                    visualization_profile=str(spec.get("visualization_profile") or ""),
                    inference=dict(inference) if isinstance(inference, dict) else {},
                    created_at=payload.get("created_at"),
                    prediction_count=prediction_count,
                    report_count=report_count,
                    manifest_path=str(manifest_path),
                    report_path=str(report_path) if report_path.exists() else None,
                    note=note.note,
                    note_updated_at=note.updated_at,
                    note_max_length=MAX_RUN_NOTE_LENGTH,
                    precision_iou50=_optional_float(report_payload, "precision_iou50"),
                    recall_iou50=_optional_float(report_payload, "recall_iou50"),
                    mean_iou=_optional_float(report_payload, "mean_iou"),
                )
            )
        return sorted(items, key=lambda item: item.created_at or "", reverse=True)

    def benchmark_page(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        task: str | None = None,
        layer: str | None = None,
        split: str | None = None,
        query: str | None = None,
    ) -> BenchmarkListPage:
        filters = {
            "task": _normalize_filter_value(task) or "",
            "layer": _normalize_filter_value(layer) or "",
            "split": _normalize_filter_value(split) or "",
            "query": (query or "").strip(),
        }
        query_text = filters["query"].lower()
        items = [
            benchmark
            for benchmark in self.benchmarks()
            if _benchmark_matches_filters(
                benchmark,
                task=filters["task"],
                layer=filters["layer"],
                split=filters["split"],
                query=query_text,
            )
        ]
        start, page_limit = _page_bounds(offset=offset, limit=limit)
        return BenchmarkListPage(
            offset=start,
            limit=page_limit,
            total=len(items),
            filters=filters,
            benchmarks=items[start : start + page_limit],
        )

    def run_page(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        task: str | None = None,
        benchmark_id: str | None = None,
        status: str | None = None,
        label: str | None = None,
        model_id: str | None = None,
        prompt_id: str | None = None,
        metric_profile: str | None = None,
        query: str | None = None,
    ) -> RunListPage:
        filters = {
            "task": _normalize_filter_value(task) or "",
            "benchmark_id": _normalize_filter_value(benchmark_id) or "",
            "status": _normalize_filter_value(status) or "",
            "label": _normalize_filter_value(label) or "",
            "model_id": _normalize_filter_value(model_id) or "",
            "prompt_id": _normalize_filter_value(prompt_id) or "",
            "metric_profile": _normalize_filter_value(metric_profile) or "",
            "query": (query or "").strip(),
        }
        query_text = filters["query"].lower()
        items = [
            run
            for run in self.runs()
            if _run_matches_filters(
                run,
                task=filters["task"],
                benchmark_id=filters["benchmark_id"],
                status=filters["status"],
                label=filters["label"],
                model_id=filters["model_id"],
                prompt_id=filters["prompt_id"],
                metric_profile=filters["metric_profile"],
                query=query_text,
            )
        ]
        start, page_limit = _page_bounds(offset=offset, limit=limit)
        return RunListPage(
            offset=start,
            limit=page_limit,
            total=len(items),
            filters=filters,
            runs=items[start : start + page_limit],
        )

    def run_note(self, run_id: str) -> RunNote:
        payload = self._run_manifest(run_id)
        return self._run_note_for_payload(run_id, payload)

    def update_run_note(self, run_id: str, note: str) -> RunNote:
        self._run_manifest(run_id)
        if not isinstance(note, str):
            raise ValueError("note must be a string.")
        if len(note) > MAX_RUN_NOTE_LENGTH:
            raise ValueError(f"note must be at most {MAX_RUN_NOTE_LENGTH} characters.")
        artifacts = RunArtifacts(self.layout.root, run_id)
        artifacts.ensure()
        updated = RunNote(
            run_id=run_id,
            note=note,
            updated_at=utc_now_iso(),
            path=str(artifacts.note_path),
        )
        atomic_write_json(artifacts.note_path, updated.to_dict())
        return updated

    def archive_run(self, run_id: str) -> dict[str, Any]:
        payload = self._run_manifest(run_id)
        manifest_path = self.layout.runs_dir / run_id / "run.json"
        payload["status"] = "archived"
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["archived_at"] = utc_now_iso()
        payload["metadata"] = metadata
        atomic_write_json(manifest_path, payload)
        return {
            "run_id": run_id,
            "status": "archived",
            "manifest_path": str(manifest_path),
        }

    def delete_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.layout.runs_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"run does not exist: {run_id}")
        trash_path = self.layout.move_to_trash(run_dir, category="runs")
        return {
            "run_id": run_id,
            "deleted": True,
            "trash_path": str(trash_path) if trash_path is not None else None,
        }

    def rank_board(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        task: str | None = None,
        benchmark_id: str | None = None,
        status: str | None = None,
        label: str | None = None,
        model_id: str | None = None,
        prompt_id: str | None = None,
        metric_profile: str | None = None,
        min_score: float | None = None,
        sort_by: str = DEFAULT_RANK_SORT_BY,
        sort_order: str = "desc",
        query: str | None = None,
        rank_scheme: Mapping[str, Any] | str | None = None,
    ) -> RankBoard:
        resolved_rank_scheme = _normalize_rank_scheme(rank_scheme)
        resolved_sort_by = _normalize_rank_sort_by(
            sort_by,
            weighted=resolved_rank_scheme is not None,
        )
        resolved_sort_order = "asc" if str(sort_order).lower() == "asc" else "desc"
        filters = {
            "task": _normalize_filter_value(task) or "",
            "benchmark_id": _normalize_filter_value(benchmark_id) or "",
            "status": _normalize_filter_value(status) or "",
            "label": _normalize_filter_value(label) or "",
            "model_id": _normalize_filter_value(model_id) or "",
            "prompt_id": _normalize_filter_value(prompt_id) or "",
            "metric_profile": _normalize_filter_value(metric_profile) or "",
            "min_score": "" if min_score is None else str(float(min_score)),
            "query": (query or "").strip(),
            "rank_scheme": resolved_rank_scheme.name if resolved_rank_scheme else "",
        }
        query_text = filters["query"].lower()
        entries: list[RankBoardEntry] = []
        for run in self.runs():
            if filters["task"] and run.spec_task != filters["task"]:
                continue
            if filters["benchmark_id"] and run.benchmark_id != filters["benchmark_id"]:
                continue
            if filters["status"] and run.status != filters["status"]:
                continue
            if filters["label"] and filters["label"] not in run.target_labels:
                continue
            if filters["model_id"] and run.model_id != filters["model_id"]:
                continue
            if filters["prompt_id"] and run.prompt_id != filters["prompt_id"]:
                continue
            if filters["metric_profile"] and run.metric_profile != filters["metric_profile"]:
                continue
            if query_text and not _rank_query_matches(run, query_text):
                continue
            f1_iou50 = _rank_f1_iou50(run)
            weighted = (
                _rank_weighted_score(run, resolved_rank_scheme)
                if resolved_rank_scheme is not None
                else None
            )
            if resolved_rank_scheme is not None and weighted is None:
                continue
            score = weighted.score if weighted is not None else f1_iou50
            score_components = weighted.components if weighted is not None else []
            min_score_value = (
                score
                if resolved_sort_by == "weighted_score"
                else _rank_run_metric_value(run, resolved_sort_by)
            )
            if not isinstance(min_score_value, (int, float)):
                min_score_value = score
            if min_score is not None and (
                min_score_value is None or min_score_value < float(min_score)
            ):
                continue
            entries.append(
                RankBoardEntry(
                    rank=0,
                    f1_iou50=f1_iou50,
                    run_id=run.run_id,
                    score=score,
                    status=run.status,
                    benchmark_id=run.benchmark_id,
                    task=run.spec_task,
                    target_labels=run.target_labels,
                    model_id=run.model_id,
                    prompt_id=run.prompt_id,
                    metric_profile=run.metric_profile,
                    prediction_count=run.prediction_count,
                    precision_iou50=run.precision_iou50,
                    recall_iou50=run.recall_iou50,
                    mean_iou=run.mean_iou,
                    created_at=run.created_at,
                    note=run.note,
                    score_components=score_components,
                )
            )
        facets = _rank_facets(entries)
        evaluated_count = sum(1 for entry in entries if entry.f1_iou50 is not None)
        ranked = _sort_rank_entries(
            entries,
            sort_by=resolved_sort_by,
            sort_order=resolved_sort_order,
        )
        ranked = [
            RankBoardEntry(**{**asdict(entry), "rank": index})
            for index, entry in enumerate(ranked, start=1)
        ]
        start = max(0, int(offset))
        page_limit = max(1, int(limit))
        return RankBoard(
            offset=start,
            limit=page_limit,
            total=len(ranked),
            evaluated_count=evaluated_count,
            filters=filters,
            primary_metric="weighted_score" if resolved_rank_scheme else DEFAULT_RANK_SORT_BY,
            primary_metric_label=(
                resolved_rank_scheme.name if resolved_rank_scheme else RANK_PRIMARY_METRIC_LABEL
            ),
            sort_by=resolved_sort_by,
            sort_order=resolved_sort_order,
            score_formula=(
                _rank_scheme_formula(resolved_rank_scheme)
                if resolved_rank_scheme
                else RANK_PRIMARY_METRIC_LABEL
            ),
            rank_scheme=resolved_rank_scheme.to_dict() if resolved_rank_scheme else None,
            facets=facets,
            entries=ranked[start : start + page_limit],
        )

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

    def _run_note_for_payload(self, run_id: str, run_payload: dict[str, Any]) -> RunNote:
        note_path = RunArtifacts(self.layout.root, run_id).note_path
        payload = _read_json_or_none(note_path)
        if payload is not None:
            return RunNote(
                run_id=run_id,
                note=str(payload.get("note") or ""),
                updated_at=payload.get("updated_at"),
                path=str(note_path),
            )
        metadata = run_payload.get("metadata")
        if isinstance(metadata, dict):
            legacy_note = metadata.get("note")
            if isinstance(legacy_note, str):
                return RunNote(
                    run_id=run_id,
                    note=legacy_note,
                    updated_at=metadata.get("note_updated_at"),
                    path=str(note_path),
                )
        return RunNote(run_id=run_id, note="", updated_at=None, path=str(note_path))

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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_filter_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized == "all":
        return None
    return normalized


def _page_bounds(*, offset: int, limit: int) -> tuple[int, int]:
    return max(0, int(offset)), max(1, int(limit))


def _benchmark_matches_filters(
    benchmark: BenchmarkSummary,
    *,
    task: str,
    layer: str,
    split: str,
    query: str,
) -> bool:
    if task and task not in benchmark.tasks:
        return False
    if layer and layer not in benchmark.layers:
        return False
    if split and split != benchmark.split:
        return False
    if query and not _query_matches_fields(
        query,
        [
            benchmark.benchmark_id,
            benchmark.split,
            benchmark.root,
            benchmark.manifest_path,
            benchmark.source_manifest_path,
            " ".join(benchmark.tasks),
            " ".join(benchmark.layers),
        ],
    ):
        return False
    return True


def _run_matches_filters(
    run: RunSummary,
    *,
    task: str,
    benchmark_id: str,
    status: str,
    label: str,
    model_id: str,
    prompt_id: str,
    metric_profile: str,
    query: str,
) -> bool:
    if task and run.spec_task != task:
        return False
    if benchmark_id and run.benchmark_id != benchmark_id:
        return False
    if status and run.status != status:
        return False
    if label and label not in run.target_labels:
        return False
    if model_id and run.model_id != model_id:
        return False
    if prompt_id and run.prompt_id != prompt_id:
        return False
    if metric_profile and run.metric_profile != metric_profile:
        return False
    if query and not _run_query_matches(run, query):
        return False
    return True


def _query_matches_fields(query: str, fields: list[object]) -> bool:
    return any(query in str(field or "").lower() for field in fields)


def _run_query_matches(run: RunSummary, query: str) -> bool:
    return _query_matches_fields(
        query,
        [
            run.run_id,
            run.status,
            run.benchmark_id,
            run.spec_task,
            run.model_id,
            run.model_path,
            run.prompt_id,
            run.metric_profile,
            run.note,
            " ".join(run.target_labels),
        ],
    )


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


def _rank_f1_iou50(run: RunSummary) -> float | None:
    precision = run.precision_iou50
    recall = run.recall_iou50
    if precision is None or recall is None:
        return None
    denominator = precision + recall
    if denominator <= 0:
        return 0.0
    return (2 * precision * recall) / denominator


def _normalize_rank_scheme(value: Mapping[str, Any] | str | None) -> RankScheme | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("rank_scheme must be a JSON object.") from exc
    else:
        payload = dict(value)
    if not isinstance(payload, dict):
        raise ValueError("rank_scheme must be a JSON object.")
    name = str(payload.get("name") or payload.get("rank_profile") or "weighted_score").strip()
    terms_payload = payload.get("terms") or payload.get("weights")
    if not isinstance(terms_payload, list) or not terms_payload:
        raise ValueError("rank_scheme.terms must be a non-empty list.")
    terms: list[RankSchemeTerm] = []
    for index, item in enumerate(terms_payload):
        if not isinstance(item, dict):
            raise ValueError(f"rank_scheme.terms[{index}] must be a JSON object.")
        benchmark_id = str(item.get("benchmark_id") or "").strip()
        metric = _normalize_rank_metric(str(item.get("metric") or ""))
        missing = str(item.get("missing") or "drop").strip().lower()
        if not benchmark_id:
            raise ValueError(f"rank_scheme.terms[{index}].benchmark_id is required.")
        if metric not in _allowed_rank_metrics():
            raise ValueError(f"rank_scheme.terms[{index}].metric is not supported: {metric}")
        if missing not in {"drop", "skip", "zero"}:
            raise ValueError("rank_scheme term missing must be one of: drop, skip, zero.")
        try:
            weight = float(item.get("weight"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rank_scheme.terms[{index}].weight must be numeric.") from exc
        if weight <= 0:
            raise ValueError(f"rank_scheme.terms[{index}].weight must be positive.")
        terms.append(
            RankSchemeTerm(
                benchmark_id=benchmark_id,
                metric=metric,
                weight=weight,
                missing=missing,
            )
        )
    return RankScheme(name=name or "weighted_score", terms=terms)


def _allowed_rank_metrics() -> set[str]:
    return {
        "f1_iou50",
        "precision_iou50",
        "recall_iou50",
        "mean_iou",
        "prediction_count",
    }


def _normalize_rank_metric(value: str) -> str:
    normalized = str(value or DEFAULT_RANK_SORT_BY).strip()
    if normalized == "score":
        return DEFAULT_RANK_SORT_BY
    return normalized


def _normalize_rank_sort_by(value: str, *, weighted: bool = False) -> str:
    normalized = str(value or DEFAULT_RANK_SORT_BY).strip()
    if normalized == "score":
        return WEIGHTED_RANK_SORT_BY if weighted else DEFAULT_RANK_SORT_BY
    if weighted and normalized == DEFAULT_RANK_SORT_BY:
        return WEIGHTED_RANK_SORT_BY
    allowed = {
        "f1_iou50",
        "precision_iou50",
        "recall_iou50",
        "mean_iou",
        "prediction_count",
        "created_at",
        "run_id",
        WEIGHTED_RANK_SORT_BY,
    }
    if normalized in allowed:
        return normalized
    return WEIGHTED_RANK_SORT_BY if weighted else DEFAULT_RANK_SORT_BY


def _sort_rank_entries(
    entries: list[RankBoardEntry],
    *,
    sort_by: str,
    sort_order: str,
) -> list[RankBoardEntry]:
    valued = [entry for entry in entries if _rank_sort_value(entry, sort_by) is not None]
    missing = [entry for entry in entries if _rank_sort_value(entry, sort_by) is None]
    valued.sort(
        key=lambda entry: _rank_sort_value(entry, sort_by),  # type: ignore[arg-type]
        reverse=sort_order == "desc",
    )
    missing.sort(key=lambda entry: entry.run_id)
    return valued + missing


def _rank_sort_value(entry: RankBoardEntry, sort_by: str) -> float | int | str | None:
    if sort_by == WEIGHTED_RANK_SORT_BY:
        return entry.score
    if sort_by == "f1_iou50":
        return entry.f1_iou50
    if sort_by == "precision_iou50":
        return entry.precision_iou50
    if sort_by == "recall_iou50":
        return entry.recall_iou50
    if sort_by == "mean_iou":
        return entry.mean_iou
    if sort_by == "prediction_count":
        return entry.prediction_count
    if sort_by == "created_at":
        return entry.created_at or None
    if sort_by == "run_id":
        return entry.run_id or None
    return entry.f1_iou50


def _rank_run_metric_value(run: RunSummary, sort_by: str) -> float | int | str | None:
    if sort_by == WEIGHTED_RANK_SORT_BY:
        return None
    if sort_by == "f1_iou50":
        return _rank_f1_iou50(run)
    if sort_by == "precision_iou50":
        return run.precision_iou50
    if sort_by == "recall_iou50":
        return run.recall_iou50
    if sort_by == "mean_iou":
        return run.mean_iou
    if sort_by == "prediction_count":
        return run.prediction_count
    if sort_by == "created_at":
        return run.created_at or None
    if sort_by == "run_id":
        return run.run_id or None
    return _rank_f1_iou50(run)


def _rank_weighted_score(run: RunSummary, scheme: RankScheme | None) -> WeightedRankScore | None:
    if scheme is None:
        return None
    matched_terms = [term for term in scheme.terms if term.benchmark_id == run.benchmark_id]
    if not matched_terms:
        return None
    total_weight = 0.0
    weighted_sum = 0.0
    components: list[dict[str, Any]] = []
    for term in matched_terms:
        value = _rank_run_metric_value(run, term.metric)
        numeric_value = float(value) if isinstance(value, (int, float)) else None
        if numeric_value is None:
            components.append(
                {
                    "benchmark_id": term.benchmark_id,
                    "metric": term.metric,
                    "weight": term.weight,
                    "missing": term.missing,
                    "value": None,
                    "contribution": None,
                }
            )
            if term.missing == "drop":
                return None
            if term.missing == "skip":
                continue
            numeric_value = 0.0
        contribution = numeric_value * term.weight
        components.append(
            {
                "benchmark_id": term.benchmark_id,
                "metric": term.metric,
                "weight": term.weight,
                "missing": term.missing,
                "value": numeric_value,
                "contribution": contribution,
            }
        )
        weighted_sum += contribution
        total_weight += term.weight
    if total_weight <= 0:
        return None
    return WeightedRankScore(score=weighted_sum / total_weight, components=components)


def _rank_scheme_formula(scheme: RankScheme) -> str:
    parts = [
        f"{term.weight:g}*{term.benchmark_id}.{term.metric}[missing={term.missing}]"
        for term in scheme.terms
    ]
    return f"{scheme.name}: " + " + ".join(parts)


def _rank_facets(entries: list[RankBoardEntry]) -> dict[str, list[dict[str, Any]]]:
    return {
        "tasks": _rank_facet(entries, lambda entry: [entry.task or "unknown"]),
        "benchmarks": _rank_facet(entries, lambda entry: [entry.benchmark_id or "unknown"]),
        "statuses": _rank_facet(entries, lambda entry: [entry.status or "unknown"]),
        "labels": _rank_facet(
            entries,
            lambda entry: entry.target_labels if entry.target_labels else ["unscoped"],
        ),
        "models": _rank_facet(entries, lambda entry: [entry.model_id or "unknown"]),
        "prompts": _rank_facet(entries, lambda entry: [entry.prompt_id or "unknown"]),
        "metric_profiles": _rank_facet(
            entries,
            lambda entry: [entry.metric_profile or "unknown"],
        ),
    }


def _rank_facet(
    entries: list[RankBoardEntry],
    values: Callable[[RankBoardEntry], list[str]],
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for entry in entries:
        for value in values(entry):
            key = str(value or "unknown")
            counts[key] = counts.get(key, 0) + 1
    return [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _rank_query_matches(run: RunSummary, query: str) -> bool:
    return _run_query_matches(run, query)


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
