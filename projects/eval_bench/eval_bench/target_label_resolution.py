from __future__ import annotations

from typing import Any

from .database import EvalBenchDatabase
from .label_policy import target_label_resolution_payload
from .store import EvalBenchStore


def resolve_target_label_scope(
    *,
    database: EvalBenchDatabase,
    store: EvalBenchStore,
    benchmark_id: str | None = None,
    task: str | None = None,
    prompt_id: str | None = None,
    explicit: Any = None,
) -> dict[str, Any]:
    prompt_id = _optional_string(prompt_id)
    benchmark_id = _optional_string(benchmark_id)
    resolved_task = _optional_string(task)
    prompt_metadata: dict[str, Any] = {}
    warnings: list[str] = []
    if prompt_id:
        record = database.get_prompt_template(prompt_id)
        if record is None:
            warnings.append(f"prompt template does not exist: {prompt_id}")
        else:
            prompt_metadata = dict(record.metadata)
            resolved_task = resolved_task or record.task
    benchmark_labels: list[str] = []
    if benchmark_id:
        benchmark = next(
            (item for item in store.benchmarks() if item.benchmark_id == benchmark_id),
            None,
        )
        if benchmark is None:
            raise FileNotFoundError(f"benchmark does not exist: {benchmark_id}")
        benchmark_labels = benchmark.labels
        if resolved_task and benchmark.tasks and resolved_task not in benchmark.tasks:
            warnings.append(
                f"task={resolved_task} is not advertised by benchmark {benchmark_id}: "
                f"{benchmark.tasks}"
            )
    payload = target_label_resolution_payload(
        task=resolved_task,
        prompt_id=prompt_id,
        explicit=explicit,
        prompt_metadata=prompt_metadata,
        benchmark_id=benchmark_id,
        benchmark_labels=benchmark_labels,
    )
    payload["warnings"] = [*payload["warnings"], *warnings]
    return payload


def _optional_string(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None
