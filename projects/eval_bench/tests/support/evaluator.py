from __future__ import annotations

from pathlib import Path

from eval_bench.artifacts import RunArtifacts
from eval_bench.schema import (
    BenchmarkRef,
    EvalRunManifest,
    EvalSpec,
    ModelRef,
    PromptRef,
)


def write_eval_run(
    store_root: Path,
    *,
    task: str,
    run_id: str | None = None,
    benchmark_id: str = "bench1",
    benchmark_root: Path | None = None,
    split: str = "val",
    split_path: Path | None = None,
    manifest_entries: str = "part1/json/a.json\n",
    model_id: str = "model-a",
    prompt_id: str | None = None,
    target_labels: list[str] | None = None,
) -> RunArtifacts:
    resolved_run_id = run_id or f"run_{task}"
    resolved_split_path = (
        split_path
        or store_root
        / "benchmarks"
        / benchmark_id
        / "splits"
        / f"{split}.txt"
    )
    resolved_split_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_split_path.write_text(manifest_entries, encoding="utf-8")
    resolved_root = benchmark_root or store_root / "benchmarks" / benchmark_id / "data"
    artifacts = RunArtifacts(store_root, resolved_run_id)
    artifacts.write_manifest(
        EvalRunManifest(
            run_id=resolved_run_id,
            model=ModelRef(model_id=model_id, path=f"outputs/{model_id}/best"),
            benchmark=BenchmarkRef(
                benchmark_id=benchmark_id,
                root=str(resolved_root),
                split=split,
                tasks=["detection", "keypoint"],
                manifest_path=str(resolved_split_path),
            ),
            spec=EvalSpec(
                spec_id=f"{task}.default",
                task=task,  # type: ignore[arg-type]
                prompt=PromptRef(prompt_id=prompt_id or f"{task}.prompt"),
                target_labels=target_labels or [],
            ),
        )
    )
    return artifacts
