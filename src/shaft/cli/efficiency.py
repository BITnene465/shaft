from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from shaft.observability import (
    TRAINING_EFFICIENCY_FILENAME,
    TRAINING_EFFICIENCY_SCHEMA_VERSION,
    ShaftTrainingEfficiencyContract,
)


def _load_summary(path: str | Path) -> tuple[str, dict[str, Any]]:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / TRAINING_EFFICIENCY_FILENAME
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or int(payload.get("schema_version", -1)) != TRAINING_EFFICIENCY_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported training-efficiency summary: {resolved}")
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"Training-efficiency summary has no aggregate: {resolved}")
    return resolved.parent.name or resolved.stem, payload


def _validate_comparison_contracts(
    summaries: Sequence[tuple[str, dict[str, Any]]],
    *,
    allow_workload_variation: bool = False,
) -> None:
    if len(summaries) < 2:
        return
    parsed: list[tuple[str, ShaftTrainingEfficiencyContract]] = []
    for name, payload in summaries:
        raw_contract = payload.get("contract")
        if not isinstance(raw_contract, dict):
            raise ValueError(
                f"Run {name!r} has no typed training-efficiency contract; "
                "use --allow-incompatible only for an explicitly non-fair comparison."
            )
        parsed.append((name, ShaftTrainingEfficiencyContract.from_dict(dict(raw_contract))))

    baseline_name, baseline = parsed[0]
    baseline_identity = baseline.comparison_identity()
    mismatches: list[str] = []
    for name, contract in parsed[1:]:
        identity = contract.comparison_identity()
        changed = [
            field_name
            for field_name in ShaftTrainingEfficiencyContract.COMPARISON_IDENTITY_FIELDS
            if identity[field_name] != baseline_identity[field_name]
        ]
        if changed:
            mismatches.append(f"{name} vs {baseline_name}: {', '.join(changed)}")

    baseline_span = (
        int(summaries[0][1].get("initial_global_step", -1)),
        int(summaries[0][1].get("final_global_step", -1)),
    )
    for name, payload in summaries[1:]:
        span = (
            int(payload.get("initial_global_step", -1)),
            int(payload.get("final_global_step", -1)),
        )
        if span != baseline_span:
            mismatches.append(f"{name} vs {baseline_name}: committed_step_span")

    exact_workload_fields = [
        "update_applied_steps",
        "microbatches",
        "physical_packs",
        "weighted_supervision_coverage_microbatches",
        "vision_coverage_batches",
    ]
    if not allow_workload_variation:
        exact_workload_fields.extend(
            (
                "logical_segments",
                "useful_tokens",
                "supervised_tokens",
                "vision_patches",
                "sequence_length_sum",
                "sequence_length_square_sum",
            )
        )
    baseline_aggregate = summaries[0][1]["aggregate"]
    for name, payload in summaries[1:]:
        aggregate = payload["aggregate"]
        changed = [
            field_name
            for field_name in exact_workload_fields
            if aggregate.get(field_name) != baseline_aggregate.get(field_name)
        ]
        if not allow_workload_variation and not math.isclose(
            float(aggregate.get("weighted_supervision_mass", 0.0)),
            float(baseline_aggregate.get("weighted_supervision_mass", 0.0)),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            changed.append("weighted_supervision_mass")
        if changed:
            mismatches.append(
                f"{name} vs {baseline_name}: committed workload " + ", ".join(changed)
            )

    incomplete = [name for name, payload in summaries if not payload.get("complete_history")]
    if incomplete:
        mismatches.append(f"incomplete committed history: {', '.join(incomplete)}")
    incomplete_sources = [
        name for name, contract in parsed if not contract.source_contract_complete
    ]
    if incomplete_sources:
        mismatches.append("incomplete source identity: " + ", ".join(incomplete_sources))
    if mismatches:
        raise ValueError(
            "Training-efficiency summaries are not a fair A/B comparison: " + "; ".join(mismatches)
        )


def build_comparison(
    paths: Sequence[str | Path],
    *,
    allow_incompatible: bool = False,
    allow_workload_variation: bool = False,
) -> list[dict[str, Any]]:
    summaries = [_load_summary(path) for path in paths]
    if not allow_incompatible:
        _validate_comparison_contracts(
            summaries,
            allow_workload_variation=allow_workload_variation,
        )
    comparison_mode = (
        "incompatible"
        if allow_incompatible
        else "capacity"
        if allow_workload_variation
        else "exact_workload"
    )
    baseline_aggregate = summaries[0][1]["aggregate"] if summaries else None
    rows: list[dict[str, Any]] = []
    for name, payload in summaries:
        aggregate = payload["aggregate"]
        rows.append(
            {
                "run": name,
                "steps": int(aggregate["optimizer_steps"]),
                "useful_tokens_per_second": float(aggregate["useful_tokens_per_second"]),
                "logical_segments_per_second": float(
                    aggregate.get(
                        "logical_segments_per_second",
                        _safe_ratio(
                            aggregate.get("logical_segments", 0),
                            aggregate.get("critical_path_seconds", 0.0),
                        ),
                    )
                ),
                "vision_patches_per_second": float(
                    aggregate.get(
                        "vision_patches_per_second",
                        _safe_ratio(
                            aggregate.get("vision_patches", 0),
                            aggregate.get("critical_path_seconds", 0.0),
                        ),
                    )
                ),
                "supervised_tokens_per_second": float(
                    aggregate.get(
                        "supervised_tokens_per_second",
                        _safe_ratio(
                            aggregate.get("supervised_tokens", 0),
                            aggregate.get("critical_path_seconds", 0.0),
                        ),
                    )
                ),
                "padding_fraction": float(aggregate["padding_fraction"]),
                "segments_per_pack": float(aggregate["segments_per_pack"]),
                "mean_sequence_length": float(aggregate["mean_sequence_length"]),
                "sequence_length_std": float(aggregate["sequence_length_std"]),
                "useful_tokens": int(aggregate["useful_tokens"]),
                "supervised_tokens": int(aggregate["supervised_tokens"]),
                "vision_patches": int(aggregate["vision_patches"]),
                "update_applied_steps": int(aggregate["update_applied_steps"]),
                "weighted_supervision_coverage_fraction": float(
                    aggregate["weighted_supervision_coverage_fraction"]
                ),
                "vision_coverage_fraction": float(aggregate["vision_coverage_fraction"]),
                "rank_time_skew": float(payload["rank_time_skew"]),
                "critical_path_p50_seconds": float(aggregate.get("critical_path_p50_seconds", 0.0)),
                "critical_path_p95_seconds": float(aggregate.get("critical_path_p95_seconds", 0.0)),
                "peak_device_memory_allocated_gib": _bytes_to_gib(
                    payload.get("peak_device_memory_allocated_bytes")
                ),
                "peak_device_memory_reserved_gib": _bytes_to_gib(
                    payload.get("peak_device_memory_reserved_bytes")
                ),
                "comparison_mode": comparison_mode,
                "exact_workload_enforced": comparison_mode == "exact_workload",
                "workload_matched": (
                    baseline_aggregate is not None
                    and _logical_workload_matches(baseline_aggregate, aggregate)
                ),
                "complete_history": bool(payload["complete_history"]),
            }
        )
    rate_fields = (
        "useful_tokens_per_second",
        "logical_segments_per_second",
        "supervised_tokens_per_second",
        "vision_patches_per_second",
    )
    if rows:
        baseline = rows[0]
        for row in rows:
            for field_name in rate_fields:
                row[f"{field_name}_delta_fraction"] = _relative_delta(
                    row[field_name],
                    baseline[field_name],
                )
    return rows


def _markdown(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        "| run | mode | steps | useful tok/s Δ | seg/s Δ | supervised tok/s Δ | "
        "vision patch/s Δ | padding | seg/pack | peak GiB | p95 | rank skew | complete |\n"
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {run} | {mode} | {steps} | {useful} | {segments_rate} | "
            "{supervised_rate} | {vision_rate} | {padding:.2%} | {segments:.2f} | "
            "{peak_memory} | {p95:.3f}s | {skew:.2%} | {complete} |".format(
                run=row["run"],
                mode=row["comparison_mode"],
                steps=row["steps"],
                useful=_format_rate_delta(row, "useful_tokens_per_second"),
                segments_rate=_format_rate_delta(
                    row,
                    "logical_segments_per_second",
                ),
                supervised_rate=_format_rate_delta(
                    row,
                    "supervised_tokens_per_second",
                ),
                vision_rate=_format_rate_delta(
                    row,
                    "vision_patches_per_second",
                ),
                padding=row["padding_fraction"],
                segments=row["segments_per_pack"],
                peak_memory=_format_optional_float(
                    row["peak_device_memory_allocated_gib"],
                    digits=2,
                ),
                p95=row["critical_path_p95_seconds"],
                skew=row["rank_time_skew"],
                complete="yes" if row["complete_history"] else "no",
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare committed Shaft training-efficiency summaries."
    )
    parser.add_argument("runs", nargs="+", help="Run directories or summary JSON files.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--allow-workload-variation",
        action="store_true",
        help=(
            "Allow token/segment/vision work to differ while keeping the run identity, "
            "step span, update/microbatch/physical-pack counts, and telemetry coverage fixed."
        ),
    )
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="Compare runs even when model/data/topology/training contracts differ.",
    )
    args = parser.parse_args(argv)
    rows = build_comparison(
        args.runs,
        allow_incompatible=bool(args.allow_incompatible),
        allow_workload_variation=bool(args.allow_workload_variation),
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False) if args.json else _markdown(rows))


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    denominator_value = float(denominator)
    if not math.isfinite(denominator_value) or denominator_value <= 0:
        return 0.0
    return float(numerator) / denominator_value


def _logical_workload_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    integer_fields = (
        "logical_segments",
        "useful_tokens",
        "supervised_tokens",
        "vision_patches",
        "sequence_length_sum",
        "sequence_length_square_sum",
    )
    if any(left.get(name) != right.get(name) for name in integer_fields):
        return False
    return math.isclose(
        float(left.get("weighted_supervision_mass", 0.0)),
        float(right.get("weighted_supervision_mass", 0.0)),
        rel_tol=1e-6,
        abs_tol=1e-6,
    )


def _bytes_to_gib(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1024**3)


def _relative_delta(value: float | int, baseline: float | int) -> float | None:
    baseline_value = float(baseline)
    if not math.isfinite(baseline_value) or baseline_value <= 0:
        return None
    return float(value) / baseline_value - 1.0


def _format_rate_delta(row: dict[str, Any], field_name: str) -> str:
    rate = float(row[field_name])
    delta = row[f"{field_name}_delta_fraction"]
    delta_text = "n/a" if delta is None else f"{float(delta):+.2%}"
    return f"{rate:.1f} ({delta_text})"


def _format_optional_float(value: float | None, *, digits: int) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"
