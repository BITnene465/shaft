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
        or int(payload.get("schema_version", -1))
        != TRAINING_EFFICIENCY_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported training-efficiency summary: {resolved}")
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"Training-efficiency summary has no aggregate: {resolved}")
    return resolved.parent.name or resolved.stem, payload


def _validate_comparison_contracts(
    summaries: Sequence[tuple[str, dict[str, Any]]],
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
        parsed.append(
            (name, ShaftTrainingEfficiencyContract.from_dict(dict(raw_contract)))
        )

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

    exact_workload_fields = (
        "logical_segments",
        "useful_tokens",
        "supervised_tokens",
        "vision_patches",
        "update_applied_steps",
        "weighted_supervision_coverage_microbatches",
        "vision_coverage_batches",
    )
    baseline_aggregate = summaries[0][1]["aggregate"]
    for name, payload in summaries[1:]:
        aggregate = payload["aggregate"]
        changed = [
            field_name
            for field_name in exact_workload_fields
            if aggregate.get(field_name) != baseline_aggregate.get(field_name)
        ]
        if not math.isclose(
            float(aggregate.get("weighted_supervision_mass", 0.0)),
            float(baseline_aggregate.get("weighted_supervision_mass", 0.0)),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            changed.append("weighted_supervision_mass")
        if changed:
            mismatches.append(
                f"{name} vs {baseline_name}: committed workload "
                + ", ".join(changed)
            )

    incomplete = [name for name, payload in summaries if not payload.get("complete_history")]
    if incomplete:
        mismatches.append(f"incomplete committed history: {', '.join(incomplete)}")
    incomplete_sources = [
        name for name, contract in parsed if not contract.source_contract_complete
    ]
    if incomplete_sources:
        mismatches.append(
            "incomplete source identity: " + ", ".join(incomplete_sources)
        )
    if mismatches:
        raise ValueError(
            "Training-efficiency summaries are not a fair A/B comparison: "
            + "; ".join(mismatches)
        )


def build_comparison(
    paths: Sequence[str | Path],
    *,
    allow_incompatible: bool = False,
) -> list[dict[str, Any]]:
    summaries = [_load_summary(path) for path in paths]
    if not allow_incompatible:
        _validate_comparison_contracts(summaries)
    rows: list[dict[str, Any]] = []
    for name, payload in summaries:
        aggregate = payload["aggregate"]
        rows.append(
            {
                "run": name,
                "steps": int(aggregate["optimizer_steps"]),
                "useful_tokens_per_second": float(
                    aggregate["useful_tokens_per_second"]
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
                "vision_coverage_fraction": float(
                    aggregate["vision_coverage_fraction"]
                ),
                "rank_time_skew": float(payload["rank_time_skew"]),
                "complete_history": bool(payload["complete_history"]),
            }
        )
    return rows


def _markdown(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        "| run | steps | useful tok/s | padding | segments/pack | seq mean±std | "
        "tokens | supervised | vision | updates | weighted cov. | vision cov. | "
        "rank skew | complete |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {run} | {steps} | {throughput:.1f} | {padding:.2%} | "
            "{segments:.2f} | {seq_mean:.1f}±{seq_std:.1f} | {tokens} | "
            "{supervised} | {vision} | {updates} | {weighted_coverage:.1%} | "
            "{vision_coverage:.1%} | {skew:.2%} | {complete} |".format(
                run=row["run"],
                steps=row["steps"],
                throughput=row["useful_tokens_per_second"],
                padding=row["padding_fraction"],
                segments=row["segments_per_pack"],
                seq_mean=row["mean_sequence_length"],
                seq_std=row["sequence_length_std"],
                tokens=row["useful_tokens"],
                supervised=row["supervised_tokens"],
                vision=row["vision_patches"],
                updates=row["update_applied_steps"],
                weighted_coverage=row["weighted_supervision_coverage_fraction"],
                vision_coverage=row["vision_coverage_fraction"],
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
        "--allow-incompatible",
        action="store_true",
        help="Compare runs even when model/data/topology/training contracts differ.",
    )
    args = parser.parse_args(argv)
    rows = build_comparison(
        args.runs,
        allow_incompatible=bool(args.allow_incompatible),
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False) if args.json else _markdown(rows))
