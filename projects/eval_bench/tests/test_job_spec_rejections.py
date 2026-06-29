from __future__ import annotations

import pytest

from eval_bench.job_spec import job_templates, resolve_job_payload


def test_payload_rejects_manifest_and_job_manifest_dual_sources() -> None:
    manifest = job_templates()["eval_job"]["manifest"]

    with pytest.raises(ValueError, match="must not define both manifest and job_manifest"):
        resolve_job_payload({"manifest": manifest, "job_manifest": manifest})


def test_legacy_eval_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="manifest-first suite schema"):
        resolve_job_payload(
            {
                "backend": "dry_run",
                "model_id": "model-a",
                "model_path": "outputs/model-a/best",
                "benchmark_id": "bench1",
                "task": "detection",
                "prompt_id": "grounding_layout.test.main",
                "target_labels": ["icon", "image"],
            }
        )


@pytest.mark.parametrize("kind", ["eval", "preannotate"])
def test_legacy_job_kind_aliases_are_rejected(kind: str) -> None:
    section_key = "eval" if kind == "eval" else "preannotate"
    section = (
        {
            "model_id": "model-a",
            "model_path": "outputs/model-a/best",
            "benchmark_id": "bench1",
            "task": "detection",
            "prompt_id": "grounding_layout.test.main",
        }
        if kind == "eval"
        else {
            "model_id": "model-a",
            "source_root": "raw",
            "output_root": "out",
        }
    )
    with pytest.raises(ValueError, match=f"unsupported job kind: {kind}"):
        resolve_job_payload(
            {
                "manifest": {
                    "kind": kind,
                    "runtime": {"mode": "existing_service", "engine": "dry_run"},
                    section_key: section,
                }
            }
        )
