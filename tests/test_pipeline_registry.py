from __future__ import annotations

from shaft.pipeline import PIPELINE_REGISTRY, ShaftRLHFPipeline, ShaftSFTPipeline


def test_shaft_sft_pipeline_registered() -> None:
    pipeline_cls = PIPELINE_REGISTRY.get("shaft_sft")
    assert pipeline_cls is ShaftSFTPipeline


def test_shaft_rlhf_pipeline_registered() -> None:
    pipeline_cls = PIPELINE_REGISTRY.get("shaft_rlhf")
    assert pipeline_cls is ShaftRLHFPipeline
