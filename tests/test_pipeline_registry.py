from __future__ import annotations

from shaft.pipeline import PIPELINE_REGISTRY, ShaftRLHFPipeline, ShaftTrainPipeline


def test_shaft_train_pipeline_registered() -> None:
    pipeline_cls = PIPELINE_REGISTRY.get("shaft_train")
    assert pipeline_cls is ShaftTrainPipeline


def test_shaft_rlhf_pipeline_registered() -> None:
    pipeline_cls = PIPELINE_REGISTRY.get("shaft_rlhf")
    assert pipeline_cls is ShaftRLHFPipeline
