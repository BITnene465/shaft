from __future__ import annotations

from shaft.pipeline import PIPELINE_REGISTRY, ShaftTrainPipeline


def test_shaft_train_pipeline_registered() -> None:
    pipeline_cls = PIPELINE_REGISTRY.get("shaft_train")
    assert pipeline_cls is ShaftTrainPipeline
