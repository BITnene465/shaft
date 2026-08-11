from __future__ import annotations

from shaft.pipeline import (
    PIPELINE_REGISTRY,
    ShaftOPDPipeline,
    ShaftRLPipeline,
    ShaftSFTPipeline,
)
from shaft.pipeline.domains import (
    TRAINING_DOMAIN_REGISTRY,
    TrainingDomainRegistry,
    TrainingDomainSpec,
)

import pytest


def test_shaft_sft_pipeline_registered() -> None:
    pipeline_cls = PIPELINE_REGISTRY.get("shaft_sft")
    assert pipeline_cls is ShaftSFTPipeline


def test_shaft_rl_pipeline_has_one_canonical_registry_key() -> None:
    assert PIPELINE_REGISTRY.get("shaft_rl") is ShaftRLPipeline
    assert not PIPELINE_REGISTRY.has("shaft_rlhf")


def test_shaft_opd_pipeline_registered() -> None:
    pipeline_cls = PIPELINE_REGISTRY.get("shaft_opd")
    assert pipeline_cls is ShaftOPDPipeline


def test_training_domains_are_parallel_and_algorithm_owned() -> None:
    assert PIPELINE_REGISTRY.keys() == ["shaft_opd", "shaft_rl", "shaft_sft"]
    assert TRAINING_DOMAIN_REGISTRY.keys() == ["opd", "rl", "sft"]
    assert TRAINING_DOMAIN_REGISTRY.resolve("sft").name == "sft"
    assert TRAINING_DOMAIN_REGISTRY.resolve("dpo").name == "rl"
    assert TRAINING_DOMAIN_REGISTRY.resolve("ppo").name == "rl"
    assert TRAINING_DOMAIN_REGISTRY.resolve("grpo").name == "rl"
    assert TRAINING_DOMAIN_REGISTRY.resolve("opd").name == "opd"


def test_training_domain_registry_rejects_cross_domain_algorithm_collision() -> None:
    registry = TrainingDomainRegistry()
    runner = lambda config: {}  # noqa: E731
    registry.register(TrainingDomainSpec("sft", runner))
    with pytest.raises(ValueError, match="Duplicate training domain"):
        registry.register(TrainingDomainSpec("sft", lambda config: {}))
