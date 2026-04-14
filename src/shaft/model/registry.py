from __future__ import annotations

from shaft.plugins import Registry

from .types import ModelArtifacts

MODEL_REGISTRY: Registry[ModelArtifacts] = Registry("model")


def register_model(name: str):
    return MODEL_REGISTRY.register(name)
