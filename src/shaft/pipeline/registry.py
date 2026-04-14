from __future__ import annotations

from shaft.plugins import Registry

PIPELINE_REGISTRY: Registry[type] = Registry("pipeline")


def register_pipeline(name: str):
    return PIPELINE_REGISTRY.register(name)
