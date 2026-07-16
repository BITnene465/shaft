from __future__ import annotations

from shaft.plugins import Registry

DATA_SOURCE_REGISTRY: Registry[object] = Registry("data_source")


def register_data_source(name: str):
    return DATA_SOURCE_REGISTRY.register(name)
