from __future__ import annotations

from shaft.plugins import Registry

from .base import Algorithm

ALGORITHM_REGISTRY: Registry[type[Algorithm]] = Registry("algorithm")


def register_algorithm(name: str):
    return ALGORITHM_REGISTRY.register(name)
