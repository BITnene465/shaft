from __future__ import annotations

from shaft.plugins import Registry

from .base import RLRuntime


RL_RUNTIME_REGISTRY: Registry[type[RLRuntime]] = Registry("rl_runtime")


def register_rl_runtime(name: str):
    return RL_RUNTIME_REGISTRY.register(name)


def build_rl_runtime(name: str) -> RLRuntime:
    runtime_cls = RL_RUNTIME_REGISTRY.get(name)
    return runtime_cls()
