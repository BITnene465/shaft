from __future__ import annotations

from collections.abc import Callable

from shaft.plugins import Registry

DataSourceFactory = Callable[..., object]
MixStrategy = Callable[..., list[tuple[str, int]]]

DATA_SOURCE_REGISTRY: Registry[DataSourceFactory] = Registry("data_source")
MIX_STRATEGY_REGISTRY: Registry[MixStrategy] = Registry("data_mix_strategy")


def register_data_source(name: str):
    return DATA_SOURCE_REGISTRY.register(name)


def register_mix_strategy(name: str):
    return MIX_STRATEGY_REGISTRY.register(name)
