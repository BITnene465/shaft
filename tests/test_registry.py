from __future__ import annotations

import pytest

from shaft.plugins import Registry


def test_register_and_create() -> None:
    registry: Registry[dict] = Registry("unit")

    @registry.register("demo")
    def builder(x: int):
        return {"x": x}

    out = registry.create("demo", 7)
    assert out["x"] == 7


def test_duplicate_key_raises() -> None:
    registry: Registry[dict] = Registry("unit")

    @registry.register("demo")
    def builder_a():
        return {}

    with pytest.raises(ValueError):

        @registry.register("demo")
        def builder_b():
            return {}
