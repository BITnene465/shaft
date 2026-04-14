from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., T] | T] = {}

    def register(self, key: str, value: Callable[..., T] | T | None = None):
        normalized = key.strip().lower()
        if not normalized:
            raise ValueError(f"Registry key cannot be empty: {self.name}")

        def _decorator(obj: Callable[..., T] | T):
            if normalized in self._items and self._items[normalized] is not obj:
                raise ValueError(f"Duplicate registry key {normalized!r} in {self.name}.")
            self._items[normalized] = obj
            return obj

        if value is not None:
            return _decorator(value)
        return _decorator

    def has(self, key: str) -> bool:
        return key.strip().lower() in self._items

    def get(self, key: str) -> Callable[..., T] | T:
        normalized = key.strip().lower()
        if normalized not in self._items:
            raise KeyError(f"{normalized!r} not found in registry {self.name}.")
        return self._items[normalized]

    def create(self, key: str, *args: Any, **kwargs: Any) -> T:
        item = self.get(key)
        if callable(item):
            return item(*args, **kwargs)
        if args or kwargs:
            raise TypeError(f"Registry item {key!r} in {self.name} is not callable.")
        return item

    def keys(self) -> list[str]:
        return sorted(self._items.keys())

