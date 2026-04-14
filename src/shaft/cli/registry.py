from __future__ import annotations

from typing import Any, Protocol

from shaft.plugins import Registry


class CommandSpec(Protocol):
    help: str

    @classmethod
    def configure_parser(cls, parser: Any) -> None: ...

    @classmethod
    def run(cls, args: Any) -> Any: ...


COMMAND_REGISTRY: Registry[type[CommandSpec]] = Registry("cli_command")


def register_command(name: str):
    return COMMAND_REGISTRY.register(name)
