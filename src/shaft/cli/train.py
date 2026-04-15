from __future__ import annotations

import argparse
import sys

from . import rlhf as _rlhf  # noqa: F401
from . import sft as _sft  # noqa: F401
from .registry import COMMAND_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified training entry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in COMMAND_REGISTRY.keys():
        command_cls = COMMAND_REGISTRY.get(command_name)
        subparser = subparsers.add_parser(command_name, help=getattr(command_cls, "help", ""))
        command_cls.configure_parser(subparser)
        subparser.set_defaults(_command_cls=command_cls)
    return parser


def _normalize_argv(argv: list[str] | None) -> list[str]:
    # Backward compatibility:
    # `python scripts/train.py --config xxx.yaml` -> defaults to `sft`.
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        return tokens
    known_commands = set(COMMAND_REGISTRY.keys())
    if tokens[0] in known_commands:
        return tokens
    if tokens[0].startswith("-"):
        return ["sft", *tokens]
    return tokens


def main(argv: list[str] | None = None) -> None:
    argv = _normalize_argv(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    command_cls = getattr(args, "_command_cls")
    command_cls.run(args)


if __name__ == "__main__":
    main()
