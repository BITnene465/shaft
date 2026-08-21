from __future__ import annotations

import os
import re
from pathlib import Path


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_project_environment(path: str | Path) -> tuple[str, ...]:
    """Load a simple project-local environment file without overriding the shell.

    The format intentionally supports only ``NAME=value`` and ``export NAME=value``.
    Values may be wrapped in matching single or double quotes. Shell expansion and
    command substitution are never evaluated.
    """

    environment_path = Path(path)
    if not environment_path.is_file():
        return ()

    loaded: list[str] = []
    for line_number, raw_line in enumerate(
        environment_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not _ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError(
                f"Invalid environment assignment in {environment_path}:{line_number}."
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return tuple(loaded)
