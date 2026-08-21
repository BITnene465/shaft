from __future__ import annotations

import os
import re
from pathlib import Path


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


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


def configure_rank_local_triton_cache() -> str | None:
    """Resolve an optional node-local Triton cache directory for this torchrun rank."""

    explicit_cache_dir = str(os.environ.get("TRITON_CACHE_DIR", "")).strip()
    if explicit_cache_dir:
        return explicit_cache_dir

    configured_root = str(os.environ.get("SHAFT_TRITON_CACHE_ROOT", "")).strip()
    if not configured_root:
        return None
    cache_root = Path(configured_root).expanduser()
    if not cache_root.is_absolute():
        raise ValueError("SHAFT_TRITON_CACHE_ROOT must be an absolute node-local path.")

    run_id = str(os.environ.get("TORCHELASTIC_RUN_ID", "standalone")).strip()
    safe_run_id = _PATH_COMPONENT.sub("-", run_id).strip("-.") or "standalone"
    safe_run_id = safe_run_id[:128]
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as exc:
        raise ValueError("LOCAL_RANK must be an integer for Triton cache isolation.") from exc
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be >= 0 for Triton cache isolation.")

    cache_dir = cache_root / safe_run_id / f"rank-{local_rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    return str(cache_dir)
