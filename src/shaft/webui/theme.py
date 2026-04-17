from __future__ import annotations

from pathlib import Path


def templates_dir() -> Path:
    return Path(__file__).with_name("templates")


def static_dir() -> Path:
    return Path(__file__).with_name("static")
