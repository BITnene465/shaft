from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from .app import create_app, main

__all__ = [
    "create_app",
    "main",
]
