from __future__ import annotations

import importlib
from typing import Final

_BUILTIN_ADAPTER_MODULES: Final[tuple[str, ...]] = (
    "vlm_structgen.tasks.grounding.adapter",
    "vlm_structgen.tasks.keypoint_sequence.adapter",
    "vlm_structgen.tasks.joint_structure.adapter",
)

_REGISTERED = False


def ensure_builtin_task_adapters_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    for module_name in _BUILTIN_ADAPTER_MODULES:
        importlib.import_module(module_name)
    _REGISTERED = True

