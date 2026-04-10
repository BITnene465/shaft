"""Core data interfaces."""

from __future__ import annotations

from typing import Any

__all__ = [
    "SFTCollator",
    "SFTDataset",
    "build_mixed_train_loader",
    "ResolvedDatasetSources",
    "resolve_training_data_sources",
]


def __getattr__(name: str) -> Any:
    if name == "SFTCollator":
        from .collator import SFTCollator

        return SFTCollator
    if name == "SFTDataset":
        from .dataset import SFTDataset

        return SFTDataset
    if name == "build_mixed_train_loader":
        from .mixed_loader import build_mixed_train_loader

        return build_mixed_train_loader
    if name == "ResolvedDatasetSources":
        from .registry_loader import ResolvedDatasetSources

        return ResolvedDatasetSources
    if name == "resolve_training_data_sources":
        from .registry_loader import resolve_training_data_sources

        return resolve_training_data_sources
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
