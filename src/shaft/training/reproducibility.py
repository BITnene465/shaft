from __future__ import annotations

from transformers import enable_full_determinism, set_seed


def initialize_training_randomness(
    *,
    seed: int,
    full_determinism: bool,
) -> None:
    """Initialize model/adapter randomness before any training component is built."""

    resolved_seed = int(seed)
    if full_determinism:
        enable_full_determinism(resolved_seed)
        return
    set_seed(resolved_seed)
