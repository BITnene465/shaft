from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
import random
from typing import Any, TypeVar, cast

import numpy as np
import torch
from transformers import enable_full_determinism, set_seed


_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


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


@contextmanager
def preserve_training_rng_state() -> Iterator[None]:
    """Make observational work unable to advance the training RNG stream.

    Eval loader construction and sampled generation can consume Python, NumPy,
    Torch CPU, or rank-local CUDA randomness.  Exact resume therefore requires
    the entire observation boundary to restore every process-local stream.
    """

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_cpu_state = torch.random.get_rng_state()
    cuda_device: int | None = None
    cuda_state: torch.Tensor | None = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_device = torch.cuda.current_device()
        cuda_state = torch.cuda.get_rng_state(cuda_device)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_cpu_state)
        if cuda_device is not None and cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, cuda_device)


def isolate_training_rng_during_eval(function: _CallableT) -> _CallableT:
    """Decorate Trainer.evaluate while leaving standalone evaluation unchanged."""

    @wraps(function)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not bool(getattr(self, "is_in_train", False)):
            return function(self, *args, **kwargs)
        with preserve_training_rng_state():
            return function(self, *args, **kwargs)

    return cast(_CallableT, wrapped)
