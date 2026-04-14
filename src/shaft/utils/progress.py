from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm

SHAFT_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def create_progress_bar(
    *,
    total: int | None,
    desc: str,
    unit: str = "it",
    leave: bool = False,
    dynamic_ncols: bool = True,
    mininterval: float = 0.2,
    colour: str | None = None,
) -> tqdm:
    kwargs: dict[str, Any] = {
        "total": total,
        "desc": desc,
        "unit": unit,
        "leave": leave,
        "dynamic_ncols": dynamic_ncols,
        "mininterval": float(mininterval),
        "bar_format": SHAFT_BAR_FORMAT,
    }
    if colour is not None:
        kwargs["colour"] = str(colour)
    return tqdm(**kwargs)
