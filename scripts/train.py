#!/usr/bin/env python
from __future__ import annotations

import os
import warnings


def _configure_early_warning_filters() -> None:
    rank = int(os.environ.get("RANK", "0") or "0")
    if rank != 0:
        warnings.filterwarnings(
            "ignore",
            message=r"TRL currently supports vLLM versions:.*",
            category=UserWarning,
        )
    else:
        warnings.filterwarnings(
            "once",
            message=r"TRL currently supports vLLM versions:.*",
            category=UserWarning,
        )
    warnings.filterwarnings(
        "ignore",
        message=r"The cuda\.(cudart|nvrtc) module is deprecated.*",
        category=FutureWarning,
    )


_configure_early_warning_filters()

from shaft.cli.train import main  # noqa: E402


if __name__ == "__main__":
    main()
    
