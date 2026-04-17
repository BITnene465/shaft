from __future__ import annotations

import os
from typing import Any

from .train_sampler_mixin import ShaftTrainSamplerMixin

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

try:
    from trl import DPOTrainer as _TRLDPOTrainer
except Exception as exc:  # noqa: BLE001
    _TRLDPOTrainer = object
    _DPO_IMPORT_ERROR = exc
else:
    _DPO_IMPORT_ERROR = None

try:
    from trl.experimental.ppo import PPOTrainer as _TRLPPOTrainer
except Exception as exc:  # noqa: BLE001
    _TRLPPOTrainer = object
    _PPO_IMPORT_ERROR = exc
else:
    _PPO_IMPORT_ERROR = None


class ShaftDPOTrainer(ShaftTrainSamplerMixin, _TRLDPOTrainer):
    """TRL DPOTrainer wrapper with Shaft naming."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _DPO_IMPORT_ERROR is not None:
            raise ImportError(
                "TRL DPO trainer is unavailable. Install RLHF deps: `uv pip install -e \".[rlhf]\"`."
            ) from _DPO_IMPORT_ERROR
        super().__init__(*args, **kwargs)


class ShaftPPOTrainer(ShaftTrainSamplerMixin, _TRLPPOTrainer):
    """TRL PPOTrainer wrapper with Shaft naming."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _PPO_IMPORT_ERROR is not None:
            raise ImportError(
                "TRL PPO trainer is unavailable. Install RLHF deps: `uv pip install -e \".[rlhf]\"`."
            ) from _PPO_IMPORT_ERROR
        super().__init__(*args, **kwargs)
