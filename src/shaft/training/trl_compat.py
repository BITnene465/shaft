from __future__ import annotations

from importlib import metadata
import os

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


SUPPORTED_TRL_SPEC_TEXT = ">=1.9.2,<2.0.0"
SUPPORTED_TRL_SPEC = SpecifierSet(SUPPORTED_TRL_SPEC_TEXT)


def _resolve_trl_version_error() -> tuple[str | None, Exception | None]:
    try:
        raw_version = metadata.version("trl")
    except metadata.PackageNotFoundError as exc:
        return None, exc
    try:
        version = Version(raw_version)
    except InvalidVersion as exc:
        return raw_version, exc
    if version not in SUPPORTED_TRL_SPEC:
        return raw_version, ImportError(
            "Unsupported TRL runtime: "
            f"installed={raw_version}, required={SUPPORTED_TRL_SPEC_TEXT}."
        )
    return raw_version, None


TRL_VERSION, _TRL_VERSION_ERROR = _resolve_trl_version_error()

if _TRL_VERSION_ERROR is None:
    try:
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer as TRLDPOTrainer
    except Exception as exc:  # noqa: BLE001
        TRLDPOConfig = None  # type: ignore[assignment]
        TRLDPOTrainer = object
        DPO_IMPORT_ERROR = exc
    else:
        DPO_IMPORT_ERROR = None
else:
    TRLDPOConfig = None  # type: ignore[assignment]
    TRLDPOTrainer = object
    DPO_IMPORT_ERROR = _TRL_VERSION_ERROR

if _TRL_VERSION_ERROR is None:
    try:
        from trl.experimental.ppo import PPOConfig as TRLPPOConfig
        from trl.experimental.ppo import PPOTrainer as TRLPPOTrainer
    except Exception as exc:  # noqa: BLE001
        TRLPPOConfig = None  # type: ignore[assignment]
        TRLPPOTrainer = object
        PPO_IMPORT_ERROR = exc
    else:
        PPO_IMPORT_ERROR = None
else:
    TRLPPOConfig = None  # type: ignore[assignment]
    TRLPPOTrainer = object
    PPO_IMPORT_ERROR = _TRL_VERSION_ERROR

if _TRL_VERSION_ERROR is None:
    try:
        from trl import GRPOConfig as TRLGRPOConfig
        from trl import GRPOTrainer as TRLGRPOTrainer
    except Exception as exc:  # noqa: BLE001
        TRLGRPOConfig = None  # type: ignore[assignment]
        TRLGRPOTrainer = object
        GRPO_IMPORT_ERROR = exc
    else:
        GRPO_IMPORT_ERROR = None
else:
    TRLGRPOConfig = None  # type: ignore[assignment]
    TRLGRPOTrainer = object
    GRPO_IMPORT_ERROR = _TRL_VERSION_ERROR


def trl_install_hint(component: str) -> str:
    version = TRL_VERSION or "<missing>"
    return (
        f"TRL {component} is unavailable (installed={version}, "
        f"required={SUPPORTED_TRL_SPEC_TEXT}). Install RLHF deps: "
        '`uv pip install -e ".[rlhf]"`.'
    )


__all__ = [
    "DPO_IMPORT_ERROR",
    "GRPO_IMPORT_ERROR",
    "PPO_IMPORT_ERROR",
    "SUPPORTED_TRL_SPEC",
    "SUPPORTED_TRL_SPEC_TEXT",
    "TRLDPOConfig",
    "TRLDPOTrainer",
    "TRLGRPOConfig",
    "TRLGRPOTrainer",
    "TRLPPOConfig",
    "TRLPPOTrainer",
    "TRL_VERSION",
    "trl_install_hint",
]
