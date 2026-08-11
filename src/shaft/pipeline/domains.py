from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from shaft.config import RuntimeConfig
from shaft.config.algorithm import resolve_algorithm_profile


TrainingDomainRunner = Callable[[RuntimeConfig], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class TrainingDomainSpec:
    """One top-level training domain and the algorithms it exclusively owns."""

    name: str
    runner: TrainingDomainRunner

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if not name:
            raise ValueError("Training domain name cannot be empty.")
        if not callable(self.runner):
            raise TypeError(f"Training domain {name!r} runner must be callable.")
        object.__setattr__(self, "name", name)


class TrainingDomainRegistry:
    """Resolve algorithms to domains without central algorithm conditionals."""

    def __init__(self) -> None:
        self._domains: dict[str, TrainingDomainSpec] = {}

    def register(self, spec: TrainingDomainSpec) -> TrainingDomainSpec:
        existing_domain = self._domains.get(spec.name)
        if existing_domain is not None and existing_domain is not spec:
            raise ValueError(f"Duplicate training domain {spec.name!r}.")
        self._domains[spec.name] = spec
        return spec

    def resolve(self, algorithm: str) -> TrainingDomainSpec:
        profile = resolve_algorithm_profile(algorithm)
        return self.get(profile.domain)

    def get(self, name: str) -> TrainingDomainSpec:
        normalized = str(name).strip().lower()
        try:
            return self._domains[normalized]
        except KeyError as exc:
            raise KeyError(
                f"Training domain {normalized!r} is not registered."
            ) from exc

    def keys(self) -> list[str]:
        return sorted(self._domains)

def _run_sft(config: RuntimeConfig) -> dict[str, Any]:
    from .sft import run_sft

    return run_sft(config)


def _run_rl(config: RuntimeConfig) -> dict[str, Any]:
    from .rl import run_rl

    return run_rl(config)


def _run_opd(config: RuntimeConfig) -> dict[str, Any]:
    from .opd import run_opd

    return run_opd(config)


TRAINING_DOMAIN_REGISTRY = TrainingDomainRegistry()
TRAINING_DOMAIN_REGISTRY.register(
    TrainingDomainSpec(name="sft", runner=_run_sft)
)
TRAINING_DOMAIN_REGISTRY.register(
    TrainingDomainSpec(name="opd", runner=_run_opd)
)
TRAINING_DOMAIN_REGISTRY.register(
    TrainingDomainSpec(
        name="rl",
        runner=_run_rl,
    )
)


def run_training_domain(config: RuntimeConfig) -> dict[str, Any]:
    spec = TRAINING_DOMAIN_REGISTRY.resolve(config.algorithm.name)
    return spec.runner(config)
