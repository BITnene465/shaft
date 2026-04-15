from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    name: str = "shaft"
    seed: int = 42
    output_dir: str = "outputs/default"
    run_id: str | None = None


@dataclass
class PluginsConfig:
    hooks: list[str] = field(default_factory=list)
    interceptors: list[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    fmt: str = "text"  # text | json
    file_path: str | None = None
    rank_zero_only: bool = True


@dataclass
class ProgressConfig:
    enabled: bool = True
    leave: bool = False
    mininterval: float = 0.2
