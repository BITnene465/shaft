from __future__ import annotations

from dataclasses import dataclass, field

from .algorithm import AlgorithmConfig, RLHFConfig
from .base import ExperimentConfig, LoggingConfig, PluginsConfig, ProgressConfig
from .data import DataConfig
from .model import ModelConfig
from .training import EvalConfig, TrainConfig


@dataclass
class RuntimeConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    rlhf: RLHFConfig = field(default_factory=RLHFConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
