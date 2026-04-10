from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vlm_structgen.core.config import _from_dict
from vlm_structgen.core.routing import normalize_route_key
from vlm_structgen.runtime.infer.config import (
    InferAppConfig,
    InferEvalConfig,
    InferModelConfig,
    InferPromptConfig,
    InferTaskConfig,
    resolve_prompt_profile_for_mapping,
)


@dataclass
class TwoStageStageInferenceConfig:
    model: InferModelConfig = field(default_factory=InferModelConfig)
    task: InferTaskConfig = field(default_factory=InferTaskConfig)
    prompt: InferPromptConfig = field(default_factory=InferPromptConfig)
    eval: InferEvalConfig = field(default_factory=InferEvalConfig)
    batch_size: int = 1


@dataclass
class TwoStageInferenceConfig:
    stage1: TwoStageStageInferenceConfig = field(default_factory=TwoStageStageInferenceConfig)
    stage2: TwoStageStageInferenceConfig = field(default_factory=TwoStageStageInferenceConfig)
    app: InferAppConfig = field(default_factory=InferAppConfig)
    output_dir: str | None = None
    padding_ratio: float = 0.5


def _load_yaml_payload(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_two_stage_inference_config(path: str | Path | None) -> TwoStageInferenceConfig:
    payload = _load_yaml_payload(path)
    if path is not None:
        resolve_prompt_profile_for_mapping(payload.get("stage1"), config_path=Path(path))
        resolve_prompt_profile_for_mapping(payload.get("stage2"), config_path=Path(path))
    config = _from_dict(TwoStageInferenceConfig, payload)
    if config.stage1.task.route is not None and str(config.stage1.task.route).strip():
        config.stage1.task.route = normalize_route_key(config.stage1.task.route)
    if config.stage2.task.route is not None and str(config.stage2.task.route).strip():
        config.stage2.task.route = normalize_route_key(config.stage2.task.route)
    return config
