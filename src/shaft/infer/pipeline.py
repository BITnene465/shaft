from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .engine import InferEngine, InferRequest
from .schema import InferPipelineConfig, InferStageConfig


@dataclass
class InferStageResult:
    stage: str
    engine: str
    output_key: str
    output_text: str
    prompt: str


class InferPipeline:
    def __init__(self, *, engines: dict[str, InferEngine], stages: list[InferStageConfig]) -> None:
        if not engines:
            raise ValueError("InferPipeline requires at least one engine.")
        if not stages:
            raise ValueError("InferPipeline requires at least one stage.")
        self.engines = engines
        self.stages = stages

    @classmethod
    def from_config(cls, config: InferPipelineConfig) -> "InferPipeline":
        engines = {name: InferEngine.from_model_config(spec) for name, spec in config.engines.items()}
        return cls(engines=engines, stages=list(config.stages))

    def run(self, *, image_path: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        context: dict[str, Any] = dict(inputs or {})
        context["image_path"] = image_path
        traces: list[InferStageResult] = []

        for stage in self.stages:
            engine = self.engines.get(stage.engine)
            if engine is None:
                raise KeyError(f"Engine {stage.engine!r} not found for stage {stage.name!r}.")
            user_prompt = stage.user_prompt_template.format(**context)
            response = engine.run(
                InferRequest(
                    image_path=image_path,
                    system_prompt=stage.system_prompt,
                    user_prompt=user_prompt,
                    generation=stage.generation,
                )
            )
            output_key = stage.output_key or stage.name
            context[output_key] = response.text
            traces.append(
                InferStageResult(
                    stage=stage.name,
                    engine=stage.engine,
                    output_key=output_key,
                    output_text=response.text,
                    prompt=response.prompt,
                )
            )

        context["__trace__"] = traces
        return context

