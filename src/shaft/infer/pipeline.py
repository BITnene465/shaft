from __future__ import annotations

from dataclasses import dataclass
import copy
import threading
import time
from typing import Any

from shaft.codec import decode_with_codec
from shaft.prompting import ShaftPromptProgram
from .engine import ShaftInferEngine, ShaftInferRequest
from .execution import (
    ShaftInferCancelledError,
    ShaftInferExecutionControl,
    ShaftInferExecutionControlUnsupportedError,
)
from .schema import InferPipelineConfig, InferStageConfig, compile_stage_prompt


@dataclass
class ShaftInferStageAttempt:
    attempt: int
    success: bool
    latency_ms: float
    prompt: str | None = None
    output_text: str | None = None
    error: str | None = None


@dataclass
class ShaftInferStageResult:
    stage: str
    engine: str
    output_key: str
    codec: str
    success: bool
    attempts: int
    latency_ms: float
    output_text: str | None
    parsed: Any = None
    error: str | None = None
    prompt: str | None = None
    history: list[ShaftInferStageAttempt] | None = None
    prompt_audit: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedStage:
    config: InferStageConfig
    prompt: ShaftPromptProgram


class ShaftInferPipeline:
    def __init__(
        self, *, engines: dict[str, ShaftInferEngine], stages: list[InferStageConfig]
    ) -> None:
        if not engines:
            raise ValueError("ShaftInferPipeline requires at least one engine.")
        if not stages:
            raise ValueError("ShaftInferPipeline requires at least one stage.")
        self.engines = dict(engines)
        self._stages = tuple(_resolve_stage(stage) for stage in stages)

    @property
    def stages(self) -> tuple[InferStageConfig, ...]:
        return tuple(copy.deepcopy(item.config) for item in self._stages)

    @classmethod
    def from_config(cls, config: InferPipelineConfig) -> "ShaftInferPipeline":
        engines = {
            name: ShaftInferEngine.from_engine_config(spec) for name, spec in config.engines.items()
        }
        return cls(engines=engines, stages=list(config.stages))

    def run(
        self,
        *,
        image_path: str,
        inputs: dict[str, Any] | None = None,
        cancellation_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        context: dict[str, Any] = dict(inputs or {})
        context["image_path"] = image_path
        traces: list[ShaftInferStageResult] = []

        for resolved in self._stages:
            stage = resolved.config
            result = self._run_stage(
                stage=stage,
                prompt_program=resolved.prompt,
                image_path=image_path,
                context=context,
                cancellation_event=cancellation_event,
            )
            traces.append(result)
            if result.success:
                context[result.output_key] = result.parsed
                context[f"{result.output_key}__raw"] = result.output_text
            elif stage.fail_fast:
                context["__trace__"] = traces
                raise RuntimeError(
                    f"Stage {stage.name!r} failed after {result.attempts} attempt(s): {result.error}"
                )
            else:
                context[result.output_key] = None
                context[f"{result.output_key}__raw"] = result.output_text
                context[f"{result.output_key}__error"] = result.error

        context["__trace__"] = traces
        return context

    def _run_stage(
        self,
        *,
        stage: InferStageConfig,
        prompt_program: ShaftPromptProgram,
        image_path: str,
        context: dict[str, Any],
        cancellation_event: threading.Event | None,
    ) -> ShaftInferStageResult:
        engine = self.engines.get(stage.engine)
        if engine is None:
            raise KeyError(f"Engine {stage.engine!r} not found for stage {stage.name!r}.")
        output_key = stage.output_key or stage.name
        codec_name = str(stage.codec).strip().lower()
        max_retries = max(int(stage.max_retries), 0)
        timeout_seconds = (
            float(stage.timeout_seconds) if stage.timeout_seconds is not None else None
        )
        attempts: list[ShaftInferStageAttempt] = []
        stage_start = time.perf_counter()
        stage_started_monotonic = time.monotonic()
        execution = (
            ShaftInferExecutionControl(
                deadline_monotonic=(
                    stage_started_monotonic + timeout_seconds
                    if timeout_seconds is not None
                    else None
                ),
                cancellation_event=cancellation_event,
            )
            if timeout_seconds is not None or cancellation_event is not None
            else None
        )
        latest_error: str | None = None
        latest_output_text: str | None = None
        latest_prompt: str | None = None
        latest_parsed: Any = None
        prompt_values = {
            name: context[name] for name in prompt_program.schema.names if name in context
        }
        user_prompt, prompt_audit = prompt_program.render_with_audit(
            prompt_values,
            context=f"infer stage {stage.name!r}",
        )

        for attempt_index in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                if execution is not None:
                    execution.checkpoint(context=f"Stage {stage.name!r}")
                    validator = getattr(engine, "validate_execution_control", None)
                    if not callable(validator):
                        raise ShaftInferExecutionControlUnsupportedError(
                            f"Engine {stage.engine!r} does not declare execution-control support "
                            f"for stage {stage.name!r}; refusing to run an unbounded attempt."
                        )
                    validator(execution)
                response = engine.run(
                    ShaftInferRequest(
                        image_path=image_path,
                        system_prompt=stage.system_prompt,
                        user_prompt=user_prompt,
                        generation=stage.generation,
                        min_pixels=stage.min_pixels,
                        max_pixels=stage.max_pixels,
                        backend_options=stage.backend_options,
                        execution=execution,
                    )
                )
                if execution is not None:
                    execution.checkpoint(context=f"Stage {stage.name!r}")
                decoded = decode_with_codec(codec_name, response.text)
                if execution is not None:
                    execution.checkpoint(
                        context=f"Stage {stage.name!r} codec decode"
                    )
                if not decoded.valid:
                    raise ValueError(
                        decoded.error or f"codec={codec_name!r} failed to decode model output."
                    )
                elapsed_seconds = time.perf_counter() - t0
                elapsed_ms = elapsed_seconds * 1000.0
                attempts.append(
                    ShaftInferStageAttempt(
                        attempt=attempt_index + 1,
                        success=True,
                        latency_ms=elapsed_ms,
                        prompt=response.prompt,
                        output_text=response.text,
                    )
                )
                latest_output_text = response.text
                latest_prompt = response.prompt
                latest_parsed = decoded.parsed
                return ShaftInferStageResult(
                    stage=stage.name,
                    engine=stage.engine,
                    output_key=output_key,
                    codec=codec_name,
                    success=True,
                    attempts=attempt_index + 1,
                    latency_ms=(time.perf_counter() - stage_start) * 1000.0,
                    output_text=latest_output_text,
                    parsed=latest_parsed,
                    prompt=latest_prompt,
                    history=attempts,
                    prompt_audit=prompt_audit,
                )
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, ShaftInferCancelledError):
                    raise
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latest_error = str(exc)
                attempts.append(
                    ShaftInferStageAttempt(
                        attempt=attempt_index + 1,
                        success=False,
                        latency_ms=elapsed_ms,
                        error=latest_error,
                    )
                )
                if isinstance(
                    exc,
                    (
                        TimeoutError,
                        ShaftInferExecutionControlUnsupportedError,
                    ),
                ):
                    break
                if attempt_index < max_retries:
                    backoff_seconds = float(stage.retry_backoff_seconds)
                    if backoff_seconds <= 0:
                        continue
                    if execution is not None:
                        remaining = execution.remaining_seconds()
                        wait_seconds = (
                            min(backoff_seconds, remaining)
                            if remaining is not None
                            else backoff_seconds
                        )
                        if execution.cancellation_event is not None:
                            execution.cancellation_event.wait(wait_seconds)
                        elif wait_seconds > 0:
                            time.sleep(wait_seconds)
                        try:
                            execution.checkpoint(context=f"Stage {stage.name!r} retry backoff")
                        except ShaftInferCancelledError:
                            raise
                        except TimeoutError as control_exc:
                            latest_error = str(control_exc)
                            break
                        if remaining is not None and wait_seconds < backoff_seconds:
                            latest_error = (
                                f"Stage {stage.name!r} deadline expired during retry backoff."
                            )
                            break
                    else:
                        time.sleep(backoff_seconds)

        return ShaftInferStageResult(
            stage=stage.name,
            engine=stage.engine,
            output_key=output_key,
            codec=codec_name,
            success=False,
            attempts=len(attempts),
            latency_ms=(time.perf_counter() - stage_start) * 1000.0,
            output_text=latest_output_text,
            parsed=latest_parsed,
            error=latest_error,
            prompt=latest_prompt,
            history=attempts,
            prompt_audit=prompt_audit,
        )


def _resolve_stage(stage: InferStageConfig) -> _ResolvedStage:
    config = copy.deepcopy(stage)
    return _ResolvedStage(
        config=config,
        prompt=compile_stage_prompt(config, source=f"infer stage {config.name!r}"),
    )
