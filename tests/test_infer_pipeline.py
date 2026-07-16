from __future__ import annotations

import json
import threading
import time

import pytest

from shaft.infer import ShaftInferCancelledError
from shaft.infer.engine import ShaftInferRequest, ShaftInferResponse
from shaft.infer.pipeline import ShaftInferPipeline
from shaft.infer.schema import InferStageConfig


class _DummyEngine:
    def __init__(self, tag: str):
        self.tag = tag

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        text = f"{self.tag}:{request.user_prompt}"
        return ShaftInferResponse(text=text, prompt=request.user_prompt, output_ids=[1, 2, 3])


class _FlakyEngine:
    def __init__(self, *, fail_times: int, payload: str):
        self.fail_times = int(fail_times)
        self.payload = payload
        self.calls = 0

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("temporary failure")
        return ShaftInferResponse(
            text=self.payload, prompt=request.user_prompt, output_ids=[4, 5, 6]
        )


class _SlowEngine:
    def __init__(self, *, sleep_seconds: float, payload: str):
        self.sleep_seconds = float(sleep_seconds)
        self.payload = payload

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        time.sleep(self.sleep_seconds)
        return ShaftInferResponse(
            text=self.payload, prompt=request.user_prompt, output_ids=[7, 8, 9]
        )


class _DeclaredDeadlineEngine:
    def __init__(self, *, sleep_seconds: float, payload: str):
        self.sleep_seconds = float(sleep_seconds)
        self.payload = payload
        self.validated_execution = None
        self.request_execution = None

    def validate_execution_control(self, execution) -> None:  # noqa: ANN001
        self.validated_execution = execution

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        self.request_execution = request.execution
        time.sleep(self.sleep_seconds)
        return ShaftInferResponse(
            text=self.payload, prompt=request.user_prompt, output_ids=[7, 8, 9]
        )


class _CancelBeforeRetryEngine:
    def validate_execution_control(self, execution) -> None:  # noqa: ANN001
        _ = execution

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        assert request.execution is not None
        assert request.execution.cancellation_event is not None
        request.execution.cancellation_event.set()
        raise RuntimeError("temporary failure")


class _RaisesCancellationEngine:
    def __init__(self):
        self.calls = 0

    def validate_execution_control(self, execution) -> None:  # noqa: ANN001
        _ = execution

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        _ = request
        self.calls += 1
        raise ShaftInferCancelledError("pipeline cancelled")


class _CountingControlEngine:
    def __init__(self):
        self.calls = 0

    def validate_execution_control(self, execution) -> None:  # noqa: ANN001
        _ = execution

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        self.calls += 1
        return ShaftInferResponse(text="unused", prompt=request.user_prompt, output_ids=[])


class _RecorderEngine:
    def __init__(self):
        self.last_request: ShaftInferRequest | None = None

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        self.last_request = request
        return ShaftInferResponse(text="ok", prompt=request.user_prompt, output_ids=[10])


def test_multistage_multi_engine_orchestration() -> None:
    pipeline = ShaftInferPipeline(
        engines={
            "det": _DummyEngine("det"),
            "struct": _DummyEngine("struct"),
        },
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
            ),
            InferStageConfig(
                name="stage2",
                engine="struct",
                output_key="struct_out",
                user_prompt_template="use {{ det_out | json }} and refine",
                arguments={"det_out": {"type": "json"}},
            ),
        ],
    )
    outputs = pipeline.run(image_path="/tmp/fake.png", inputs={"task": "arrow"})
    assert "det_out" in outputs
    assert "struct_out" in outputs
    assert outputs["det_out"].startswith("det:")
    assert outputs["det_out"] in outputs["struct_out"]
    assert len(outputs["__trace__"]) == 2


def test_stage_prompt_renderer_keeps_json_braces_literal_and_image_first_contract() -> None:
    recorder = _RecorderEngine()
    pipeline = ShaftInferPipeline(
        engines={"det": recorder},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                user_prompt_template=(
                    'Return schema {"type":"object"}; payload={{ payload | json }}'
                ),
                arguments={"payload": {"type": "json"}},
            )
        ],
    )

    pipeline.run(
        image_path="/tmp/fake.png",
        inputs={"payload": {"z": 2, "a": 1}},
    )

    assert recorder.last_request is not None
    assert recorder.last_request.image_path == "/tmp/fake.png"
    assert recorder.last_request.user_prompt == (
        'Return schema {"type":"object"}; payload={"a":1,"z":2}'
    )


def test_stage_prompt_validation_fails_before_retryable_engine_work() -> None:
    engine = _FlakyEngine(fail_times=0, payload="unused")
    pipeline = ShaftInferPipeline(
        engines={"det": engine},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                user_prompt_template="value={{ value }}",
                arguments={"value": {"type": "string"}},
                max_retries=2,
            )
        ],
    )

    with pytest.raises(ValueError, match="Missing prompt arguments.*value"):
        pipeline.run(image_path="/tmp/fake.png", inputs={})
    assert engine.calls == 0


def test_direct_pipeline_construction_rejects_legacy_format_placeholder() -> None:
    with pytest.raises(ValueError, match="legacy.*det_out.*double braces"):
        ShaftInferPipeline(
            engines={"det": _DummyEngine("det")},
            stages=[
                InferStageConfig(
                    name="stage1",
                    engine="det",
                    user_prompt_template="legacy {det_out}",
                )
            ],
        )


def test_pipeline_resolves_an_immutable_stage_snapshot() -> None:
    recorder = _RecorderEngine()
    stages = [
        InferStageConfig(
            name="stage1",
            engine="det",
            user_prompt_template="original",
        )
    ]
    pipeline = ShaftInferPipeline(engines={"det": recorder}, stages=stages)
    stages.append(InferStageConfig(name="stage2", engine="det", user_prompt_template="added"))
    exposed = pipeline.stages[0]
    exposed.user_prompt_template = "mutated"

    outputs = pipeline.run(image_path="/tmp/fake.png", inputs={})

    assert len(outputs["__trace__"]) == 1
    assert recorder.last_request is not None
    assert recorder.last_request.user_prompt == "original"


def test_stage_codec_parses_json_object() -> None:
    payload = json.dumps({"score": 0.9, "ok": True}, ensure_ascii=False)
    pipeline = ShaftInferPipeline(
        engines={"det": _FlakyEngine(fail_times=0, payload=payload)},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
                codec="json_object",
            )
        ],
    )
    outputs = pipeline.run(image_path="/tmp/fake.png", inputs={})
    assert isinstance(outputs["det_out"], dict)
    assert outputs["det_out"]["ok"] is True
    assert outputs["det_out__raw"] == payload
    trace = outputs["__trace__"][0]
    assert trace.success is True
    assert trace.codec == "json_object"


def test_stage_retry_then_success() -> None:
    pipeline = ShaftInferPipeline(
        engines={"det": _FlakyEngine(fail_times=1, payload='{"ok":1}')},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
                codec="json_object",
                max_retries=2,
                retry_backoff_seconds=0.0,
            )
        ],
    )
    outputs = pipeline.run(image_path="/tmp/fake.png", inputs={})
    assert outputs["det_out"]["ok"] == 1
    trace = outputs["__trace__"][0]
    assert trace.success is True
    assert trace.attempts == 2
    assert trace.history is not None
    assert trace.history[0].success is False
    assert trace.history[1].success is True


def test_stage_fail_fast_false_does_not_raise() -> None:
    pipeline = ShaftInferPipeline(
        engines={"det": _FlakyEngine(fail_times=3, payload="{}")},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
                codec="json_object",
                max_retries=1,
                fail_fast=False,
            )
        ],
    )
    outputs = pipeline.run(image_path="/tmp/fake.png", inputs={})
    assert outputs["det_out"] is None
    assert isinstance(outputs["det_out__error"], str)
    trace = outputs["__trace__"][0]
    assert trace.success is False
    assert trace.attempts == 2


def test_stage_fail_fast_true_raises() -> None:
    pipeline = ShaftInferPipeline(
        engines={"det": _FlakyEngine(fail_times=2, payload="{}")},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
                codec="json_object",
                max_retries=0,
                fail_fast=True,
            )
        ],
    )
    with pytest.raises(RuntimeError, match="failed after"):
        pipeline.run(image_path="/tmp/fake.png", inputs={})


def test_stage_timeout_fails_closed_when_engine_does_not_declare_support() -> None:
    pipeline = ShaftInferPipeline(
        engines={"det": _SlowEngine(sleep_seconds=0.02, payload='{"ok":1}')},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
                codec="json_object",
                timeout_seconds=0.001,
                fail_fast=False,
            )
        ],
    )
    outputs = pipeline.run(image_path="/tmp/fake.png", inputs={})
    assert outputs["det_out"] is None
    assert "does not declare execution-control support" in outputs["det_out__error"]


def test_stage_passes_absolute_deadline_to_engine() -> None:
    engine = _DeclaredDeadlineEngine(sleep_seconds=0.0, payload='{"ok":1}')
    pipeline = ShaftInferPipeline(
        engines={"det": engine},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
                codec="json_object",
                timeout_seconds=1.0,
            )
        ],
    )

    outputs = pipeline.run(image_path="/tmp/fake.png", inputs={})

    assert outputs["det_out"] == {"ok": 1}
    assert engine.validated_execution is engine.request_execution
    remaining = engine.request_execution.deadline_monotonic - time.monotonic()
    assert 0 < remaining <= 1.0


def test_stage_checks_deadline_again_after_declared_adapter_returns() -> None:
    engine = _DeclaredDeadlineEngine(sleep_seconds=0.02, payload='{"ok":1}')
    pipeline = ShaftInferPipeline(
        engines={"det": engine},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
                codec="json_object",
                timeout_seconds=0.001,
                fail_fast=False,
            )
        ],
    )

    outputs = pipeline.run(image_path="/tmp/fake.png", inputs={})

    assert outputs["det_out"] is None
    assert "deadline expired" in outputs["det_out__error"]


def test_stage_checks_deadline_after_codec_decode(monkeypatch) -> None:
    from shaft.codec import decode_with_codec as decode

    def _slow_decode(codec_name: str, text: str):
        time.sleep(0.02)
        return decode(codec_name, text)

    monkeypatch.setattr("shaft.infer.pipeline.decode_with_codec", _slow_decode)
    pipeline = ShaftInferPipeline(
        engines={"det": _DeclaredDeadlineEngine(sleep_seconds=0.0, payload='{"ok":1}')},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
                codec="json_object",
                timeout_seconds=0.001,
                fail_fast=False,
            )
        ],
    )

    outputs = pipeline.run(image_path="/tmp/fake.png", inputs={})

    assert outputs["det_out"] is None
    assert "codec decode deadline expired" in outputs["det_out__error"]


def test_pipeline_cancellation_during_codec_decode_is_terminal(monkeypatch) -> None:
    from shaft.codec import decode_with_codec as decode

    cancellation_event = threading.Event()

    def _cancelling_decode(codec_name: str, text: str):
        result = decode(codec_name, text)
        cancellation_event.set()
        return result

    monkeypatch.setattr("shaft.infer.pipeline.decode_with_codec", _cancelling_decode)
    pipeline = ShaftInferPipeline(
        engines={"det": _DeclaredDeadlineEngine(sleep_seconds=0.0, payload='{"ok":1}')},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
                codec="json_object",
                fail_fast=False,
            )
        ],
    )

    with pytest.raises(ShaftInferCancelledError, match="codec decode was cancelled"):
        pipeline.run(
            image_path="/tmp/fake.png",
            inputs={},
            cancellation_event=cancellation_event,
        )


def test_pipeline_cancellation_contract_fails_closed_before_unknown_engine_work() -> None:
    recorder = _RecorderEngine()
    pipeline = ShaftInferPipeline(
        engines={"det": recorder},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
                fail_fast=False,
            )
        ],
    )

    outputs = pipeline.run(
        image_path="/tmp/fake.png",
        inputs={},
        cancellation_event=threading.Event(),
    )

    assert outputs["det_out"] is None
    assert "does not declare execution-control support" in outputs["det_out__error"]
    assert recorder.last_request is None


def test_cancellation_only_contract_interrupts_retry_backoff() -> None:
    cancellation_event = threading.Event()
    pipeline = ShaftInferPipeline(
        engines={"det": _CancelBeforeRetryEngine()},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
                max_retries=1,
                retry_backoff_seconds=10.0,
                fail_fast=False,
            )
        ],
    )
    started = time.monotonic()

    with pytest.raises(ShaftInferCancelledError, match="cancelled"):
        pipeline.run(
            image_path="/tmp/fake.png",
            inputs={},
            cancellation_event=cancellation_event,
        )

    assert time.monotonic() - started < 1.0


def test_pipeline_cancellation_never_continues_to_later_stage() -> None:
    first = _RaisesCancellationEngine()
    second = _CountingControlEngine()
    pipeline = ShaftInferPipeline(
        engines={"first": first, "second": second},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="first",
                user_prompt_template="first",
                fail_fast=False,
            ),
            InferStageConfig(
                name="stage2",
                engine="second",
                user_prompt_template="second",
                fail_fast=False,
            ),
        ],
    )

    with pytest.raises(ShaftInferCancelledError, match="pipeline cancelled"):
        pipeline.run(
            image_path="/tmp/fake.png",
            cancellation_event=threading.Event(),
        )

    assert first.calls == 1
    assert second.calls == 0


def test_stage_runtime_overrides_are_passed_to_engine_request() -> None:
    recorder = _RecorderEngine()
    pipeline = ShaftInferPipeline(
        engines={"det": recorder},
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
                codec="text",
                min_pixels=200704,
                max_pixels=1048576,
                backend_options={"seed": 7},
            )
        ],
    )
    outputs = pipeline.run(image_path="/tmp/fake.png", inputs={})
    assert outputs["det_out"] == "ok"
    assert recorder.last_request is not None
    assert recorder.last_request.min_pixels == 200704
    assert recorder.last_request.max_pixels == 1048576
    assert recorder.last_request.backend_options == {"seed": 7}
