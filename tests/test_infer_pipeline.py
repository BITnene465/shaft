from __future__ import annotations

import json
import time

import pytest

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
        return ShaftInferResponse(text=self.payload, prompt=request.user_prompt, output_ids=[4, 5, 6])


class _SlowEngine:
    def __init__(self, *, sleep_seconds: float, payload: str):
        self.sleep_seconds = float(sleep_seconds)
        self.payload = payload

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        time.sleep(self.sleep_seconds)
        return ShaftInferResponse(text=self.payload, prompt=request.user_prompt, output_ids=[7, 8, 9])


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
                user_prompt_template="use {det_out} and refine",
            ),
        ],
    )
    outputs = pipeline.run(image_path="/tmp/fake.png", inputs={"task": "arrow"})
    assert "det_out" in outputs
    assert "struct_out" in outputs
    assert outputs["det_out"].startswith("det:")
    assert outputs["det_out"] in outputs["struct_out"]
    assert len(outputs["__trace__"]) == 2


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


def test_stage_timeout_marks_failure_when_not_fail_fast() -> None:
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
    assert "timeout" in outputs["det_out__error"].lower()


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
