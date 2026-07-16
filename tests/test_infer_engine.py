from __future__ import annotations

import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading
import time
from types import SimpleNamespace
import urllib.request
import base64
from io import BytesIO

from PIL import Image
import pytest
import torch

from shaft.infer import (
    InferEngineConfig,
    InferGenerationConfig,
    InferPipelineConfig,
    InferStageConfig,
    ShaftInferCancelledError,
    ShaftInferExecutionControl,
    ShaftInferExecutionControlUnsupportedError,
    ShaftInferEngine,
    ShaftInferPipeline,
    ShaftInferRequest,
)
import shaft.infer.engine as infer_engine_module
from shaft.infer.engine import VLLMOpenAIInferAdapter
from shaft.model import ModelMeta, ShaftImageTextInferencePolicy


class _JSONHTTPResponse:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload
        self._read = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = exc_type, exc, tb
        return False

    def read(self, size: int = -1) -> bytes:
        _ = size
        if self._read:
            return b""
        self._read = True
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def test_model_owned_local_policy_keeps_image_first(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image)
    captured: dict[str, object] = {}

    class _CapturingTemplate:
        def apply_chat_template(self, *, renderer, messages):  # noqa: ANN001
            captured["renderer"] = renderer
            captured["messages"] = messages
            return "rendered"

    prepared = ShaftImageTextInferencePolicy().prepare_local(
        image_path=str(image),
        system_prompt="system",
        user_prompt="dynamic text",
        messages=None,
        min_pixels=None,
        max_pixels=None,
        backend_options=None,
        template=_CapturingTemplate(),
        renderer="renderer",
    )

    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0] == {"type": "image"}
    assert messages[1]["content"][1] == {"type": "text", "text": "dynamic text"}
    assert prepared.prompt == "rendered"


def test_smoke_vlm_engine_can_generate(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image)

    model_config = InferEngineConfig(
        model_type="smoke_vlm",
        model_name_or_path="unused",
        device="cpu",
        generation=InferGenerationConfig(max_new_tokens=8, do_sample=False),
    )
    engine = ShaftInferEngine.from_engine_config(model_config)
    response = engine.run(
        ShaftInferRequest(
            image_path=str(image),
            system_prompt="you are tester",
            user_prompt="return json",
        )
    )
    assert isinstance(response.text, str)
    assert "return json" in response.prompt
    assert isinstance(response.output_ids, list)


def test_smoke_vlm_local_policy_fails_closed_for_pixel_budget(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image)
    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="smoke_vlm",
            model_name_or_path="unused",
            device="cpu",
        )
    )

    with pytest.raises(ValueError, match="does not support min_pixels/max_pixels"):
        engine.run(
            ShaftInferRequest(
                image_path=str(image),
                user_prompt="return json",
                max_pixels=1024,
            )
        )


def test_infer_engine_generate_does_not_emit_invalid_sampling_flags_for_greedy() -> None:

    model_config = InferEngineConfig(
        model_type="smoke_vlm",
        model_name_or_path="unused",
        device="cpu",
        generation=InferGenerationConfig(max_new_tokens=8, do_sample=False),
    )
    engine = ShaftInferEngine.from_engine_config(model_config)
    adapter = engine.adapter

    captured = {}

    class DummyGenerationConfig(SimpleNamespace):
        def clone(self) -> "DummyGenerationConfig":
            return DummyGenerationConfig(**self.__dict__)

    class DummyModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=False)
            self.generation_config = DummyGenerationConfig(
                use_cache=False,
                max_new_tokens=16,
                do_sample=True,
                top_p=0.95,
                temperature=0.5,
                top_k=7,
                repetition_penalty=1.2,
                eos_token_id=adapter.tokenizer.eos_token_id,
                pad_token_id=adapter.tokenizer.pad_token_id,
            )

        def to(self, _device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            captured["kwargs"] = kwargs
            return torch.tensor([[1, 2, 3]])

    adapter.model = DummyModel()
    batch = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }
    adapter._generate(
        batch=batch, generation=InferGenerationConfig(max_new_tokens=4, do_sample=False)
    )

    gen_config = captured["kwargs"]["generation_config"]
    assert adapter.model.config.use_cache is True
    assert adapter.model.generation_config.use_cache is True
    assert gen_config.use_cache is True
    assert gen_config.do_sample is False
    assert gen_config.top_p == 1.0
    assert gen_config.top_k == 50
    assert gen_config.temperature == 1.0


def test_hf_local_infer_engine_passes_device_map_to_model_loader(monkeypatch) -> None:
    captured = {}
    real_builder = infer_engine_module.build_model_tokenizer_processor

    def fake_builder(runtime_config):  # noqa: ANN001
        captured["device_map"] = runtime_config.model.device_map
        return real_builder(runtime_config)

    monkeypatch.setattr(infer_engine_module, "build_model_tokenizer_processor", fake_builder)

    _ = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="smoke_vlm",
            model_name_or_path="unused",
            device="cpu",
            device_map="auto",
            generation=InferGenerationConfig(max_new_tokens=8, do_sample=False),
        )
    )

    assert captured["device_map"] == "auto"


def test_vllm_openai_engine_can_generate(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image)
    captured: dict[str, object] = {}

    payload = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"ok": true}'},
            }
        ],
    }

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        assert isinstance(request, urllib.request.Request)
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        captured["body"] = json.loads((request.data or b"{}").decode("utf-8"))
        return _JSONHTTPResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    model_config = InferEngineConfig(
        model_type="qwen3vl",
        model_name_or_path="arrow_mixed_4b",
        backend="vllm_openai",
        endpoint="http://127.0.0.1:8001/v1/chat/completions",
        request_timeout_seconds=12.5,
        api_key="test-key",
        generation=InferGenerationConfig(max_new_tokens=16, do_sample=False),
    )
    engine = ShaftInferEngine.from_engine_config(model_config)
    response = engine.run(
        ShaftInferRequest(
            image_path=str(image),
            system_prompt="you are tester",
            user_prompt="return json",
            min_pixels=200704,
            max_pixels=1048576,
            backend_options={"seed": 11},
        )
    )
    assert response.text == '{"ok": true}'
    assert response.backend == "vllm_openai"
    assert response.output_ids == []

    assert captured["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert 0 < float(captured["timeout"]) <= 12.5
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer test-key"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "arrow_mixed_4b"
    assert body["max_tokens"] == 16
    assert body["temperature"] == 0.0
    assert "mm_processor_kwargs" not in body
    assert body["seed"] == 11
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["content"][0]["type"] == "image_url"
    assert body["messages"][1]["content"][1] == {
        "type": "text",
        "text": "return json",
    }


def test_vllm_openai_qwen35vl_disables_thinking_by_default(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image)
    captured: dict[str, object] = {}

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["body"] = json.loads((request.data or b"{}").decode("utf-8"))
        return _JSONHTTPResponse({"choices": [{"message": {"role": "assistant", "content": "{}"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="qwen36vl",
            model_name_or_path="models/Qwen3.6-27B",
            backend="vllm_openai",
            endpoint="http://127.0.0.1:8001",
        )
    )
    _ = engine.run(ShaftInferRequest(image_path=str(image), user_prompt="return json"))

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }


def test_vllm_openai_qwen35vl_allows_chat_template_kwargs_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image)
    captured: dict[str, object] = {}

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["body"] = json.loads((request.data or b"{}").decode("utf-8"))
        return _JSONHTTPResponse({"choices": [{"message": {"role": "assistant", "content": "{}"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="qwen36vl",
            model_name_or_path="models/Qwen3.6-27B",
            template="qwen35vl_thinking",
            backend="vllm_openai",
            endpoint="http://127.0.0.1:8001",
        )
    )
    _ = engine.run(
        ShaftInferRequest(
            image_path=str(image),
            user_prompt="return json",
            backend_options={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "preserve_thinking": False,
                }
            },
        )
    )

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }


def test_vllm_openai_engine_resizes_image_before_request(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "large.jpg"
    Image.new("RGB", (4096, 2748), color=(255, 255, 255)).save(image)
    captured: dict[str, object] = {}

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        assert isinstance(request, urllib.request.Request)
        captured["body"] = json.loads((request.data or b"{}").decode("utf-8"))
        return _JSONHTTPResponse({"choices": [{"message": {"role": "assistant", "content": "[]"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="qwen3vl",
            model_name_or_path="banana",
            backend="vllm_openai",
            endpoint="http://127.0.0.1:8001",
            generation=InferGenerationConfig(max_new_tokens=16, do_sample=False),
        )
    )
    _ = engine.run(
        ShaftInferRequest(
            image_path=str(image),
            user_prompt="return json",
            max_pixels=1_000_000,
        )
    )

    body = captured["body"]
    assert isinstance(body, dict)
    image_url = body["messages"][0]["content"][0]["image_url"]["url"]
    encoded = image_url.split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as sent:
        assert sent.size == (1216, 800)
    assert "mm_processor_kwargs" not in body


def test_vllm_openai_engine_rejects_pixel_budget_backend_options(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image)

    def _unexpected_urlopen(request, timeout=0):  # noqa: ANN001
        raise AssertionError("request should fail before HTTP call")

    monkeypatch.setattr(urllib.request, "urlopen", _unexpected_urlopen)

    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="qwen3vl",
            model_name_or_path="banana",
            backend="vllm_openai",
            endpoint="http://127.0.0.1:8001/v1/chat/completions",
        )
    )

    with pytest.raises(ValueError, match="model inference policy"):
        engine.run(
            ShaftInferRequest(
                image_path=str(image),
                user_prompt="return json",
                backend_options={"mm_processor_kwargs": {"max_pixels": 1_000_000}},
            )
        )


def test_vllm_stage_deadline_shortens_http_timeout_and_surfaces_timeout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image)
    captured: dict[str, float] = {}

    def _blocking_urlopen(request, timeout=0):  # noqa: ANN001
        _ = request
        captured["timeout"] = float(timeout)
        raise socket.timeout("blocked")

    monkeypatch.setattr(urllib.request, "urlopen", _blocking_urlopen)
    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="qwen3vl",
            model_name_or_path="banana",
            backend="vllm_openai",
            endpoint="http://127.0.0.1:8001",
            request_timeout_seconds=60.0,
        )
    )
    control = ShaftInferExecutionControl(deadline_monotonic=time.monotonic() + 0.25)

    with pytest.raises(TimeoutError, match="deadline expired"):
        engine.run(
            ShaftInferRequest(
                image_path=str(image),
                user_prompt="return json",
                execution=control,
            )
        )

    assert 0 < captured["timeout"] <= 0.25


def test_vllm_stage_deadline_interrupts_stalled_partial_http_body(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image)
    partial_sent = threading.Event()
    release_handler = threading.Event()

    class _StalledBodyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length:
                self.rfile.read(content_length)
            partial = b'{"choices":[{"message":{"content":"'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(partial) + 1024))
            self.end_headers()
            self.wfile.write(partial)
            self.wfile.flush()
            partial_sent.set()
            release_handler.wait(timeout=5.0)

        def log_message(self, format, *args) -> None:  # noqa: A002, ANN001
            _ = format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), _StalledBodyHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        pipeline = ShaftInferPipeline.from_config(
            InferPipelineConfig(
                engines={
                    "remote": InferEngineConfig(
                        model_type="qwen3vl",
                        model_name_or_path="banana",
                        backend="vllm_openai",
                        endpoint=endpoint,
                        request_timeout_seconds=5.0,
                    )
                },
                stages=[
                    InferStageConfig(
                        name="stage1",
                        engine="remote",
                        output_key="out",
                        user_prompt_template="return json",
                        timeout_seconds=0.5,
                        fail_fast=False,
                    )
                ],
            )
        )
        started = time.monotonic()

        outputs = pipeline.run(image_path=str(image))

        elapsed = time.monotonic() - started
    finally:
        release_handler.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)

    assert partial_sent.is_set()
    assert elapsed < 2.0
    assert outputs["out"] is None
    assert "deadline expired" in outputs["out__error"]


def test_vllm_expired_deadline_fails_before_http(monkeypatch) -> None:
    def _unexpected_urlopen(request, timeout=0):  # noqa: ANN001
        raise AssertionError("expired request must fail before HTTP")

    monkeypatch.setattr(urllib.request, "urlopen", _unexpected_urlopen)
    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="qwen3vl",
            model_name_or_path="banana",
            backend="vllm_openai",
            endpoint="http://127.0.0.1:8001",
        )
    )

    with pytest.raises(TimeoutError, match="deadline expired"):
        engine.run(
            ShaftInferRequest(
                image_path="/does/not/matter.png",
                user_prompt="return json",
                execution=ShaftInferExecutionControl(deadline_monotonic=time.monotonic() - 1.0),
            )
        )


def test_hf_local_deadline_fails_closed_before_model_work() -> None:
    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="smoke_vlm",
            model_name_or_path="unused",
            device="cpu",
        )
    )

    with pytest.raises(
        ShaftInferExecutionControlUnsupportedError,
        match="cannot honor an absolute deadline",
    ):
        engine.run(
            ShaftInferRequest(
                image_path="/does/not/matter.png",
                user_prompt="return json",
                execution=ShaftInferExecutionControl(deadline_monotonic=time.monotonic() + 10.0),
            )
        )


def test_hf_local_cancellation_fails_closed_before_model_work() -> None:
    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="smoke_vlm",
            model_name_or_path="unused",
            device="cpu",
        )
    )

    with pytest.raises(
        ShaftInferExecutionControlUnsupportedError,
        match="cannot honor cooperative cancellation",
    ):
        engine.run(
            ShaftInferRequest(
                image_path="/does/not/matter.png",
                user_prompt="return json",
                execution=ShaftInferExecutionControl(cancellation_event=threading.Event()),
            )
        )


def test_hf_local_pre_cancelled_control_precedes_capability_rejection() -> None:
    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="smoke_vlm",
            model_name_or_path="unused",
            device="cpu",
        )
    )
    cancellation_event = threading.Event()
    cancellation_event.set()

    with pytest.raises(ShaftInferCancelledError, match="was cancelled"):
        engine.run(
            ShaftInferRequest(
                image_path="/does/not/matter.png",
                execution=ShaftInferExecutionControl(cancellation_event=cancellation_event),
            )
        )


def test_hf_local_expired_deadline_precedes_capability_rejection() -> None:
    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="smoke_vlm",
            model_name_or_path="unused",
            device="cpu",
        )
    )

    with pytest.raises(TimeoutError, match="deadline expired"):
        engine.run(
            ShaftInferRequest(
                image_path="/does/not/matter.png",
                execution=ShaftInferExecutionControl(deadline_monotonic=time.monotonic() - 1.0),
            )
        )


def test_vllm_cancellation_fails_closed_before_http(monkeypatch) -> None:
    def _unexpected_urlopen(request, timeout=0):  # noqa: ANN001
        raise AssertionError("unsupported cancellation must fail before HTTP")

    monkeypatch.setattr(urllib.request, "urlopen", _unexpected_urlopen)
    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="qwen3vl",
            model_name_or_path="banana",
            backend="vllm_openai",
            endpoint="http://127.0.0.1:8001",
        )
    )

    with pytest.raises(
        ShaftInferExecutionControlUnsupportedError,
        match="cannot honor cooperative cancellation",
    ):
        engine.run(
            ShaftInferRequest(
                image_path="/does/not/matter.png",
                execution=ShaftInferExecutionControl(cancellation_event=threading.Event()),
            )
        )


def test_unregistered_model_inference_policy_fails_closed_before_http(monkeypatch) -> None:
    def _unexpected_urlopen(request, timeout=0):  # noqa: ANN001
        raise AssertionError("unsupported model must fail before HTTP")

    monkeypatch.setattr(urllib.request, "urlopen", _unexpected_urlopen)
    model_adapter = ModelMeta(
        model_type="unsupported",
        family="unsupported",
        default_template="unsupported",
    ).resolve_adapter(model_name_or_path="unsupported")
    adapter = VLLMOpenAIInferAdapter(
        endpoint="http://127.0.0.1:8001",
        model_name="unsupported",
        model_adapter=model_adapter,
    )

    with pytest.raises(ValueError, match="does not support the vllm_openai backend"):
        adapter.run(
            ShaftInferRequest(
                image_path="/does/not/matter.png",
                user_prompt="return json",
            )
        )
