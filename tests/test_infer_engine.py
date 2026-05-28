from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request
import base64
from io import BytesIO

from PIL import Image
import torch

from shaft.infer import (
    InferEngineConfig,
    InferGenerationConfig,
    ShaftInferEngine,
    ShaftInferRequest,
)


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
            min_pixels=200704,
            max_pixels=1048576,
            backend_options={"seed": 11},
        )
    )
    assert isinstance(response.text, str)
    assert "return json" in response.prompt
    assert isinstance(response.output_ids, list)


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
    adapter._generate(batch=batch, generation=InferGenerationConfig(max_new_tokens=4, do_sample=False))

    gen_config = captured["kwargs"]["generation_config"]
    assert adapter.model.config.use_cache is True
    assert adapter.model.generation_config.use_cache is True
    assert gen_config.use_cache is True
    assert gen_config.do_sample is False
    assert gen_config.top_p == 1.0
    assert gen_config.top_k == 50
    assert gen_config.temperature == 1.0


def test_vllm_openai_engine_can_generate(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image)
    captured: dict[str, object] = {}

    class _DummyHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def read(self) -> bytes:
            payload = {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"ok": true}'},
                    }
                ],
            }
            return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        assert isinstance(request, urllib.request.Request)
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        captured["body"] = json.loads((request.data or b"{}").decode("utf-8"))
        return _DummyHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    model_config = InferEngineConfig(
        model_type="qwen3vl",
        model_name_or_path="arrow_mixed_4b",
        backend="vllm_openai",
        endpoint="http://127.0.0.1:8001",
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
    assert captured["timeout"] == 12.5
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer test-key"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "arrow_mixed_4b"
    assert body["max_tokens"] == 16
    assert body["temperature"] == 0.0
    assert body["mm_processor_kwargs"]["min_pixels"] == 200704
    assert body["mm_processor_kwargs"]["max_pixels"] == 1048576
    assert body["seed"] == 11
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["content"][0]["type"] == "image_url"


def test_vllm_openai_engine_resizes_image_before_request(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "large.jpg"
    Image.new("RGB", (4096, 2748), color=(255, 255, 255)).save(image)
    captured: dict[str, object] = {}

    class _DummyHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": "[]"}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        assert isinstance(request, urllib.request.Request)
        captured["body"] = json.loads((request.data or b"{}").decode("utf-8"))
        return _DummyHTTPResponse()

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
    assert body["mm_processor_kwargs"]["max_pixels"] == 1_000_000
