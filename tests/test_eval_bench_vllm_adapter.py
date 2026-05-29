from __future__ import annotations

import json
from pathlib import Path
import urllib.request

from PIL import Image

from eval_bench.adapters.vllm_openai import OpenAICompatibleVLLMAdapter


def test_vllm_adapter_enforces_qwen_pixel_budget_before_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "large.jpg"
    Image.new("RGB", (4096, 2748), color=(255, 255, 255)).save(image_path)
    captured: dict[str, object] = {}

    class _DummyHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def read(self) -> bytes:
            payload = {
                "choices": [{"message": {"content": "[]"}}],
            }
            return json.dumps(payload).encode("utf-8")

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001
        assert isinstance(req, urllib.request.Request)
        captured["timeout"] = timeout
        captured["body"] = json.loads((req.data or b"{}").decode("utf-8"))
        return _DummyHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    adapter = OpenAICompatibleVLLMAdapter(
        endpoint="http://127.0.0.1:8000",
        served_model_name="banana",
        timeout_s=12,
    )
    result = adapter.generate(
        image_path=image_path,
        system_prompt="system",
        user_prompt="user",
        max_tokens=128,
        temperature=0.0,
        top_p=1.0,
        top_k=20,
        max_pixels=1_000_000,
    )

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["top_k"] == 20
    image_url = body["messages"][1]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    assert "mm_processor_kwargs" not in body
    assert result.image_request["source_width"] == 4096
    assert result.image_request["source_height"] == 2748
    assert result.image_request["target_width"] == 1216
    assert result.image_request["target_height"] == 800
    assert result.image_request["target_pixels"] <= 1_000_000
    assert result.image_request["resized"] is True
