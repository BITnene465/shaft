from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
from urllib import request

from shaft.utils.qwen_pixel_budget import image_to_data_url_with_qwen_pixel_budget


@dataclass(frozen=True)
class GeneratedText:
    text: str
    latency_ms: float
    raw_response: dict[str, Any]
    image_request: dict[str, Any]
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


class OpenAICompatibleVLLMAdapter:
    def __init__(
        self,
        *,
        endpoint: str,
        served_model_name: str,
        api_key: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.endpoint = _chat_completions_url(endpoint)
        self.served_model_name = served_model_name
        self.api_key = api_key
        self.timeout_s = timeout_s

    def generate(
        self,
        *,
        image_path: Path,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> GeneratedText:
        image_data_url, image_budget = image_to_data_url_with_qwen_pixel_budget(
            image_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        payload = {
            "model": self.served_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if extra_body:
            for key, value in extra_body.items():
                normalized_key = str(key)
                if normalized_key == "chat_template_kwargs":
                    payload[normalized_key] = value
                    continue
                if normalized_key in payload:
                    continue
                payload[normalized_key] = value
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(self.endpoint, data=body, headers=headers, method="POST")
        start = time.perf_counter()
        with request.urlopen(req, timeout=self.timeout_s) as response:
            raw_response = json.loads(response.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - start) * 1000.0
        return GeneratedText(
            text=_extract_text(raw_response),
            latency_ms=latency_ms,
            raw_response=raw_response,
            image_request=image_budget.to_dict(),
            finish_reason=_extract_finish_reason(raw_response),
            usage=_extract_usage(raw_response),
        )


def _chat_completions_url(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            return "".join(chunks)
    text = first.get("text")
    return text if isinstance(text, str) else ""


def _extract_finish_reason(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    value = first.get("finish_reason")
    return value if isinstance(value, str) else None


def _extract_usage(payload: dict[str, Any]) -> dict[str, Any] | None:
    usage = payload.get("usage")
    return dict(usage) if isinstance(usage, dict) else None
