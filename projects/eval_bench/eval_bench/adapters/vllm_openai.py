from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
import time
from typing import Any
from urllib import request


@dataclass(frozen=True)
class GeneratedText:
    text: str
    latency_ms: float
    raw_response: dict[str, Any]


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
    ) -> GeneratedText:
        payload = {
            "model": self.served_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
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
        )


def _chat_completions_url(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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
