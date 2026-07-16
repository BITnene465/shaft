from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from shaft.utils.qwen_pixel_budget import image_to_data_url_with_qwen_pixel_budget

from .inference import ShaftImageTextInferencePolicy, ShaftPreparedOpenAIInference


@dataclass(frozen=True)
class QwenVLInferencePolicy(ShaftImageTextInferencePolicy):
    """Qwen VL media and chat-template behavior shared by local and OpenAI backends."""

    supports_pixel_budget: bool = True
    supports_thinking_templates: bool = False

    def prepare_openai(
        self,
        *,
        image_path: str,
        user_prompt: str,
        system_prompt: str,
        messages: list[dict[str, Any]] | None,
        min_pixels: int | None,
        max_pixels: int | None,
        backend_options: dict[str, Any] | None,
        template_type: str,
    ) -> ShaftPreparedOpenAIInference:
        self._validate_pixel_budget(min_pixels=min_pixels, max_pixels=max_pixels)
        options = self._prepare_backend_options(
            backend_options=backend_options,
            template_type=template_type,
        )
        if messages is not None:
            if min_pixels is not None or max_pixels is not None:
                raise ValueError(
                    "Qwen pixel budget cannot be applied safely to caller-supplied messages; "
                    "omit messages or pre-encode media without min_pixels/max_pixels."
                )
            prepared_messages = copy.deepcopy(messages)
        else:
            data_url, _ = image_to_data_url_with_qwen_pixel_budget(
                image_path,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            prepared_messages = _openai_messages(
                data_url=data_url,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )
        return ShaftPreparedOpenAIInference(
            messages=prepared_messages,
            backend_options=options,
        )

    def _prepare_backend_options(
        self,
        *,
        backend_options: dict[str, Any] | None,
        template_type: str,
    ) -> dict[str, Any]:
        options = copy.deepcopy(backend_options or {})
        blocked = {
            "min_pixels",
            "min-pixels",
            "max_pixels",
            "max-pixels",
            "mm_processor_kwargs",
            "mm-processor-kwargs",
        }
        for key, value in options.items():
            if value in (None, "", False):
                continue
            if str(key).strip().lower() in blocked:
                raise ValueError(
                    f"backend_options must not set {key!r}; Qwen pixel budget is applied "
                    "by the model inference policy before the request."
                )
        if self.supports_thinking_templates and "chat_template_kwargs" not in options:
            thinking = str(template_type).strip().lower() == "qwen35vl_thinking"
            options["chat_template_kwargs"] = {
                "enable_thinking": thinking,
                "preserve_thinking": thinking,
            }
        return options


def _openai_messages(
    *,
    data_url: str,
    user_prompt: str,
    system_prompt: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_prompt},
            ],
        }
    )
    return messages
