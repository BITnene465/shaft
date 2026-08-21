from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from shaft.utils.qwen_pixel_budget import image_to_data_url_with_qwen_pixel_budget
from shaft.utils.messages import count_message_content_type

from .inference import ShaftImageTextInferencePolicy, ShaftPreparedOpenAIInference


@dataclass(frozen=True)
class QwenVLInferencePolicy(ShaftImageTextInferencePolicy):
    """Qwen VL media and chat-template behavior shared by local and OpenAI backends."""

    supports_pixel_budget: bool = True
    supports_thinking_templates: bool = False

    def prepare_openai(
        self,
        *,
        image_paths: tuple[str, ...],
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
        prepared_messages = None
        if messages is not None:
            prepared_messages = copy.deepcopy(messages)
            _validate_openai_image_placeholders(
                messages=prepared_messages,
                image_count=len(image_paths),
            )
        data_urls = []
        for image_path in image_paths:
            data_url, _ = image_to_data_url_with_qwen_pixel_budget(
                image_path,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            data_urls.append(data_url)
        if prepared_messages is not None:
            prepared_messages = _replace_openai_image_placeholders(
                messages=prepared_messages,
                data_urls=tuple(data_urls),
            )
        else:
            prepared_messages = _openai_messages(
                data_urls=tuple(data_urls),
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
            from shaft.template import build_template_meta

            template_meta = build_template_meta(str(template_type).strip().lower())
            chat_template_options = dict(template_meta.chat_template_options)
            if not chat_template_options:
                raise ValueError(
                    f"Template {template_type!r} does not declare Qwen chat-template options."
                )
            options["chat_template_kwargs"] = chat_template_options
        return options


def _openai_messages(
    *,
    data_urls: tuple[str, ...],
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
                *(
                    {"type": "image_url", "image_url": {"url": data_url}}
                    for data_url in data_urls
                ),
                {"type": "text", "text": user_prompt},
            ],
        }
    )
    return messages


def _replace_openai_image_placeholders(
    *,
    messages: list[dict[str, Any]],
    data_urls: tuple[str, ...],
) -> list[dict[str, Any]]:
    cursor = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            continue
        replaced: list[Any] = []
        for item in content:
            if not isinstance(item, dict):
                replaced.append(item)
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type == "image_url":
                raise ValueError(
                    "Caller-supplied image_url content cannot be combined with image_paths; "
                    "use ordered type='image' placeholders as the single media source."
                )
            if item_type != "image":
                replaced.append(item)
                continue
            if cursor >= len(data_urls):
                raise ValueError(
                    "Inference message image placeholder count exceeds ordered image_paths."
                )
            replaced.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_urls[cursor]},
                }
            )
            cursor += 1
        message["content"] = replaced
    if cursor != len(data_urls):
        raise ValueError(
            "Inference message image placeholder count must match ordered image_paths: "
            f"placeholders={cursor}, images={len(data_urls)}."
        )
    return messages


def _validate_openai_image_placeholders(
    *,
    messages: list[dict[str, Any]],
    image_count: int,
) -> None:
    image_url_count = count_message_content_type(messages, "image_url")
    if image_url_count:
        raise ValueError(
            "Caller-supplied image_url content cannot be combined with image_paths; "
            "use ordered type='image' placeholders as the single media source."
        )
    placeholder_count = count_message_content_type(messages, "image")
    if placeholder_count != int(image_count):
        raise ValueError(
            "Inference message image placeholder count must match ordered image_paths: "
            f"placeholders={placeholder_count}, images={image_count}."
        )
