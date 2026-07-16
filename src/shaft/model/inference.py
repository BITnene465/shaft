from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class ShaftPreparedLocalInference:
    """Model-owned inputs ready for a local processor invocation."""

    prompt: str
    image: Any
    min_pixels: int | None
    max_pixels: int | None


@dataclass(frozen=True)
class ShaftPreparedOpenAIInference:
    """Model-owned messages and optional request fields for an OpenAI backend."""

    messages: list[dict[str, Any]]
    backend_options: dict[str, Any]


@dataclass(frozen=True)
class ShaftInferencePolicy:
    """Fail-closed model inference contract.

    A model family must opt into each backend and own its media/message/template
    semantics. Generic inference adapters only execute already-prepared inputs.
    """

    def prepare_local(
        self,
        *,
        image_path: str,
        user_prompt: str,
        system_prompt: str,
        messages: list[dict[str, Any]] | None,
        min_pixels: int | None,
        max_pixels: int | None,
        backend_options: dict[str, Any] | None,
        template: Any,
        renderer: Any,
    ) -> ShaftPreparedLocalInference:
        _ = (
            image_path,
            user_prompt,
            system_prompt,
            messages,
            min_pixels,
            max_pixels,
            backend_options,
            template,
            renderer,
        )
        raise ValueError(
            f"Inference policy {type(self).__name__!r} does not support the hf_local backend. "
            "Register an explicit model-owned inference policy before using this model for inference."
        )

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
        _ = (
            image_path,
            user_prompt,
            system_prompt,
            messages,
            min_pixels,
            max_pixels,
            backend_options,
            template_type,
        )
        raise ValueError(
            f"Inference policy {type(self).__name__!r} does not support the vllm_openai backend. "
            "Register an explicit model-owned inference policy before using this model for inference."
        )


@dataclass(frozen=True)
class ShaftImageTextInferencePolicy(ShaftInferencePolicy):
    """Explicit local image-text policy used by simple HF-compatible models."""

    supports_pixel_budget: bool = False

    def prepare_local(
        self,
        *,
        image_path: str,
        user_prompt: str,
        system_prompt: str,
        messages: list[dict[str, Any]] | None,
        min_pixels: int | None,
        max_pixels: int | None,
        backend_options: dict[str, Any] | None,
        template: Any,
        renderer: Any,
    ) -> ShaftPreparedLocalInference:
        self._validate_pixel_budget(min_pixels=min_pixels, max_pixels=max_pixels)
        if backend_options:
            raise ValueError(
                "hf_local backend_options are unsupported by this model inference policy; "
                f"received keys={tuple(sorted(str(key) for key in backend_options))}."
            )
        prepared_messages = (
            copy.deepcopy(messages)
            if messages is not None
            else _local_messages(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )
        )
        prompt = template.apply_chat_template(
            renderer=renderer,
            messages=prepared_messages,
        )
        with Image.open(image_path) as image_obj:
            image = image_obj.convert("RGB")
        return ShaftPreparedLocalInference(
            prompt=prompt,
            image=image,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    def _validate_pixel_budget(
        self,
        *,
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> None:
        _validate_pixel_budget_values(min_pixels=min_pixels, max_pixels=max_pixels)
        if not self.supports_pixel_budget and (min_pixels is not None or max_pixels is not None):
            raise ValueError(
                f"Inference policy {type(self).__name__!r} does not support min_pixels/max_pixels."
            )


def _local_messages(*, user_prompt: str, system_prompt: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_prompt}],
        }
    )
    return messages


def _validate_pixel_budget_values(
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> None:
    if min_pixels is not None and int(min_pixels) <= 0:
        raise ValueError("min_pixels must be > 0.")
    if max_pixels is not None and int(max_pixels) <= 0:
        raise ValueError("max_pixels must be > 0.")
    if min_pixels is not None and max_pixels is not None:
        if int(min_pixels) > int(max_pixels):
            raise ValueError("min_pixels must be <= max_pixels.")
