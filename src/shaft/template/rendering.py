from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ShaftChatRenderer:
    __slots__ = ("_render", "_tokenize")

    def __init__(
        self,
        *,
        render: Callable[..., Any],
        tokenize: Callable[..., Any],
    ) -> None:
        self._render = render
        self._tokenize = tokenize

    @classmethod
    def from_components(cls, *, processor: Any, tokenizer: Any) -> "ShaftChatRenderer":
        owner = processor if hasattr(processor, "apply_chat_template") else tokenizer
        return cls(render=owner.apply_chat_template, tokenize=tokenizer)

    def render(
        self,
        *,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
        options: dict[str, Any] | None = None,
    ) -> str:
        return str(
            self._render(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **dict(options or {}),
            )
        )

    def tokenize(self, text: str) -> tuple[int, ...]:
        tokenized = self._tokenize(
            [text],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return tuple(int(token_id) for token_id in tokenized["input_ids"][0])
