from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class Template(ABC):
    def __init__(self, template_meta: "TemplateMeta") -> None:
        self.template_meta = template_meta
        self.name = template_meta.template_type

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = list(messages)
        if self.template_meta.default_system and not any(str(msg.get("role", "")).strip().lower() == "system" for msg in normalized):
            normalized = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.template_meta.default_system}],
                },
                *normalized,
            ]
        return normalized

    @abstractmethod
    def apply_chat_template(
        self,
        *,
        processor: Any,
        tokenizer: Any,
        messages: list[dict[str, Any]],
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def decode(self, *, tokenizer: Any, token_ids: list[int]) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class TemplateMeta:
    template_type: str
    template_cls: type[Template] | None
    default_system: str | None = None
    stop_words: tuple[str, ...] = field(default_factory=tuple)
    support_multi_round: bool = True
    auto_add_generation_prompt: bool = True
    response_prefix: str = ""
    thinking_prefix: str = ""
