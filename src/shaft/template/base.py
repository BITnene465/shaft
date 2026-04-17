from __future__ import annotations

from typing import Any

from .types import Template


class ShaftChatTemplate(Template):
    name = "shaft_chat"

    def __init__(self, template_meta):
        super().__init__(template_meta)

    def apply_chat_template(
        self,
        *,
        processor: Any,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool | None = None,
    ) -> str:
        owner = processor if hasattr(processor, "apply_chat_template") else tokenizer
        normalized_messages = self.prepare_messages(messages)
        resolved_add_generation_prompt = (
            self.template_meta.auto_add_generation_prompt
            if add_generation_prompt is None
            else bool(add_generation_prompt)
        )
        return owner.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=resolved_add_generation_prompt,
        )

    def decode(self, *, tokenizer: Any, token_ids: list[int]) -> str:
        if hasattr(tokenizer, "decode"):
            return str(tokenizer.decode(token_ids, skip_special_tokens=True)).strip()
        if hasattr(tokenizer, "batch_decode"):
            decoded = tokenizer.batch_decode([token_ids], skip_special_tokens=True)
            if decoded:
                return str(decoded[0]).strip()
        return " ".join(str(x) for x in token_ids)
