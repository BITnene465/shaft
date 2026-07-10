from __future__ import annotations

from typing import Any

from .base import ShaftChatTemplate
from .rendering import ShaftChatRenderer


class ShaftDelimitedChatTemplate(ShaftChatTemplate):
    assistant_start: str
    message_end: str

    @staticmethod
    def _find_token_sequence(
        token_ids: tuple[int, ...],
        pattern: tuple[int, ...],
        *,
        start: int,
    ) -> int | None:
        if not pattern:
            raise ValueError("Supervision delimiter must not tokenize to an empty sequence.")
        last_start = len(token_ids) - len(pattern)
        for index in range(start, last_start + 1):
            if token_ids[index : index + len(pattern)] == pattern:
                return index
        return None

    def _build_trainable_prefix_spans(
        self,
        *,
        messages: list[dict[str, Any]],
        assistant_indices: list[int],
        rendered_prefix_token_ids: tuple[int, ...],
        renderer: ShaftChatRenderer,
    ) -> tuple[tuple[int, int], ...]:
        _ = messages
        assistant_start_ids = renderer.tokenize(self.assistant_start)
        message_end_ids = renderer.tokenize(self.message_end)
        spans: list[tuple[int, int]] = []
        cursor = 0
        while True:
            start = self._find_token_sequence(
                rendered_prefix_token_ids,
                assistant_start_ids,
                start=cursor,
            )
            if start is None:
                break
            end_start = self._find_token_sequence(
                rendered_prefix_token_ids,
                message_end_ids,
                start=start + len(assistant_start_ids),
            )
            if end_start is None:
                # The final generation prompt is an open assistant segment and belongs
                # to target_text, not to historical prefix supervision.
                break
            end = end_start + len(message_end_ids)
            spans.append((start, end))
            cursor = end

        if len(spans) != len(assistant_indices):
            raise ValueError(
                f"{type(self).__name__} assistant span count does not match normalized messages: "
                f"expected {len(assistant_indices)}, resolved {len(spans)}."
            )
        return tuple(spans)
