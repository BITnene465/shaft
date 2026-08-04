from __future__ import annotations

from typing import Any

from .base import ShaftChatTemplate
from .rendering import ShaftChatRenderer


class ShaftDelimitedChatTemplate(ShaftChatTemplate):
    assistant_start: str
    message_start_format: str
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

    def _build_truncatable_prefix_spans(
        self,
        *,
        messages: list[dict[str, Any]],
        rendered_prefix_token_ids: tuple[int, ...],
        renderer: ShaftChatRenderer,
    ) -> tuple[tuple[int, int], ...]:
        message_end_ids = renderer.tokenize(self.message_end)
        spans: list[tuple[int, int]] = []
        cursor = 0
        for message in messages:
            role = str(message.get("role", "user")).strip().lower() or "user"
            message_start_ids = renderer.tokenize(
                self.message_start_format.format(role=role)
            )
            start = self._find_token_sequence(
                rendered_prefix_token_ids,
                message_start_ids,
                start=cursor,
            )
            if start is None:
                # Some synthetic or external renderers do not implement this template's
                # delimiters. Such prefixes remain usable but cannot be safely truncated.
                return ()
            content_start = start + len(message_start_ids)
            end_start = self._find_token_sequence(
                rendered_prefix_token_ids,
                message_end_ids,
                start=content_start,
            )
            if end_start is None:
                return ()
            if content_start < end_start:
                spans.append((content_start, end_start))
            cursor = end_start + len(message_end_ids)
        return tuple(spans)
