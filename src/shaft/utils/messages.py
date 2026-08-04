from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def iter_message_content_dicts(
    messages: Iterable[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    """Yield structured content chunks without assuming every message uses list content."""

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, dict):
            candidates = (content,)
        elif isinstance(content, (list, tuple)):
            candidates = content
        else:
            # OpenAI-compatible system/user text commonly uses a plain string.
            continue
        for item in candidates:
            if isinstance(item, dict):
                yield item


def count_message_content_type(
    messages: Iterable[dict[str, Any]],
    content_type: str,
) -> int:
    expected = str(content_type).strip().lower()
    return sum(
        1
        for item in iter_message_content_dicts(messages)
        if str(item.get("type", "")).strip().lower() == expected
    )
