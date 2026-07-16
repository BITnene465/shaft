from __future__ import annotations

from .delimited import ShaftDelimitedChatTemplate


class QwenChatTemplate(ShaftDelimitedChatTemplate):
    assistant_start = "<|im_start|>assistant\n"
    message_end = "<|im_end|>\n"
