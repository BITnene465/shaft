from __future__ import annotations

from collections.abc import Callable

from shaft.plugins import Registry

from .base import ShaftCodecResult

CodecFn = Callable[[str], ShaftCodecResult]

CODEC_REGISTRY: Registry = Registry("codec")


def register_codec(name: str):
    return CODEC_REGISTRY.register(str(name).strip().lower())


def decode_with_codec(codec: str, raw_text: str) -> ShaftCodecResult:
    codec_name = str(codec).strip().lower()
    decoder = CODEC_REGISTRY.get(codec_name)
    return decoder(raw_text)
