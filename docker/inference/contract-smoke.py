#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from openai import OpenAI
import yaml

from shaft.codec import decode_with_codec
from shaft.utils.qwen_pixel_budget import image_to_data_url_with_qwen_pixel_budget


def _load_prompt(path: Path, prompt_id: str) -> tuple[str | None, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts") or []
    for prompt in prompts:
        if prompt.get("id") == prompt_id:
            return prompt.get("system_prompt"), prompt["user_prompt"]
    raise SystemExit(f"prompt id not found: {prompt_id} in {path}")


def _message_payload(system_prompt: str | None, user_prompt: str, image_url: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    )
    return messages


def _codec_contract(raw_text: str) -> dict[str, Any]:
    decoded = decode_with_codec("json_any", raw_text)
    return {
        "codec": "json_any",
        "valid": decoded.valid,
        "partial": decoded.partial,
        "error_type": decoded.error_type,
        "error": decoded.error,
        "parsed": decoded.parsed if decoded.valid else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Shaft inference contract smoke request.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="banana_v4_1_step8000")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--prompt-id", default="main")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=2_000_000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()

    prompt_path = Path(args.prompt_path)
    system_prompt, user_prompt = _load_prompt(prompt_path, args.prompt_id)
    image_url, budget = image_to_data_url_with_qwen_pixel_budget(
        args.image,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    messages = _message_payload(system_prompt, user_prompt, image_url)

    client = OpenAI(base_url=args.endpoint, api_key="EMPTY")
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    raw_text = response.choices[0].message.content or ""
    finish_reason = response.choices[0].finish_reason

    payload = {
        "endpoint": args.endpoint,
        "model": args.model,
        "image": str(Path(args.image).resolve()),
        "prompt_path": str(prompt_path.resolve()),
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(
            ((system_prompt or "") + "\n" + user_prompt).encode("utf-8")
        ).hexdigest(),
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "pixel_budget": budget.to_dict(),
        "finish_reason": finish_reason,
        "raw_text": raw_text,
        "parser": _codec_contract(raw_text),
        "usage": response.usage.model_dump() if response.usage else None,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output == "-":
        print(text)
    else:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
