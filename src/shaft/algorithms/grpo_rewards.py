from __future__ import annotations

from collections.abc import Callable
from typing import Any

from shaft.codec import CODEC_REGISTRY
from shaft.metrics.builtin import _coerce_instances, _iou_xyxy
from shaft.plugins import Registry

GRPO_REWARD_REGISTRY: Registry[Callable[..., list[float]]] = Registry("grpo_reward")


def register_grpo_reward(name: str):
    return GRPO_REWARD_REGISTRY.register(name)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, str):
        return value.strip()
    return value


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", "")).strip()
        if "text" in content:
            return str(content.get("text", "")).strip()
        if "content" in content:
            return _message_content_to_text(content["content"])
        return ""
    if isinstance(content, list):
        return "".join(_message_content_to_text(item) for item in content).strip()
    return str(content).strip()


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        return _message_content_to_text(completion.get("content", completion))
    if isinstance(completion, list):
        if completion and all(isinstance(item, dict) and "role" in item for item in completion):
            return "".join(_message_content_to_text(item.get("content", "")) for item in completion).strip()
        return "".join(_completion_to_text(item) for item in completion).strip()
    return str(completion).strip()


def _decode_target(target_text: str, *, codec_name: str) -> Any:
    codec = CODEC_REGISTRY.get(codec_name)
    decoded = codec(str(target_text))
    if decoded.valid:
        return decoded.parsed if decoded.parsed is not None else decoded.raw_text
    return str(target_text).strip()


@register_grpo_reward("parse_success")
def grpo_reward_parse_success(
    *,
    prediction_texts: list[str],
    codec_name: str,
    **params: Any,
) -> list[float]:
    _ = params
    codec = CODEC_REGISTRY.get(codec_name)
    rewards: list[float] = []
    for text in prediction_texts:
        decoded = codec(str(text))
        rewards.append(1.0 if decoded.valid and not decoded.partial else 0.0)
    return rewards


@register_grpo_reward("exact_match")
def grpo_reward_exact_match(
    *,
    prediction_texts: list[str],
    target_texts: list[str],
    codec_name: str,
    **params: Any,
) -> list[float]:
    _ = params
    codec = CODEC_REGISTRY.get(codec_name)
    rewards: list[float] = []
    for prediction_text, target_text in zip(prediction_texts, target_texts, strict=True):
        decoded = codec(str(prediction_text))
        if not decoded.valid or decoded.partial:
            rewards.append(0.0)
            continue
        prediction_value = decoded.parsed if decoded.parsed is not None else decoded.raw_text
        target_value = _decode_target(str(target_text), codec_name=codec_name)
        rewards.append(1.0 if _normalize_value(prediction_value) == _normalize_value(target_value) else 0.0)
    return rewards


def _grounding_iou_reward_value(
    prediction_value: Any,
    target_value: Any,
    *,
    min_iou: float,
) -> float:
    pred_instances = _coerce_instances(prediction_value)
    target_instances = _coerce_instances(target_value)
    pred_count = len(pred_instances)
    target_count = len(target_instances)
    if pred_count == 0 and target_count == 0:
        return 1.0
    if pred_count == 0 or target_count == 0:
        return 0.0

    candidates: list[tuple[float, int, int]] = []
    for pred_idx, (pred_label, pred_bbox) in enumerate(pred_instances):
        for target_idx, (target_label, target_bbox) in enumerate(target_instances):
            if pred_label and target_label and pred_label != target_label:
                continue
            iou = _iou_xyxy(pred_bbox, target_bbox)
            if iou >= min_iou:
                candidates.append((iou, pred_idx, target_idx))
    candidates.sort(key=lambda item: item[0], reverse=True)

    used_pred: set[int] = set()
    used_target: set[int] = set()
    matched_iou = 0.0
    for iou, pred_idx, target_idx in candidates:
        if pred_idx in used_pred or target_idx in used_target:
            continue
        used_pred.add(pred_idx)
        used_target.add(target_idx)
        matched_iou += float(iou)

    denominator = max(pred_count, target_count)
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, matched_iou / float(denominator)))


@register_grpo_reward("grounding_iou")
def grpo_reward_grounding_iou(
    *,
    prediction_texts: list[str],
    target_texts: list[str],
    codec_name: str,
    **params: Any,
) -> list[float]:
    codec = CODEC_REGISTRY.get(codec_name)
    min_iou = float(params.get("min_iou", 0.0))
    min_iou = max(0.0, min(1.0, min_iou))
    rewards: list[float] = []
    for prediction_text, target_text in zip(prediction_texts, target_texts, strict=True):
        decoded = codec(str(prediction_text))
        if not decoded.valid or decoded.partial:
            rewards.append(0.0)
            continue
        prediction_value = decoded.parsed if decoded.parsed is not None else decoded.raw_text
        target_value = _decode_target(str(target_text), codec_name=codec_name)
        rewards.append(
            _grounding_iou_reward_value(
                prediction_value,
                target_value,
                min_iou=min_iou,
            )
        )
    return rewards


def build_grpo_reward_functions(
    reward_configs: list[Any],
) -> list[Callable[..., list[float]]]:
    reward_functions: list[Callable[..., list[float]]] = []
    for reward_config in reward_configs:
        reward_impl = GRPO_REWARD_REGISTRY.get(reward_config.name)
        codec_name = str(reward_config.codec)
        params = dict(reward_config.params)

        def _reward_func(
            *,
            completions,
            target_text,
            _reward_impl=reward_impl,
            _codec_name=codec_name,
            _params=params,
            **kwargs,
        ):
            _ = kwargs
            values = _reward_impl(
                prediction_texts=[_completion_to_text(text) for text in completions],
                target_texts=[str(text) for text in target_text],
                codec_name=_codec_name,
                **_params,
                **kwargs,
            )
            return [float(value) for value in values]

        _reward_func.__name__ = f"grpo_reward_{reward_config.name}"
        reward_functions.append(_reward_func)
    return reward_functions
