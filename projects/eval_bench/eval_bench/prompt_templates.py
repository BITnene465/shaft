from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_EVAL_MAX_PIXELS = 2_000_000


DEFAULT_PROMPT_SPECS = (
    {
        "prompt_id": "grounding_arrow.latest",
        "label": "Arrow Detection",
        "task": "detection",
        "path": "configs/prompts/grounding_arrow.yaml",
        "parser": "raw_data_detection_v1",
        "metric_profile": "detection_iou_v1",
        "visualization_profile": "default",
        "generation": {"max_tokens": 4096, "temperature": 0, "top_p": 1},
        "data": {"max_pixels": MAIN_EVAL_MAX_PIXELS, "batch_size": 1},
        "target_labels": ["arrow"],
    },
    {
        "prompt_id": "grounding_layout.latest",
        "label": "Layout Detection",
        "task": "detection",
        "path": "configs/prompts/grounding_layout.yaml",
        "parser": "raw_data_detection_v1",
        "metric_profile": "detection_iou_v1",
        "visualization_profile": "default",
        "generation": {"max_tokens": 4096, "temperature": 0, "top_p": 1},
        "data": {"max_pixels": MAIN_EVAL_MAX_PIXELS, "batch_size": 1},
        "target_labels": ["icon", "image", "shape"],
    },
    {
        "prompt_id": "grounding_shape.latest",
        "label": "Shape Detection",
        "task": "detection",
        "path": "configs/prompts/grounding_shape.yaml",
        "parser": "raw_data_detection_v1",
        "metric_profile": "detection_iou_v1",
        "visualization_profile": "default",
        "generation": {"max_tokens": 4096, "temperature": 0, "top_p": 1},
        "data": {"max_pixels": MAIN_EVAL_MAX_PIXELS, "batch_size": 1},
        "target_labels": ["shape"],
    },
    {
        "prompt_id": "grounding_icon_image.latest",
        "label": "Icon/Image Detection",
        "task": "detection",
        "path": "configs/prompts/grounding_icon_image.yaml",
        "parser": "raw_data_detection_v1",
        "metric_profile": "detection_iou_v1",
        "visualization_profile": "default",
        "generation": {"max_tokens": 4096, "temperature": 0, "top_p": 1},
        "data": {"max_pixels": MAIN_EVAL_MAX_PIXELS, "batch_size": 1},
        "target_labels": ["icon", "image"],
    },
    {
        "prompt_id": "point_arrow.latest",
        "label": "Arrow Point",
        "task": "keypoint",
        "path": "configs/prompts/point_arrow.yaml",
        "parser": "raw_data_keypoint_v1",
        "metric_profile": "keypoint_endpoint_v1",
        "visualization_profile": "default",
        "generation": {"max_tokens": 4096, "temperature": 0, "top_p": 1},
        "data": {"max_pixels": MAIN_EVAL_MAX_PIXELS, "batch_size": 1},
        "target_labels": ["arrow"],
    },
)


def default_prompt_templates(repo_root: str | Path = REPO_ROOT) -> list[dict[str, Any]]:
    root = Path(repo_root)
    templates: list[dict[str, Any]] = []
    for spec in DEFAULT_PROMPT_SPECS:
        prompt_path = root / str(spec["path"])
        prompt_payload = _read_prompt_file(prompt_path)
        metadata = dict(prompt_payload.get("metadata") or {})
        prompt = dict(prompt_payload.get("prompt") or {})
        templates.append(
            {
                "prompt_id": spec["prompt_id"],
                "label": spec["label"],
                "task": spec["task"],
                "system_prompt": str(prompt.get("system_prompt") or "").strip(),
                "user_prompt": str(prompt.get("user_prompt") or "").strip(),
                "parser": spec["parser"],
                "metric_profile": spec["metric_profile"],
                "visualization_profile": spec["visualization_profile"],
                "generation": dict(spec["generation"]),
                "data": dict(spec["data"]),
                "metadata": {
                    "source": "repo_config",
                    "source_path": spec["path"],
                    "source_prompt_id": metadata.get("id"),
                    "source_name": metadata.get("name"),
                    "target_labels": list(spec["target_labels"]),
                },
            }
        )
    return templates


def _read_prompt_file(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"prompt file must contain a mapping: {path}")
    return payload
