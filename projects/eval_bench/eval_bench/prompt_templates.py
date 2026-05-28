from __future__ import annotations

from pathlib import Path
from typing import Any

from shaft.prompting import load_prompt_template


REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_EVAL_MAX_PIXELS = 2_000_000


DEFAULT_PROMPT_SPECS = (
    {
        "prompt_id": "grounding_arrow.v2.4.main",
        "label": "Arrow Detection",
        "task": "detection",
        "path": "configs/prompts/pools/grounding_arrow.v2.4.yaml",
        "parser": "raw_data_detection_v1",
        "metric_profile": "detection_iou_v1",
        "visualization_profile": "default",
        "generation": {"max_tokens": 4096, "temperature": 0, "top_p": 1},
        "data": {"max_pixels": MAIN_EVAL_MAX_PIXELS, "batch_size": 1},
        "target_labels": ["arrow"],
    },
    {
        "prompt_id": "grounding_layout.v2.4.main",
        "label": "Layout Detection",
        "task": "detection",
        "path": "configs/prompts/pools/grounding_layout.v2.4.yaml",
        "parser": "raw_data_detection_v1",
        "metric_profile": "detection_iou_v1",
        "visualization_profile": "default",
        "generation": {"max_tokens": 4096, "temperature": 0, "top_p": 1},
        "data": {"max_pixels": MAIN_EVAL_MAX_PIXELS, "batch_size": 1},
        "target_labels": ["icon", "image", "shape"],
    },
    {
        "prompt_id": "grounding_shape.v2.4.main",
        "label": "Shape Detection",
        "task": "detection",
        "path": "configs/prompts/pools/grounding_shape.v2.4.yaml",
        "parser": "raw_data_detection_v1",
        "metric_profile": "detection_iou_v1",
        "visualization_profile": "default",
        "generation": {"max_tokens": 4096, "temperature": 0, "top_p": 1},
        "data": {"max_pixels": MAIN_EVAL_MAX_PIXELS, "batch_size": 1},
        "target_labels": ["shape"],
    },
    {
        "prompt_id": "grounding_icon_image.v2.4.main",
        "label": "Icon/Image Detection",
        "task": "detection",
        "path": "configs/prompts/pools/grounding_icon_image.v2.4.yaml",
        "parser": "raw_data_detection_v1",
        "metric_profile": "detection_iou_v1",
        "visualization_profile": "default",
        "generation": {"max_tokens": 4096, "temperature": 0, "top_p": 1},
        "data": {"max_pixels": MAIN_EVAL_MAX_PIXELS, "batch_size": 1},
        "target_labels": ["icon", "image"],
    },
    {
        "prompt_id": "point_arrow.v2.4.main",
        "label": "Arrow Point",
        "task": "keypoint",
        "path": "configs/prompts/pools/point_arrow.v2.4.yaml",
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
        prompt = load_prompt_template(prompt_path, variant_id="main")
        metadata = dict(prompt.metadata)
        templates.append(
            {
                "prompt_id": spec["prompt_id"],
                "label": spec["label"],
                "task": spec["task"],
                "system_prompt": prompt.system_prompt,
                "user_prompt": prompt.user_prompt,
                "parser": spec["parser"],
                "metric_profile": spec["metric_profile"],
                "visualization_profile": spec["visualization_profile"],
                "generation": dict(spec["generation"]),
                "data": dict(spec["data"]),
                "metadata": {
                    "source": "repo_config",
                    "source_path": spec["path"],
                    "source_prompt_id": prompt.prompt_id,
                    "source_name": metadata.get("name"),
                    "prompt_pool_id": metadata.get("prompt_pool_id"),
                    "prompt_version": metadata.get("prompt_version"),
                    "prompt_variant_id": metadata.get("prompt_variant_id"),
                    "target_labels": list(spec["target_labels"]),
                },
            }
        )
    return templates
