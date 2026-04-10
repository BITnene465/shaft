from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any

from PIL import Image

from vlm_structgen.core.registry import get_adapter, parse_route_key
from vlm_structgen.core.prompting import render_prompt_template
from vlm_structgen.core.utils.io import load_jsonl

# Training can encounter extremely large figure images. We only decode images
# that are already part of the trusted dataset, so disable Pillow's
# decompression bomb guard here as well.
Image.MAX_IMAGE_PIXELS = None


class SFTDataset:
    def __init__(
        self,
        jsonl_path: str | Path | Sequence[str | Path],
        num_bins: int,
        system_prompt: str,
        user_prompt: str,
        system_prompt_template: str | None = None,
        user_prompt_template: str | None = None,
        route_prompts: dict[str, dict[str, Any]] | None = None,
        path_routes: str | Sequence[str] | None = None,
    ) -> None:
        self.num_bins = int(num_bins)
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.route_prompts = self._normalize_route_prompts(route_prompts)
        self._target_token_lengths_cache: dict[int, list[int]] = {}
        self.jsonl_paths = self._normalize_jsonl_paths(jsonl_path)
        self.path_routes = self._normalize_path_routes(path_routes, len(self.jsonl_paths))
        self.records: list[dict[str, Any]] = []
        for index, path in enumerate(self.jsonl_paths):
            route = self.path_routes[index]
            task_type = None
            domain_type = None
            if route is not None:
                task_type, domain_type = parse_route_key(str(route))
            for record in load_jsonl(path):
                normalized_record = dict(record)
                if task_type is not None and domain_type is not None:
                    normalized_record["task_type"] = task_type
                    normalized_record["domain_type"] = domain_type
                self.records.append(normalized_record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        adapter = self._get_adapter(record)
        image_path = Path(record["image_path"])
        image = Image.open(image_path).convert("RGB")
        record_gt_struct = record.get("gt_struct")
        if record_gt_struct is not None:
            gt_struct = record_gt_struct
        else:
            gt_struct = adapter.build_gt_struct_from_record(record)
        training_target = adapter.build_training_target(
            gt_struct,
            image_width=record["image_width"],
            image_height=record["image_height"],
        )
        target_text = training_target["target_text"]
        loss_meta = training_target.get("loss_meta")
        condition = record.get("condition", {})
        route_key = f"{adapter.task_type}/{adapter.domain_type}"
        route_prompt = dict(self.route_prompts.get(route_key, {}))
        system_prompt = record.get("system_prompt")
        if system_prompt is None:
            record_system_template = record.get("system_prompt_template")
            if record_system_template:
                system_prompt = render_prompt_template(str(record_system_template), condition)
            else:
                route_system_prompt = route_prompt.get("system_prompt")
                route_system_template = route_prompt.get("system_prompt_template")
                if route_system_template:
                    system_prompt = render_prompt_template(str(route_system_template), condition)
                elif route_system_prompt is not None:
                    system_prompt = route_system_prompt
                else:
                    if self.system_prompt_template:
                        system_prompt = render_prompt_template(self.system_prompt_template, condition)
                    else:
                        system_prompt = self.system_prompt
        user_prompt = record.get("user_prompt")
        if user_prompt is None:
            record_user_template = record.get("user_prompt_template")
            if record_user_template:
                user_prompt = render_prompt_template(str(record_user_template), condition)
            else:
                route_user_prompt = route_prompt.get("user_prompt")
                route_user_template = route_prompt.get("user_prompt_template")
                if route_user_template:
                    user_prompt = render_prompt_template(str(route_user_template), condition)
                elif route_user_prompt is not None:
                    user_prompt = route_user_prompt
                else:
                    if self.user_prompt_template:
                        user_prompt = render_prompt_template(self.user_prompt_template, condition)
                    else:
                        user_prompt = self.user_prompt
        return {
            "task_type": adapter.task_type,
            "domain_type": adapter.domain_type,
            "sample_id": record.get("sample_id", image_path.stem),
            "image_path": str(image_path),
            "image": image,
            "image_width": int(record["image_width"]),
            "image_height": int(record["image_height"]),
            "system_prompt": str(system_prompt),
            "user_prompt": str(user_prompt),
            "target_text": str(target_text),
            "loss_meta": loss_meta,
            "gt_struct": gt_struct,
        }

    def get_target_token_lengths(self, tokenizer) -> list[int]:
        cache_key = id(tokenizer)
        cached = self._target_token_lengths_cache.get(cache_key)
        if cached is not None:
            return cached
        lengths: list[int] = []
        for record in self.records:
            adapter = self._get_adapter(record)
            record_gt_struct = record.get("gt_struct")
            if record_gt_struct is not None:
                gt_struct = record_gt_struct
            else:
                gt_struct = adapter.build_gt_struct_from_record(record)
            target_text = adapter.build_training_target(
                gt_struct,
                image_width=record["image_width"],
                image_height=record["image_height"],
            )["target_text"]
            tokenized = tokenizer(
                str(target_text),
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            lengths.append(len(tokenized["input_ids"]))
        self._target_token_lengths_cache[cache_key] = lengths
        return lengths

    def _get_adapter(self, record: dict[str, Any]):
        try:
            return get_adapter(
                task_type=record.get("task_type"),
                domain_type=record.get("domain_type"),
                num_bins=self.num_bins,
            )
        except Exception as exc:  # noqa: BLE001
            sample_id = record.get("sample_id") or Path(str(record.get("image_path", ""))).stem or "<unknown>"
            raise ValueError(
                "Dataset sample is missing a valid task/domain route. "
                f"sample_id={sample_id!r}, task_type={record.get('task_type')!r}, "
                f"domain_type={record.get('domain_type')!r}. "
                "Set route bindings via data.registry_path + "
                "data.train_datasets/data.val_datasets in config, "
                "or provide route fields in JSONL."
            ) from exc

    def _normalize_jsonl_paths(self, jsonl_path: str | Path | Sequence[str | Path]) -> list[Path]:
        if isinstance(jsonl_path, (str, Path)):
            return [Path(jsonl_path)]
        paths = [Path(item) for item in jsonl_path]
        if not paths:
            raise ValueError("jsonl_path must not be empty.")
        return paths

    def _normalize_route_prompts(
        self,
        route_prompts: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        if route_prompts is None:
            return {}
        normalized: dict[str, dict[str, Any]] = {}
        for route_key, prompt_payload in dict(route_prompts).items():
            if not isinstance(prompt_payload, dict):
                raise ValueError(
                    "route_prompts must map route to a prompt payload mapping. "
                    f"route={route_key!r}, got={type(prompt_payload).__name__}."
                )
            normalized[str(route_key)] = dict(prompt_payload)
        return normalized

    def _normalize_path_routes(
        self,
        path_routes: str | Sequence[str] | None,
        path_count: int,
    ) -> list[str | None]:
        if path_routes is None:
            return [None] * path_count
        if isinstance(path_routes, str):
            return [str(path_routes)] * path_count
        normalized = [str(item) for item in path_routes]
        if len(normalized) != path_count:
            raise ValueError(
                "path_routes length must match jsonl_path length. "
                f"jsonl_paths={path_count}, path_routes={len(normalized)}."
            )
        return normalized
