from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_structgen.core.config import ExperimentRuntimeConfig
from vlm_structgen.core.registry import get_adapter
from vlm_structgen.domains.arrow.data.two_stage import (
    _expand_crop_box_to_max_aspect_ratio,
    build_padded_crop,
    quantize_bbox_2d,
    to_crop_local_bbox,
)
from vlm_structgen.core.modeling.builder import BuildArtifacts
from vlm_structgen.core.prompting import build_chat_prompt, render_prompt_template, temporary_padding_side
from vlm_structgen.core.utils.distributed import reset_model_runtime_state, unwrap_model
from vlm_structgen.core.utils.generation import (
    build_generate_kwargs,
    find_balanced_json_end,
    trim_generated_ids_at_eos,
)
from vlm_structgen.domains.arrow.infer.config import (
    TwoStageInferenceConfig,
    load_two_stage_inference_config,
)
from vlm_structgen.runtime.infer.runner import InferenceRunner, load_inference_runner
from vlm_structgen.tasks.bootstrap import ensure_builtin_task_adapters_registered


@dataclass
class Stage2KeypointInferenceRunner:
    config: ExperimentRuntimeConfig
    artifacts: BuildArtifacts
    adapter: Any
    device: torch.device
    batch_size: int = 1

    def predict_batch(
        self,
        requests: list[Stage2Request],
        *,
        max_new_tokens: int | None = None,
    ) -> list[Stage2PredictionResult]:
        if not requests:
            return []
        sorted_requests = sorted(
            requests,
            key=lambda request: (
                int(request.crop_image.width) * int(request.crop_image.height),
                len(request.label),
                int(request.index),
            ),
        )
        raw_model = unwrap_model(self.artifacts.model)
        raw_model.eval()
        results_by_index: dict[int, Stage2PredictionResult] = {}
        effective_batch_size = max(int(self.batch_size), 1)
        for start in range(0, len(sorted_requests), effective_batch_size):
            batch_requests = sorted_requests[start : start + effective_batch_size]
            prompt_texts = [self._build_prompt(request) for request in batch_requests]
            images = [request.crop_image.convert("RGB") for request in batch_requests]
            model_inputs, input_context_length = self._prepare_inputs(images, prompt_texts=prompt_texts)
            generate_kwargs = build_generate_kwargs(
                self.artifacts.tokenizer,
                generation_config=getattr(raw_model, "generation_config", None),
                num_bins=self.adapter.num_bins,
                prompt_lengths=[input_context_length] * len(batch_requests),
                max_new_tokens=max_new_tokens or self.config.eval.max_new_tokens,
                num_beams=self.config.eval.num_beams,
                do_sample=self.config.eval.do_sample,
                temperature=self.config.eval.temperature,
                top_p=self.config.eval.top_p,
                top_k=self.config.eval.top_k,
                use_cache=self.config.eval.use_cache,
            )
            requested_max_new_tokens = int(generate_kwargs["max_new_tokens"])
            with torch.inference_mode():
                reset_model_runtime_state(raw_model)
                output_ids = raw_model.generate(**model_inputs, **generate_kwargs)
            for row_index, request in enumerate(batch_requests):
                width, height = request.crop_image.size
                continuation = output_ids[row_index, input_context_length:]
                continuation_ids = continuation.tolist()
                raw_continuation_text = self.artifacts.tokenizer.decode(continuation_ids, skip_special_tokens=False)
                json_payload_end = find_balanced_json_end(raw_continuation_text)
                trimmed_ids = trim_generated_ids_at_eos(continuation, generate_kwargs.get("eos_token_id"))
                decoded = self.artifacts.tokenizer.decode(trimmed_ids, skip_special_tokens=False)
                strict_text = self.artifacts.tokenizer.decode(trimmed_ids, skip_special_tokens=True)
                closed_json_payload = json_payload_end is not None
                effective_generated_tokens = len(trimmed_ids)
                hit_max_new_tokens = effective_generated_tokens >= requested_max_new_tokens

                lenient_prediction: dict[str, Any] | None = None
                lenient_error: str | None = None
                lenient_recovered_prefix = False
                strict_error: str | None = None
                try:
                    lenient_prediction, lenient_meta = self.adapter.decode_with_meta(
                        decoded,
                        image_width=width,
                        image_height=height,
                    )
                    lenient_recovered_prefix = bool(lenient_meta.get("recovered_prefix", False))
                except Exception as exc:  # noqa: BLE001
                    lenient_error = str(exc)

                if lenient_prediction is not None:
                    try:
                        self.adapter.decode(strict_text, image_width=width, image_height=height, strict=True)
                    except Exception as exc:  # noqa: BLE001
                        strict_error = str(exc)
                else:
                    strict_error = lenient_error

                results_by_index[request.index] = Stage2PredictionResult(
                    index=int(request.index),
                    crop_box=[int(value) for value in request.crop_box],
                    raw_text=decoded,
                    report={
                        "raw_text": decoded,
                        "generation": {
                            "requested_max_new_tokens": requested_max_new_tokens,
                            "generated_tokens": effective_generated_tokens,
                            "returned_tokens": len(trimmed_ids),
                            "hit_max_new_tokens": hit_max_new_tokens,
                            "closed_json_payload": closed_json_payload,
                            "stop_reason": (
                                "max_new_tokens"
                                if hit_max_new_tokens
                                else "eos_or_unknown"
                            ),
                        },
                        "lenient": {
                            "ok": lenient_error is None,
                            "prediction": lenient_prediction,
                            "error": lenient_error,
                            "recovered_prefix": lenient_recovered_prefix,
                        },
                        "strict": {
                            "ok": strict_error is None,
                            "prediction": lenient_prediction if strict_error is None else None,
                            "error": strict_error,
                            "recovered_prefix": False,
                        },
                    },
                )
        return [results_by_index[request.index] for request in requests]

    def _build_prompt(self, request: Stage2Request) -> str:
        template = self.config.prompt.user_prompt_template
        if template:
            prompt = render_prompt_template(
                template,
                {
                    "label": request.label,
                    "bbox_2d": request.bbox_2d,
                },
            )
        else:
            prompt = str(self.config.prompt.user_prompt)
        return build_chat_prompt(
            self.artifacts.processor,
            self.artifacts.tokenizer,
            system_prompt=self.config.prompt.system_prompt,
            user_prompt=prompt,
        )

    def _prepare_inputs(
        self,
        images: list[Image.Image],
        *,
        prompt_texts: list[str],
    ) -> tuple[dict[str, torch.Tensor], int]:
        processor_kwargs: dict[str, Any] = {
            "text": prompt_texts,
            "images": images,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.config.model.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.model.min_pixels
        if self.config.model.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.model.max_pixels
        with temporary_padding_side(self.artifacts.processor, self.artifacts.tokenizer, padding_side="left"):
            batch = self.artifacts.processor(**processor_kwargs)
        input_context_length = int(batch["input_ids"].shape[1])
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        return model_inputs, input_context_length


@dataclass
class Stage2Request:
    index: int
    image_index: int
    instance_index: int
    crop_image: Image.Image
    crop_box: list[int]
    label: str
    bbox_2d: list[int]


@dataclass
class Stage2PredictionResult:
    index: int
    crop_box: list[int]
    raw_text: str
    report: dict[str, Any]


@dataclass
class TwoStageInferenceRunner:
    stage1_runner: InferenceRunner
    stage2_runner: Stage2KeypointInferenceRunner | None
    infer_config: TwoStageInferenceConfig
    padding_ratio: float = 0.5
    stage2_max_crop_aspect_ratio: float = 180.0

    def _extract_stage1_prediction(self, report: dict[str, Any]) -> dict[str, Any] | None:
        return report["strict"]["prediction"] or report["lenient"]["prediction"]

    @staticmethod
    def _crop_aspect_ratio(image: Image.Image) -> float:
        width = max(int(image.width), 1)
        height = max(int(image.height), 1)
        return max(float(width) / float(height), float(height) / float(width))

    @staticmethod
    def _render_crop_from_box(image: Image.Image, crop_box: list[int]) -> Image.Image:
        crop_x1, crop_y1, crop_x2, crop_y2 = [int(value) for value in crop_box]
        crop_w = max(int(crop_x2 - crop_x1), 1)
        crop_h = max(int(crop_y2 - crop_y1), 1)
        canvas = Image.new("RGB", (crop_w, crop_h), color=(0, 0, 0))
        src_x1 = max(crop_x1, 0)
        src_y1 = max(crop_y1, 0)
        src_x2 = min(crop_x2, int(image.width))
        src_y2 = min(crop_y2, int(image.height))
        if src_x2 > src_x1 and src_y2 > src_y1:
            patch = image.crop((src_x1, src_y1, src_x2, src_y2))
            canvas.paste(patch, (int(src_x1 - crop_x1), int(src_y1 - crop_y1)))
        return canvas

    def _build_stage2_requests_for_image(
        self,
        *,
        image_index: int,
        image: Image.Image,
        stage1_prediction: dict[str, Any],
        start_index: int,
    ) -> tuple[list[Stage2Request], int]:
        if self.stage2_runner is None:
            return [], start_index

        requests: list[Stage2Request] = []
        next_index = int(start_index)
        for instance_index, instance in enumerate(stage1_prediction.get("instances", [])):
            bbox = instance.get("bbox", [])
            label = str(instance.get("label", ""))
            if len(bbox) != 4:
                continue
            crop_image, crop_box = build_padded_crop(
                image,
                bbox=[float(value) for value in bbox],
                padding_ratio=self.padding_ratio,
            )
            if self._crop_aspect_ratio(crop_image) > float(self.stage2_max_crop_aspect_ratio):
                crop_box = _expand_crop_box_to_max_aspect_ratio(
                    crop_box,
                    max_aspect_ratio=self.stage2_max_crop_aspect_ratio,
                )
                crop_image = self._render_crop_from_box(image, crop_box)
            crop_width, crop_height = crop_image.size
            local_bbox = to_crop_local_bbox([float(value) for value in bbox], crop_box)
            local_bbox_2d = quantize_bbox_2d(
                local_bbox,
                crop_width,
                crop_height,
                self.stage2_runner.adapter.num_bins,
            )
            requests.append(
                Stage2Request(
                    index=next_index,
                    image_index=int(image_index),
                    instance_index=int(instance_index),
                    crop_image=crop_image,
                    crop_box=[int(value) for value in crop_box],
                    label=label,
                    bbox_2d=local_bbox_2d,
                )
            )
            next_index += 1
        return requests, next_index

    def _finalize_stage2_outputs(
        self,
        *,
        stage1_predictions: list[dict[str, Any]],
        stage2_requests: list[Stage2Request],
        stage2_results: list[Stage2PredictionResult],
    ) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
        final_instances_by_image: list[list[dict[str, Any]]] = [[] for _ in stage1_predictions]
        stage2_reports_by_image: list[list[dict[str, Any]]] = [[] for _ in stage1_predictions]
        for request, result in zip(stage2_requests, stage2_results, strict=False):
            image_index = int(request.image_index)
            instance_index = int(request.instance_index)
            stage1_instances = stage1_predictions[image_index].get("instances", [])
            lenient_prediction = result.report["lenient"]["prediction"]
            strict_prediction = result.report["strict"]["prediction"]
            local_prediction = strict_prediction or lenient_prediction
            if local_prediction is None:
                stage2_reports_by_image[image_index].append(result.report)
                continue

            global_keypoints = [
                [float(point[0]) + float(request.crop_box[0]), float(point[1]) + float(request.crop_box[1])]
                for point in local_prediction.get("keypoints", [])
            ]
            final_instances_by_image[image_index].append(
                {
                    "label": request.label,
                    "bbox": [float(value) for value in stage1_instances[instance_index]["bbox"]],
                    "keypoints": global_keypoints,
                }
            )
            stage2_reports_by_image[image_index].append(result.report)
        return final_instances_by_image, stage2_reports_by_image

    def _predict_stage1(
        self,
        image: Image.Image,
        *,
        max_new_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        raw_text, report = self.stage1_runner.predict(image, max_new_tokens=max_new_tokens)
        prediction = self._extract_stage1_prediction(report) or {"instances": []}
        return raw_text, report, prediction

    def predict_batch(
        self,
        images: list[Image.Image],
        *,
        stage1_max_new_tokens: int | None = None,
        stage2_max_new_tokens: int | None = None,
        stage1_batch_size: int | None = None,
        stage2_batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        if not images:
            return []

        pil_images = [image.convert("RGB") for image in images]
        effective_stage1_batch_size = max(
            int(stage1_batch_size or getattr(getattr(self.stage1_runner, "settings", None), "batch_size", 1)),
            1,
        )
        stage1_raw_texts: list[str] = []
        stage1_reports: list[dict[str, Any]] = []
        stage1_predictions: list[dict[str, Any]] = []
        for start in range(0, len(pil_images), effective_stage1_batch_size):
            batch_images = pil_images[start : start + effective_stage1_batch_size]
            batch_outputs = self.stage1_runner.predict_batch(batch_images, max_new_tokens=stage1_max_new_tokens)
            for raw_text, report in batch_outputs:
                stage1_raw_texts.append(raw_text)
                stage1_reports.append(report)
                stage1_predictions.append(self._extract_stage1_prediction(report) or {"instances": []})

        if self.stage2_runner is None:
            return [
                {
                    "stage1_raw_text": raw_text,
                    "stage1_report": report,
                    "stage2_results": [],
                    "final_prediction": prediction,
                }
                for raw_text, report, prediction in zip(stage1_raw_texts, stage1_reports, stage1_predictions, strict=False)
            ]

        if stage2_batch_size is not None:
            self.stage2_runner.batch_size = max(int(stage2_batch_size), 1)

        stage2_requests: list[Stage2Request] = []
        next_stage2_index = 0
        for image_index, (image, prediction) in enumerate(zip(pil_images, stage1_predictions, strict=False)):
            image_requests, next_stage2_index = self._build_stage2_requests_for_image(
                image_index=image_index,
                image=image,
                stage1_prediction=prediction,
                start_index=next_stage2_index,
            )
            stage2_requests.extend(image_requests)

        if not stage2_requests:
            return [
                {
                    "stage1_raw_text": raw_text,
                    "stage1_report": report,
                    "stage2_results": [],
                    "final_prediction": {"instances": []},
                }
                for raw_text, report in zip(stage1_raw_texts, stage1_reports, strict=False)
            ]

        stage2_results = self.stage2_runner.predict_batch(
            stage2_requests,
            max_new_tokens=stage2_max_new_tokens,
        )
        final_instances_by_image, stage2_reports_by_image = self._finalize_stage2_outputs(
            stage1_predictions=stage1_predictions,
            stage2_requests=stage2_requests,
            stage2_results=stage2_results,
        )
        return [
            {
                "stage1_raw_text": raw_text,
                "stage1_report": report,
                "stage2_results": stage2_reports_by_image[index],
                "final_prediction": {"instances": final_instances_by_image[index]},
            }
            for index, (raw_text, report) in enumerate(zip(stage1_raw_texts, stage1_reports, strict=False))
        ]

    def predict(
        self,
        image: Image.Image,
        *,
        stage1_max_new_tokens: int | None = None,
        stage2_max_new_tokens: int | None = None,
        stage1_batch_size: int | None = None,
        stage2_batch_size: int | None = None,
    ) -> dict[str, Any]:
        if self.stage2_runner is None:
            stage1_raw_text, stage1_report, stage1_prediction = self._predict_stage1(
                image.convert("RGB"),
                max_new_tokens=stage1_max_new_tokens,
            )
            return {
                "stage1_raw_text": stage1_raw_text,
                "stage1_report": stage1_report,
                "stage2_results": [],
                "final_prediction": stage1_prediction,
            }
        return self.predict_batch(
            [image],
            stage1_max_new_tokens=stage1_max_new_tokens,
            stage2_max_new_tokens=stage2_max_new_tokens,
            stage1_batch_size=stage1_batch_size,
            stage2_batch_size=stage2_batch_size,
        )[0]


def load_two_stage_inference_runner(
    *,
    config_path: str | Path,
    stage1_dense_model_name_or_path: str | None = None,
    stage1_lora_adapter_path: str | Path | None = None,
    stage2_dense_model_name_or_path: str | None = None,
    stage2_lora_adapter_path: str | Path | None = None,
    device_name: str | None = None,
) -> TwoStageInferenceRunner:
    ensure_builtin_task_adapters_registered()
    infer_config: TwoStageInferenceConfig = load_two_stage_inference_config(config_path)
    resolved_stage2_dense_model_name_or_path = stage2_dense_model_name_or_path or stage1_dense_model_name_or_path
    loaded_stage1_runner = load_inference_runner(
        dense_model_name_or_path=stage1_dense_model_name_or_path,
        lora_adapter_path=stage1_lora_adapter_path,
        infer_config=infer_config.stage1,
        device_name=device_name,
    )
    loaded_stage2_runner = load_inference_runner(
        dense_model_name_or_path=resolved_stage2_dense_model_name_or_path,
        lora_adapter_path=stage2_lora_adapter_path,
        infer_config=infer_config.stage2,
        device_name=device_name,
    )
    stage2_runner = Stage2KeypointInferenceRunner(
        config=loaded_stage2_runner.config,
        artifacts=loaded_stage2_runner.artifacts,
        adapter=loaded_stage2_runner.adapter,
        device=loaded_stage2_runner.device,
        batch_size=max(int(getattr(infer_config.stage2, "batch_size", 1)), 1),
    )
    return TwoStageInferenceRunner(
        stage1_runner=loaded_stage1_runner,
        stage2_runner=stage2_runner,
        infer_config=infer_config,
        padding_ratio=infer_config.padding_ratio,
    )
