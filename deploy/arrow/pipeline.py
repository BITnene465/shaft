from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from openai import OpenAI
from PIL import Image

from .config import ArrowRuntimeConfig, load_arrow_config
from .decode import decode_stage1_output, decode_stage2_output

__all__ = [
    "Stage1Result",
    "Stage2Result",
    "TwoStageResult",
    "ArrowVLLMClient",
    "ArrowTwoStagePipeline",
    "build_padded_crop",
]


@dataclass(frozen=True)
class Stage1Result:
    raw_text: str
    decoded: dict[str, Any]


@dataclass(frozen=True)
class Stage2Result:
    raw_text: str
    decoded: dict[str, Any]
    crop_box: list[int]


@dataclass(frozen=True)
class TwoStageResult:
    stage1: Stage1Result
    stage2: list[Stage2Result]
    final_prediction: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage1": {
                "raw_text": self.stage1.raw_text,
                "decoded": self.stage1.decoded,
            },
            "stage2": [
                {
                    "raw_text": result.raw_text,
                    "decoded": result.decoded,
                    "crop_box": result.crop_box,
                }
                for result in self.stage2
            ],
            "final_prediction": self.final_prediction,
        }


class ArrowVLLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "EMPTY",
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate_with_image(
        self,
        *,
        model: str,
        prompt: str,
        image: str | Path | Image.Image,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        ) -> str:
        image_url = _image_to_data_url(image)
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content or ""


class ArrowTwoStagePipeline:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        config: ArrowRuntimeConfig | None = None,
        config_path: str | Path | None = None,
        stage1_model: str | None = None,
        stage2_model: str | None = None,
        stage1_max_tokens: int | None = None,
        stage2_max_tokens: int | None = None,
        stage1_temperature: float | None = None,
        stage2_temperature: float | None = None,
        padding_ratio: float | None = None,
    ) -> None:
        self.config = config or load_arrow_config(config_path)
        self.client = ArrowVLLMClient(base_url=base_url, api_key=api_key)
        self.stage1_model = stage1_model or self.config.stage1.route
        self.stage2_model = stage2_model or self.config.stage2.route
        self.stage1_max_tokens = int(stage1_max_tokens if stage1_max_tokens is not None else self.config.stage1.max_tokens)
        self.stage2_max_tokens = int(stage2_max_tokens if stage2_max_tokens is not None else self.config.stage2.max_tokens)
        self.stage1_do_sample = bool(self.config.stage1.do_sample)
        self.stage2_do_sample = bool(self.config.stage2.do_sample)
        self.stage1_temperature = float(stage1_temperature if stage1_temperature is not None else self.config.stage1.temperature)
        self.stage2_temperature = float(stage2_temperature if stage2_temperature is not None else self.config.stage2.temperature)
        self.stage1_top_p = float(self.config.stage1.top_p)
        self.stage2_top_p = float(self.config.stage2.top_p)
        self.padding_ratio = float(padding_ratio if padding_ratio is not None else self.config.padding_ratio)

    def predict_stage1(self, image: str | Path | Image.Image) -> Stage1Result:
        raw_text = self.client.generate_with_image(
            model=self.stage1_model,
            prompt=self.config.stage1.prompt,
            image=image,
            max_tokens=self.stage1_max_tokens,
            temperature=self.stage1_temperature if self.stage1_do_sample else 0.0,
            top_p=self.stage1_top_p if self.stage1_do_sample else 1.0,
        )
        width, height = _image_size(image)
        decoded = decode_stage1_output(
            raw_text,
            image_width=width,
            image_height=height,
            strict=False,
            protocol=self.config.protocol,
        )
        return Stage1Result(raw_text=raw_text, decoded=decoded)

    def predict_stage2(self, image: str | Path | Image.Image, crop_box: Sequence[int]) -> Stage2Result:
        raw_text = self.client.generate_with_image(
            model=self.stage2_model,
            prompt=self.config.stage2.prompt,
            image=image,
            max_tokens=self.stage2_max_tokens,
            temperature=self.stage2_temperature if self.stage2_do_sample else 0.0,
            top_p=self.stage2_top_p if self.stage2_do_sample else 1.0,
        )
        width, height = _image_size(image)
        decoded = decode_stage2_output(
            raw_text,
            image_width=width,
            image_height=height,
            strict=False,
            protocol=self.config.protocol,
        )
        return Stage2Result(raw_text=raw_text, decoded=decoded, crop_box=[int(v) for v in crop_box])

    def predict_two_stage(self, image: str | Path | Image.Image) -> TwoStageResult:
        image_rgb = _load_image_rgb(image)
        stage1 = self.predict_stage1(image_rgb)

        final_instances: list[dict[str, Any]] = []
        stage2_results: list[Stage2Result] = []
        for instance in stage1.decoded.get("instances", []):
            bbox = instance.get("bbox", [])
            if len(bbox) != 4:
                continue
            crop_image, crop_box = build_padded_crop(image_rgb, bbox=bbox, padding_ratio=self.padding_ratio)
            stage2 = self.predict_stage2(crop_image, crop_box)
            stage2_results.append(stage2)

            keypoints = [
                [float(point[0]) + float(crop_box[0]), float(point[1]) + float(crop_box[1])]
                for point in stage2.decoded.get("keypoints", [])
            ]
            final_instances.append(
                {
                    "label": instance["label"],
                    "bbox": [float(v) for v in instance.get("bbox", [])],
                    "keypoints": keypoints,
                }
            )

        final_prediction = {"instances": final_instances}
        return TwoStageResult(stage1=stage1, stage2=stage2_results, final_prediction=final_prediction)


def build_padded_crop(
    image: Image.Image,
    *,
    bbox: Sequence[float],
    padding_ratio: float = 0.3,
) -> tuple[Image.Image, list[int]]:
    image = image.convert("RGB")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    pad_x = box_w * float(padding_ratio)
    pad_y = box_h * float(padding_ratio)
    crop_x1 = int(round(x1 - pad_x))
    crop_y1 = int(round(y1 - pad_y))
    crop_x2 = int(round(x2 + pad_x))
    crop_y2 = int(round(y2 + pad_y))
    crop_box = [crop_x1, crop_y1, crop_x2, crop_y2]
    return _render_crop_with_black_padding(image, crop_box), crop_box


def _image_to_data_url(image: str | Path | Image.Image) -> str:
    if isinstance(image, Image.Image):
        image = _image_to_bytes(image)
        mime_type = "image/png"
        return f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"

    image_path = Path(image)
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        mime_type = "image/png"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _image_to_bytes(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_size(image: str | Path | Image.Image) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return int(image.width), int(image.height)
    with Image.open(image) as pil_image:
        return int(pil_image.width), int(pil_image.height)


def _load_image_rgb(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as pil_image:
        return pil_image.convert("RGB")


def _render_crop_with_black_padding(image: Image.Image, crop_box: Sequence[int]) -> Image.Image:
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
