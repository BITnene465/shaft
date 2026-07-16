from __future__ import annotations

from PIL import Image

from shaft.data import GRPODataset
from shaft.model import build_model_meta
from shaft.template import build_template


def test_grpo_dataset_applies_image_pixel_budget() -> None:
    image = Image.new("RGB", (100, 50), color=(255, 255, 255))

    class _SingleImageDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "image": image,
                "target_text": "{\"ok\":1}",
                "user_prompt": "return json",
                "dataset_name": "grpo_ds",
                "sample_id": "sample-1",
                "image_path": "/tmp/sample.png",
                "extra": {},
            }

    dataset = GRPODataset(
        _SingleImageDataset(),
        template=build_template("smoke_vlm"),
        image_preprocessor=lambda value: build_model_meta("qwen3vl")
        .resolve_adapter(model_name_or_path="models/Qwen3-VL-4B-Instruct")
        .prepare_rollout_image(value, min_pixels=None, max_pixels=2000),
    )

    sample = dataset[0]

    assert sample["image"].size[0] * sample["image"].size[1] <= 2000
    assert sample["image"].size != image.size
    assert image.size == (100, 50)


def test_grpo_dataset_does_not_apply_qwen_resize_without_model_policy() -> None:
    image = Image.new("RGB", (100, 50), color=(255, 255, 255))

    class _SingleImageDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {"image": image, "target_text": "{}", "extra": {}}

    sample = GRPODataset(
        _SingleImageDataset(),
        template=build_template("smoke_vlm"),
    )[0]

    assert sample["image"] is image
