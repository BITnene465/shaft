from __future__ import annotations

from PIL import Image

from shaft.data import GRPODataset
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
        max_pixels=2000,
    )

    sample = dataset[0]

    assert sample["image"].size[0] * sample["image"].size[1] <= 2000
    assert sample["image"].size != image.size
    assert image.size == (100, 50)
