from __future__ import annotations

import unittest

from PIL import Image

from deploy.arrow.pipeline import build_padded_crop


class DeployArrowPipelineTest(unittest.TestCase):
    def test_build_padded_crop_with_out_of_bounds_box(self) -> None:
        image = Image.new("RGB", (100, 80), color=(255, 255, 255))
        crop, crop_box = build_padded_crop(image, bbox=[10, 20, 40, 50], padding_ratio=0.5)

        self.assertEqual(len(crop_box), 4)
        self.assertGreater(crop.size[0], 0)
        self.assertGreater(crop.size[1], 0)
        self.assertEqual(crop.mode, "RGB")

    def test_build_padded_crop_preserves_black_padding(self) -> None:
        image = Image.new("RGB", (20, 20), color=(255, 255, 255))
        crop, _crop_box = build_padded_crop(image, bbox=[0, 0, 5, 5], padding_ratio=1.0)
        self.assertEqual(crop.getpixel((0, 0)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
