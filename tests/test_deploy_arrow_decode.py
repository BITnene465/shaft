from __future__ import annotations

import unittest

from deploy.arrow.decode import decode_stage1_output, decode_stage2_output


class DeployArrowDecodeTest(unittest.TestCase):
    def test_stage1_decode(self) -> None:
        decoded = decode_stage1_output(
            '[{"label":"single_arrow","bbox_2d":[10,20,30,40]}]',
            image_width=100,
            image_height=200,
            strict=True,
        )
        self.assertEqual(decoded["instances"][0]["label"], "single_arrow")
        self.assertEqual(len(decoded["instances"][0]["bbox"]), 4)

    def test_stage2_decode(self) -> None:
        decoded = decode_stage2_output(
            '{"keypoints_2d":[[10,20],[30,40]]}',
            image_width=100,
            image_height=200,
            strict=True,
        )
        self.assertEqual(len(decoded["keypoints_2d"]), 2)
        self.assertEqual(len(decoded["keypoints"]), 2)


if __name__ == "__main__":
    unittest.main()
