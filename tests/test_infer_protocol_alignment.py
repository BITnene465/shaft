from __future__ import annotations

import unittest

from vlm_structgen.core.config import ExperimentRuntimeConfig
from vlm_structgen.core.utils.generation import find_balanced_json_end
from vlm_structgen.runtime.infer.config import (
    InferDataConfig,
    InferTaskConfig,
    _apply_data_overrides,
    _apply_task_overrides,
)


class InferProtocolAlignmentTests(unittest.TestCase):
    def test_find_balanced_json_end_handles_arrays_and_objects(self) -> None:
        self.assertEqual(find_balanced_json_end('prefix [{"x":1}] tail'), 15)
        self.assertEqual(find_balanced_json_end('prefix {"keypoints_2d":[[1,2],[3,4]]} tail'), 36)
        self.assertIsNone(find_balanced_json_end("no json here"))

    def test_apply_task_overrides_merges_into_route_options(self) -> None:
        runtime = ExperimentRuntimeConfig()
        runtime.task.route = "keypoint_sequence/arrow"
        runtime.task.route_options = {
            "keypoint_sequence/arrow": {
                "decode_mode": "lenient",
            }
        }

        _apply_task_overrides(
            runtime,
            InferTaskConfig(
                route="keypoint_sequence/arrow",
                route_options={
                    "decode_mode": "strict",
                    "strict_point_distance_px": 8.0,
                }
            ),
        )

        self.assertEqual(
            runtime.task.route_options["keypoint_sequence/arrow"],
            {
                "decode_mode": "strict",
                "strict_point_distance_px": 8.0,
            },
        )

    def test_apply_data_overrides_updates_runtime_pixel_budget(self) -> None:
        runtime = ExperimentRuntimeConfig()
        runtime.data.min_pixels = 200704
        runtime.data.max_pixels = 1048576

        _apply_data_overrides(
            runtime,
            InferDataConfig(
                min_pixels=50176,
                max_pixels=262144,
            ),
        )

        self.assertEqual(runtime.data.min_pixels, 50176)
        self.assertEqual(runtime.data.max_pixels, 262144)


if __name__ == "__main__":
    unittest.main()
