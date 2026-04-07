from __future__ import annotations

import unittest

from vlm_structgen.core.config import ExperimentRuntimeConfig
from vlm_structgen.core.infer.config import InferTaskConfig, _apply_task_overrides
from vlm_structgen.core.utils.generation import find_balanced_json_end


class InferProtocolAlignmentTests(unittest.TestCase):
    def test_find_balanced_json_end_handles_arrays_and_objects(self) -> None:
        self.assertEqual(find_balanced_json_end('prefix [{"x":1}] tail'), 15)
        self.assertEqual(find_balanced_json_end('prefix {"keypoints_2d":[[1,2],[3,4]]} tail'), 36)
        self.assertIsNone(find_balanced_json_end("no json here"))

    def test_apply_task_overrides_merges_into_route_options(self) -> None:
        runtime = ExperimentRuntimeConfig()
        runtime.task.task_type = "keypoint_sequence"
        runtime.task.domain_type = "arrow"
        runtime.task.route_options = {
            "keypoint_sequence/arrow": {
                "coordinate_token_loss_weight": 1.5,
            }
        }

        _apply_task_overrides(
            runtime,
            InferTaskConfig(
                route_options={
                    "coordinate_token_loss_weight": 2.0,
                    "decode_mode": "strict",
                }
            ),
        )

        self.assertEqual(
            runtime.task.route_options["keypoint_sequence/arrow"],
            {
                "coordinate_token_loss_weight": 2.0,
                "decode_mode": "strict",
            },
        )


if __name__ == "__main__":
    unittest.main()
