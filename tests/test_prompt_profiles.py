from __future__ import annotations

import unittest

from vlm_structgen.core.config import load_config
from vlm_structgen.core.infer.config import load_one_stage_inference_config, load_two_stage_inference_config


class PromptProfileConfigTest(unittest.TestCase):
    def test_train_config_uses_prompt_profile(self) -> None:
        config = load_config("configs/train/train_stage1_lora_2b.yaml")
        self.assertEqual(config.prompt.profile, "arrow.grounding.stage1.v2")
        self.assertIn("Locate every instance", config.prompt.user_prompt)
        self.assertEqual(config.prompt.system_prompt, "")

    def test_one_stage_infer_profile_resolution(self) -> None:
        infer_config = load_one_stage_inference_config("configs/infer/infer_one_stage.yaml")
        self.assertEqual(infer_config.prompt.profile, "arrow.joint_structure.one_stage.v1")
        self.assertIn("Detect all arrows", infer_config.prompt.user_prompt or "")

    def test_two_stage_infer_profile_resolution(self) -> None:
        infer_config = load_two_stage_inference_config("configs/infer/infer_two_stage.yaml")
        self.assertEqual(infer_config.stage1.prompt.profile, "arrow.grounding.stage1.v2")
        self.assertEqual(infer_config.stage2.prompt.profile, "arrow.keypoint_sequence.stage2_template.v1")
        self.assertIn("Locate every instance", infer_config.stage1.prompt.user_prompt or "")
        self.assertIn("{{label}}", infer_config.stage2.prompt.user_prompt_template or "")


if __name__ == "__main__":
    unittest.main()
