from __future__ import annotations

import unittest

from vlm_structgen.core.config import load_config
from vlm_structgen.core.infer.config import load_one_stage_inference_config, load_two_stage_inference_config


class PromptProfileConfigTest(unittest.TestCase):
    def test_train_config_uses_prompt_profile(self) -> None:
        config = load_config("configs/train/train_stage1_lora_4b.yaml")
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
        self.assertEqual(infer_config.stage2.prompt.profile, "arrow.keypoint_sequence.stage2_fixed.v2")
        self.assertIn("Locate every instance", infer_config.stage1.prompt.user_prompt or "")
        self.assertIn("central main arrow", infer_config.stage2.prompt.user_prompt or "")
        self.assertIn("ignore other secondary arrows", infer_config.stage2.prompt.user_prompt or "")

    def test_train_config_route_prompt_profiles_resolution(self) -> None:
        config = load_config("configs/train/train_mixed_full_ft_4b.yaml")
        route_prompts = config.prompt.route_prompts
        self.assertEqual(sorted(route_prompts.keys()), ["grounding/arrow", "keypoint_sequence/arrow"])
        self.assertEqual(route_prompts["grounding/arrow"]["profile"], "arrow.grounding.stage1.v2")
        self.assertEqual(
            route_prompts["keypoint_sequence/arrow"]["profile"],
            "arrow.keypoint_sequence.stage2_fixed.v2",
        )
        self.assertIn("Locate every instance", route_prompts["grounding/arrow"]["user_prompt"])
        self.assertIn("central main arrow", route_prompts["keypoint_sequence/arrow"]["user_prompt"])


if __name__ == "__main__":
    unittest.main()
