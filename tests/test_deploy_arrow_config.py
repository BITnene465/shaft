from __future__ import annotations

import unittest

from deploy.arrow.config import load_arrow_config


class DeployArrowConfigTest(unittest.TestCase):
    def test_load_default_config(self) -> None:
        config = load_arrow_config()
        self.assertEqual(config.protocol.labels, ("single_arrow", "double_arrow"))
        self.assertEqual(config.protocol.num_bins, 1000)
        self.assertEqual(config.stage1.route, "grounding_arrow")
        self.assertEqual(config.stage2.route, "keypoint_sequence_arrow")
        self.assertFalse(config.stage1.do_sample)
        self.assertFalse(config.stage2.do_sample)
        self.assertEqual(config.stage1.temperature, 0.0)
        self.assertEqual(config.stage2.temperature, 0.0)
        self.assertEqual(config.stage1.top_p, 1.0)
        self.assertEqual(config.stage2.top_p, 1.0)
        self.assertIn("Locate every instance", config.stage1.prompt)
        self.assertIn("central main arrow", config.stage2.prompt)
        self.assertEqual(config.padding_ratio, 0.3)


if __name__ == "__main__":
    unittest.main()
