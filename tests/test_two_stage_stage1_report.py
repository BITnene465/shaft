from __future__ import annotations

import unittest

from PIL import Image

from vlm_structgen.domains.arrow.infer.two_stage import TwoStageInferenceRunner


class FakeStage1Runner:
    def __init__(self, report: tuple[str, dict]) -> None:
        self._report = report
        self.calls = 0

    def predict(self, image, *, max_new_tokens=None):
        del image
        del max_new_tokens
        self.calls += 1
        return self._report


class TwoStageStage1ReportTests(unittest.TestCase):
    def test_stage1_report_is_passed_through_without_mixed_proposals(self) -> None:
        stage1_prediction = {"instances": [{"label": "single_arrow", "bbox": [1.0, 2.0, 3.0, 4.0], "keypoints": []}]}
        stage1_report = {
            "generation": {
                "requested_max_new_tokens": 128,
                "generated_tokens": 10,
                "returned_tokens": 10,
                "hit_max_new_tokens": False,
                "closed_json_payload": True,
                "stop_reason": "eos_or_unknown",
            },
            "lenient": {
                "ok": True,
                "prediction": stage1_prediction,
                "error": None,
                "recovered_prefix": False,
            },
            "strict": {
                "ok": True,
                "prediction": stage1_prediction,
                "error": None,
                "recovered_prefix": False,
            },
        }
        fake_stage1_runner = FakeStage1Runner(("raw-stage1", stage1_report))
        runner = TwoStageInferenceRunner(
            stage1_runner=fake_stage1_runner,
            stage2_runner=None,
            infer_config=object(),
            padding_ratio=0.5,
        )

        raw_text, report, prediction = runner._predict_stage1(
            Image.new("RGB", (32, 32), color="black"),
            max_new_tokens=128,
        )

        self.assertEqual(fake_stage1_runner.calls, 1)
        self.assertEqual(raw_text, "raw-stage1")
        self.assertIs(report, stage1_report)
        self.assertEqual(prediction, stage1_prediction)
        self.assertNotIn("branches", report)
        self.assertNotIn("mixed_proposals_enabled", report["generation"])


if __name__ == "__main__":
    unittest.main()
