from __future__ import annotations

import unittest

from PIL import Image

from vlm_structgen.domains.arrow.infer.two_stage import TwoStageInferenceRunner


class FakeStage1Runner:
    def __init__(self, reports: list[tuple[str, dict]]) -> None:
        self._reports = list(reports)
        self._index = 0

    def predict(self, image, *, max_new_tokens=None):
        del image
        del max_new_tokens
        result = self._reports[self._index]
        self._index += 1
        return result


class TwoStageStage1ReportTests(unittest.TestCase):
    def _build_runner(self, reports: list[tuple[str, dict]], *, include_full_image: bool = True):
        infer_config = type(
            "InferConfig",
            (),
            {
                "stage1": type(
                    "Stage1Config",
                    (),
                    {
                        "include_full_image": include_full_image,
                        "tile_size_ratios": [0.5],
                        "min_tile_size": 16,
                        "max_tile_size": 16,
                        "tile_stride_ratio": 1.0,
                        "proposal_dedup_iou_threshold": 0.5,
                    },
                )(),
            },
        )()
        runner = TwoStageInferenceRunner(
            stage1_runner=FakeStage1Runner(reports),
            stage2_runner=None,
            infer_config=infer_config,
            padding_ratio=0.5,
        )
        runner._build_stage1_tile_boxes = lambda image: [[0, 0, 16, 16]]
        return runner

    def test_stage1_aggregate_report_preserves_branch_failures(self) -> None:
        success_prediction = {"instances": [{"label": "single_arrow", "bbox": [1.0, 2.0, 3.0, 4.0], "keypoints": []}]}
        reports = [
            (
                "full",
                {
                    "lenient": {"ok": True, "prediction": success_prediction, "error": None, "recovered_prefix": False},
                    "strict": {"ok": True, "prediction": success_prediction, "error": None, "recovered_prefix": False},
                },
            ),
            (
                "tile",
                {
                    "lenient": {"ok": False, "prediction": None, "error": "tile lenient parse failed", "recovered_prefix": False},
                    "strict": {"ok": False, "prediction": None, "error": "tile strict parse failed", "recovered_prefix": False},
                },
            ),
        ]
        runner = self._build_runner(reports)

        _raw_text, stage1_report, prediction = runner._predict_stage1_with_options(
            Image.new("RGB", (32, 32), color="black"),
            use_mixed_proposals=True,
        )

        self.assertEqual(len(prediction["instances"]), 1)
        self.assertFalse(stage1_report["lenient"]["ok"])
        self.assertFalse(stage1_report["strict"]["ok"])
        self.assertEqual(stage1_report["generation"]["num_lenient_failed_branches"], 1)
        self.assertEqual(stage1_report["generation"]["num_strict_failed_branches"], 1)
        self.assertEqual(stage1_report["lenient"]["error"]["num_failed_branches"], 1)
        self.assertEqual(stage1_report["strict"]["error"]["num_failed_branches"], 1)
        self.assertEqual(stage1_report["lenient"]["error"]["branches"][0]["source_type"], "tile_0000")
        self.assertEqual(stage1_report["strict"]["error"]["branches"][0]["source_type"], "tile_0000")

    def test_stage1_aggregate_report_stays_clean_when_all_branches_succeed(self) -> None:
        success_prediction = {"instances": [{"label": "single_arrow", "bbox": [1.0, 2.0, 3.0, 4.0], "keypoints": []}]}
        reports = [
            (
                "full",
                {
                    "lenient": {"ok": True, "prediction": success_prediction, "error": None, "recovered_prefix": False},
                    "strict": {"ok": True, "prediction": success_prediction, "error": None, "recovered_prefix": False},
                },
            ),
            (
                "tile",
                {
                    "lenient": {"ok": True, "prediction": success_prediction, "error": None, "recovered_prefix": True},
                    "strict": {"ok": True, "prediction": success_prediction, "error": None, "recovered_prefix": False},
                },
            ),
        ]
        runner = self._build_runner(reports)

        _raw_text, stage1_report, prediction = runner._predict_stage1_with_options(
            Image.new("RGB", (32, 32), color="black"),
            use_mixed_proposals=True,
        )

        self.assertEqual(len(prediction["instances"]), 1)
        self.assertTrue(stage1_report["lenient"]["ok"])
        self.assertTrue(stage1_report["strict"]["ok"])
        self.assertIsNone(stage1_report["lenient"]["error"])
        self.assertIsNone(stage1_report["strict"]["error"])
        self.assertEqual(stage1_report["generation"]["num_lenient_failed_branches"], 0)
        self.assertEqual(stage1_report["generation"]["num_strict_failed_branches"], 0)
        self.assertTrue(stage1_report["lenient"]["recovered_prefix"])


if __name__ == "__main__":
    unittest.main()
