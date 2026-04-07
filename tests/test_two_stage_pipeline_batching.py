from __future__ import annotations

import unittest
from types import SimpleNamespace

from PIL import Image

from vlm_structgen.domains.arrow.infer.two_stage import Stage2PredictionResult, TwoStageInferenceRunner


class FakeStage1Runner:
    def __init__(self, outputs_per_image: list[tuple[str, dict]]) -> None:
        self.outputs_per_image = list(outputs_per_image)
        self.settings = SimpleNamespace(batch_size=2)
        self.batch_sizes: list[int] = []
        self._offset = 0

    def predict_batch(self, images, *, max_new_tokens=None):
        del max_new_tokens
        self.batch_sizes.append(len(images))
        batch_outputs = self.outputs_per_image[self._offset : self._offset + len(images)]
        self._offset += len(images)
        return batch_outputs


class FakeStage2Runner:
    def __init__(self, *, fail_request_indices: set[int] | None = None) -> None:
        self.adapter = SimpleNamespace(num_bins=1000)
        self.batch_size = 1
        self.request_counts: list[int] = []
        self.fail_request_indices = set(fail_request_indices or set())

    def predict_batch(self, requests, *, max_new_tokens=None):
        del max_new_tokens
        self.request_counts.append(len(requests))
        results: list[Stage2PredictionResult] = []
        for request in requests:
            should_fail = int(request.index) in self.fail_request_indices
            results.append(
                Stage2PredictionResult(
                    index=int(request.index),
                    crop_box=list(request.crop_box),
                    raw_text="{}",
                    report={
                        "lenient": {
                            "ok": not should_fail,
                            "prediction": None if should_fail else {"keypoints": [[1.0, 2.0], [3.0, 4.0]]},
                            "error": "stage2 failed" if should_fail else None,
                            "recovered_prefix": False,
                        },
                        "strict": {
                            "ok": not should_fail,
                            "prediction": None if should_fail else {"keypoints": [[1.0, 2.0], [3.0, 4.0]]},
                            "error": "stage2 failed" if should_fail else None,
                            "recovered_prefix": False,
                        },
                    },
                )
            )
        return results


class TwoStagePipelineBatchingTests(unittest.TestCase):
    def test_predict_batch_batches_stage1_and_fanins_stage2_requests(self) -> None:
        def _stage1_output(label: str, bbox: list[float]) -> tuple[str, dict]:
            prediction = {"instances": [{"label": label, "bbox": bbox, "keypoints": []}]}
            return (
                "raw-stage1",
                {
                    "generation": {"closed_json_payload": True},
                    "lenient": {"ok": True, "prediction": prediction, "error": None, "recovered_prefix": False},
                    "strict": {"ok": True, "prediction": prediction, "error": None, "recovered_prefix": False},
                },
            )

        stage1_runner = FakeStage1Runner(
            [
                _stage1_output("single_arrow", [2.0, 2.0, 10.0, 10.0]),
                _stage1_output("double_arrow", [4.0, 4.0, 12.0, 12.0]),
                _stage1_output("single_arrow", [6.0, 6.0, 14.0, 14.0]),
            ]
        )
        stage2_runner = FakeStage2Runner()
        runner = TwoStageInferenceRunner(
            stage1_runner=stage1_runner,
            stage2_runner=stage2_runner,
            infer_config=object(),
            padding_ratio=0.3,
        )

        reports = runner.predict_batch(
            [
                Image.new("RGB", (32, 32), color="black"),
                Image.new("RGB", (32, 32), color="black"),
                Image.new("RGB", (32, 32), color="black"),
            ],
            stage1_batch_size=2,
            stage2_batch_size=8,
        )

        self.assertEqual(stage1_runner.batch_sizes, [2, 1])
        self.assertEqual(stage2_runner.request_counts, [3])
        self.assertEqual(stage2_runner.batch_size, 8)
        self.assertEqual(len(reports), 3)
        self.assertTrue(all(len(report["final_prediction"]["instances"]) == 1 for report in reports))
        self.assertTrue(all(len(report["stage2_results"]) == 1 for report in reports))

    def test_stage2_failed_request_is_not_kept_in_final_prediction(self) -> None:
        prediction = {"instances": [{"label": "single_arrow", "bbox": [2.0, 2.0, 10.0, 10.0], "keypoints": []}]}
        stage1_runner = FakeStage1Runner(
            [
                (
                    "raw-stage1",
                    {
                        "generation": {"closed_json_payload": True},
                        "lenient": {"ok": True, "prediction": prediction, "error": None, "recovered_prefix": False},
                        "strict": {"ok": True, "prediction": prediction, "error": None, "recovered_prefix": False},
                    },
                )
            ]
        )
        stage2_runner = FakeStage2Runner(fail_request_indices={0})
        runner = TwoStageInferenceRunner(
            stage1_runner=stage1_runner,
            stage2_runner=stage2_runner,
            infer_config=object(),
            padding_ratio=0.3,
        )

        report = runner.predict(Image.new("RGB", (32, 32), color="black"))

        self.assertEqual(report["final_prediction"]["instances"], [])
        self.assertEqual(len(report["stage2_results"]), 1)
        self.assertFalse(report["stage2_results"][0]["strict"]["ok"])


if __name__ == "__main__":
    unittest.main()
