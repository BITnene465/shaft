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
    def __init__(self) -> None:
        self.adapter = SimpleNamespace(num_bins=1000)
        self.batch_size = 1
        self.request_counts: list[int] = []

    def predict_batch(self, requests, *, max_new_tokens=None):
        del max_new_tokens
        self.request_counts.append(len(requests))
        results: list[Stage2PredictionResult] = []
        for request in requests:
            results.append(
                Stage2PredictionResult(
                    index=int(request.index),
                    crop_box=list(request.crop_box),
                    raw_text="{}",
                    report={
                        "lenient": {
                            "ok": True,
                            "prediction": {"keypoints": [[1.0, 2.0], [3.0, 4.0]]},
                            "error": None,
                            "recovered_prefix": False,
                        },
                        "strict": {
                            "ok": True,
                            "prediction": {"keypoints": [[1.0, 2.0], [3.0, 4.0]]},
                            "error": None,
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


if __name__ == "__main__":
    unittest.main()
