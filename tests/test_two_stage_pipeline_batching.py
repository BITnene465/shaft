from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from PIL import Image

from vlm_structgen.domains.arrow.data.two_stage import _build_stage2_record
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
        self.request_aspect_ratios: list[float] = []

    def predict_batch(self, requests, *, max_new_tokens=None):
        del max_new_tokens
        self.request_counts.append(len(requests))
        results: list[Stage2PredictionResult] = []
        for request in requests:
            width = max(int(request.crop_image.width), 1)
            height = max(int(request.crop_image.height), 1)
            self.request_aspect_ratios.append(max(float(width) / float(height), float(height) / float(width)))
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
        self.assertTrue(all(ratio <= 180.0 for ratio in stage2_runner.request_aspect_ratios))

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

    def test_extreme_slender_crop_is_padded_before_stage2_batch(self) -> None:
        prediction = {"instances": [{"label": "single_arrow", "bbox": [10.0, 10.0, 11.0, 694.0], "keypoints": []}]}
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
        stage2_runner = FakeStage2Runner()
        runner = TwoStageInferenceRunner(
            stage1_runner=stage1_runner,
            stage2_runner=stage2_runner,
            infer_config=object(),
            padding_ratio=0.0,
            stage2_max_crop_aspect_ratio=180.0,
        )

        report = runner.predict(Image.new("RGB", (1024, 1024), color="black"))

        self.assertEqual(len(report["final_prediction"]["instances"]), 1)
        self.assertEqual(stage2_runner.request_counts, [1])
        self.assertLessEqual(stage2_runner.request_aspect_ratios[0], 180.0)

    def test_training_and_infer_stage2_crop_settings_stay_aligned(self) -> None:
        image = Image.new("RGB", (1024, 1024), color="black")
        record = {
            "sample_id": "sample_0001",
            "image_path": "unused.png",
            "instances": [
                {
                    "label": "single_arrow",
                    "bbox": [10.0, 10.0, 11.0, 694.0],
                    "keypoints": [[10.0, 10.0], [11.0, 694.0]],
                }
            ],
        }
        instance = record["instances"][0]
        with TemporaryDirectory() as tmp_dir:
            stage2_record = _build_stage2_record(
                record,
                instance,
                image=image,
                split="val",
                target_index=0,
                sample_suffix="__pad300",
                hint_bbox=list(instance["bbox"]),
                output_dir=Path(tmp_dir),
                padding_ratio=0.3,
                num_bins=1000,
                augmentation={"padding_ratio": 0.3},
            )

        stage1_prediction = {"instances": [{"label": "single_arrow", "bbox": list(instance["bbox"]), "keypoints": []}]}
        runner = TwoStageInferenceRunner(
            stage1_runner=SimpleNamespace(),
            stage2_runner=SimpleNamespace(adapter=SimpleNamespace(num_bins=1000)),
            infer_config=object(),
            padding_ratio=0.3,
            stage2_max_crop_aspect_ratio=180.0,
        )
        requests, _ = runner._build_stage2_requests_for_image(
            image_index=0,
            image=image,
            stage1_prediction=stage1_prediction,
            start_index=0,
        )

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].crop_box, stage2_record["crop_box"])
        self.assertEqual(requests[0].label, stage2_record["instances"][0]["label"])


if __name__ == "__main__":
    unittest.main()
