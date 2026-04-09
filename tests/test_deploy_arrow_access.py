from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image

from deploy.arrow.config import load_arrow_config
from deploy.arrow.pipeline import ArrowTwoStagePipeline


class FakeVLLMClient:
    def __init__(self, *, base_url: str, api_key: str = "EMPTY") -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.calls: list[dict[str, object]] = []

    def generate_with_image(
        self,
        *,
        model: str,
        prompt: str,
        image,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "image": image,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        if model == "grounding_arrow":
            return '[{"label":"single_arrow","bbox_2d":[100,200,300,400]}]'
        if model == "keypoint_sequence_arrow":
            return '{"keypoints_2d":[[10,20],[30,40]]}'
        raise AssertionError(f"unexpected model: {model}")


class DeployArrowAccessTest(unittest.TestCase):
    def test_pipeline_uses_configured_routes_and_greedy_defaults(self) -> None:
        fake_client = FakeVLLMClient(base_url="http://127.0.0.1:8001/v1")
        config = load_arrow_config()

        with patch("deploy.arrow.pipeline.ArrowVLLMClient", return_value=fake_client):
            pipeline = ArrowTwoStagePipeline(base_url="http://127.0.0.1:8001/v1", config=config)
            image = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
            result = pipeline.predict_two_stage(image)

        self.assertEqual(len(fake_client.calls), 2)
        self.assertEqual(fake_client.calls[0]["model"], "grounding_arrow")
        self.assertEqual(fake_client.calls[1]["model"], "keypoint_sequence_arrow")
        self.assertEqual(fake_client.calls[0]["max_tokens"], 2048)
        self.assertEqual(fake_client.calls[1]["max_tokens"], 64)
        self.assertEqual(fake_client.calls[0]["temperature"], 0.0)
        self.assertEqual(fake_client.calls[1]["temperature"], 0.0)
        self.assertEqual(fake_client.calls[0]["top_p"], 1.0)
        self.assertEqual(fake_client.calls[1]["top_p"], 1.0)
        self.assertEqual(result.stage1.decoded["instances"][0]["label"], "single_arrow")
        self.assertEqual(len(result.final_prediction["instances"]), 1)
        self.assertEqual(len(result.final_prediction["instances"][0]["keypoints"]), 2)

    def test_pipeline_overrides_sampling_when_enabled(self) -> None:
        fake_client = FakeVLLMClient(base_url="http://127.0.0.1:8001/v1")
        config = load_arrow_config()
        config = type(config)(
            protocol=config.protocol,
            stage1=type(config.stage1)(
                route=config.stage1.route,
                prompt=config.stage1.prompt,
                max_tokens=config.stage1.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            ),
            stage2=type(config.stage2)(
                route=config.stage2.route,
                prompt=config.stage2.prompt,
                max_tokens=config.stage2.max_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.8,
            ),
            padding_ratio=config.padding_ratio,
        )

        with patch("deploy.arrow.pipeline.ArrowVLLMClient", return_value=fake_client):
            pipeline = ArrowTwoStagePipeline(base_url="http://127.0.0.1:8001/v1", config=config)
            image = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
            pipeline.predict_two_stage(image)

        self.assertEqual(fake_client.calls[0]["temperature"], 0.7)
        self.assertEqual(fake_client.calls[0]["top_p"], 0.9)
        self.assertEqual(fake_client.calls[1]["temperature"], 0.6)
        self.assertEqual(fake_client.calls[1]["top_p"], 0.8)


if __name__ == "__main__":
    unittest.main()
