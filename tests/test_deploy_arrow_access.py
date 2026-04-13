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
        mm_processor_kwargs: dict[str, int] | None = None,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "image": image,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "mm_processor_kwargs": mm_processor_kwargs,
            }
        )
        if "bbox_2d" in prompt:
            return '[{"label":"single_arrow","bbox_2d":[100,200,300,400]}]'
        if "keypoints_2d" in prompt:
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
        self.assertEqual(fake_client.calls[0]["model"], config.stage1.route)
        self.assertEqual(fake_client.calls[1]["model"], config.stage2.route)
        self.assertEqual(fake_client.calls[0]["max_tokens"], 2048)
        self.assertEqual(fake_client.calls[1]["max_tokens"], 64)
        self.assertEqual(fake_client.calls[0]["temperature"], 0.0)
        self.assertEqual(fake_client.calls[1]["temperature"], 0.0)
        self.assertEqual(fake_client.calls[0]["top_p"], 1.0)
        self.assertEqual(fake_client.calls[1]["top_p"], 1.0)
        self.assertIsNotNone(fake_client.calls[0]["mm_processor_kwargs"])
        self.assertIsNotNone(fake_client.calls[1]["mm_processor_kwargs"])
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
                min_pixels=config.stage1.min_pixels,
                max_pixels=config.stage1.max_pixels,
            ),
            stage2=type(config.stage2)(
                route=config.stage2.route,
                prompt=config.stage2.prompt,
                max_tokens=config.stage2.max_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.8,
                min_pixels=config.stage2.min_pixels,
                max_pixels=config.stage2.max_pixels,
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
