from __future__ import annotations

from pathlib import Path

from PIL import Image

from shaft.infer import InferEngine, InferGenerationConfig, InferModelConfig, InferRequest


def test_smoke_vlm_engine_can_generate(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image)

    model_config = InferModelConfig(
        model_type="smoke_vlm",
        model_name_or_path="unused",
        device="cpu",
        generation=InferGenerationConfig(max_new_tokens=8, do_sample=False),
    )
    engine = InferEngine.from_model_config(model_config)
    response = engine.run(
        InferRequest(
            image_path=str(image),
            system_prompt="you are tester",
            user_prompt="return json",
        )
    )
    assert isinstance(response.text, str)
    assert "return json" in response.prompt
    assert isinstance(response.output_ids, list)
