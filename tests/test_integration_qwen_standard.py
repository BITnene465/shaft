from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from shaft.infer import InferEngine, InferGenerationConfig, InferModelConfig, InferRequest
from shaft.model import MODEL_REGISTRY


@pytest.mark.integration
@pytest.mark.manual
def test_qwen3vl_standard_model_load_and_chat() -> None:
    model_path = Path("models/Qwen3-VL-4B-Instruct")
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")
    if not MODEL_REGISTRY.has("qwen3vl"):
        pytest.skip("qwen3vl model adapter is not registered in current runtime.")

    image_path = Path(__file__).parent.parent / "temp" / "unit_smoke_image.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        Image.new("RGB", (32, 32), color=(240, 240, 240)).save(image_path)

    engine = InferEngine.from_model_config(
        InferModelConfig(
            model_type="qwen3vl",
            model_name_or_path=str(model_path),
            template="qwen3vl",
            device="cpu",
            attn_implementation=None,
            torch_dtype="float32",
            generation=InferGenerationConfig(
                max_new_tokens=32,
                do_sample=False,
            ),
        )
    )

    response = engine.run(
        InferRequest(
            image_path=str(image_path),
            system_prompt="You are an accurate image description assistant.",
            user_prompt="请只回答：图片里有一张桌子。",
        )
    )

    assert isinstance(response.text, str)
    assert isinstance(response.output_ids, list)
    assert response.text.strip() != ""
