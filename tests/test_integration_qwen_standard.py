from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from accelerate import init_empty_weights
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from shaft.infer import (
    InferEngineConfig,
    InferGenerationConfig,
    ShaftInferEngine,
    ShaftInferRequest,
)
from shaft.model import MODEL_REGISTRY
from shaft.template import build_template


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

    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
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
        ShaftInferRequest(
            image_path=str(image_path),
            system_prompt="You are an accurate image description assistant.",
            user_prompt="请只回答：图片里有一张桌子。",
        )
    )

    assert isinstance(response.text, str)
    assert isinstance(response.output_ids, list)
    assert response.text.strip() != ""


@pytest.mark.integration
@pytest.mark.manual
def test_qwen36vl_processor_template_disables_thinking_by_default() -> None:
    model_path = Path("models/Qwen3.6-27B")
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]
    missing_files = [name for name in required_files if not (model_path / name).exists()]
    if missing_files:
        pytest.skip(f"Qwen3.6 model path is incomplete: missing {missing_files}")
    if importlib.util.find_spec("transformers.models.qwen3_5") is None:
        pytest.skip("Current Transformers build does not include qwen3_5 support.")
    if not MODEL_REGISTRY.has("qwen36vl"):
        pytest.skip("qwen36vl model adapter is not registered in current runtime.")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    template = build_template("qwen35vl")
    rendered = template.apply_chat_template(
        processor=processor,
        tokenizer=getattr(processor, "tokenizer", None),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Return compact JSON only."},
                ],
            }
        ],
    )

    assert "<|im_start|>assistant" in rendered
    assert "<think>\n\n</think>" in rendered


@pytest.mark.integration
@pytest.mark.manual
def test_qwen36vl_empty_model_architecture_loads() -> None:
    model_path = Path("models/Qwen3.6-27B")
    config_path = model_path / "config.json"
    if not config_path.exists():
        pytest.skip(f"Qwen3.6 config not found: {config_path}")
    if importlib.util.find_spec("transformers.models.qwen3_5") is None:
        pytest.skip("Current Transformers build does not include qwen3_5 support.")
    if not MODEL_REGISTRY.has("qwen36vl"):
        pytest.skip("qwen36vl model adapter is not registered in current runtime.")

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(
            config,
            trust_remote_code=True,
        )

    assert type(model).__name__ == "Qwen3_5ForConditionalGeneration"
    assert next(model.parameters()).device.type == "meta"
    nested_model = getattr(model, "model", None)
    assert hasattr(model, "language_model") or hasattr(nested_model, "language_model")
    assert hasattr(model, "visual") or hasattr(nested_model, "visual")
