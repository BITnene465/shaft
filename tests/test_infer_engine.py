from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch

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


def test_infer_engine_generate_does_not_emit_invalid_sampling_flags_for_greedy() -> None:

    model_config = InferModelConfig(
        model_type="smoke_vlm",
        model_name_or_path="unused",
        device="cpu",
        generation=InferGenerationConfig(max_new_tokens=8, do_sample=False),
    )
    engine = InferEngine.from_model_config(model_config)

    captured = {}

    class DummyGenerationConfig(SimpleNamespace):
        def clone(self) -> "DummyGenerationConfig":
            return DummyGenerationConfig(**self.__dict__)

    class DummyModel:
        def __init__(self):
            self.generation_config = DummyGenerationConfig(
                max_new_tokens=16,
                do_sample=True,
                top_p=0.95,
                temperature=0.5,
                top_k=7,
                repetition_penalty=1.2,
                eos_token_id=engine.tokenizer.eos_token_id,
                pad_token_id=engine.tokenizer.pad_token_id,
            )

        def to(self, _device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            captured["kwargs"] = kwargs
            return torch.tensor([[1, 2, 3]])

    engine.model = DummyModel()
    batch = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }
    engine._generate(batch=batch, generation=InferGenerationConfig(max_new_tokens=4, do_sample=False))

    gen_config = captured["kwargs"]["generation_config"]
    assert gen_config.do_sample is False
    assert gen_config.top_p == 1.0
    assert gen_config.top_k == 50
    assert gen_config.temperature == 1.0
