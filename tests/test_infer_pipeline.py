from __future__ import annotations

from shaft.infer.engine import InferRequest, InferResponse
from shaft.infer.pipeline import InferPipeline
from shaft.infer.schema import InferStageConfig


class _DummyEngine:
    def __init__(self, tag: str):
        self.tag = tag

    def run(self, request: InferRequest) -> InferResponse:
        text = f"{self.tag}:{request.user_prompt}"
        return InferResponse(text=text, prompt=request.user_prompt, output_ids=[1, 2, 3])


def test_multistage_multi_engine_orchestration() -> None:
    pipeline = InferPipeline(
        engines={
            "det": _DummyEngine("det"),
            "struct": _DummyEngine("struct"),
        },
        stages=[
            InferStageConfig(
                name="stage1",
                engine="det",
                output_key="det_out",
                user_prompt_template="locate arrows",
            ),
            InferStageConfig(
                name="stage2",
                engine="struct",
                output_key="struct_out",
                user_prompt_template="use {det_out} and refine",
            ),
        ],
    )
    outputs = pipeline.run(image_path="/tmp/fake.png", inputs={"task": "arrow"})
    assert "det_out" in outputs
    assert "struct_out" in outputs
    assert outputs["det_out"].startswith("det:")
    assert outputs["det_out"] in outputs["struct_out"]
    assert len(outputs["__trace__"]) == 2
