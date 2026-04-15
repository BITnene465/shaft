from __future__ import annotations

from pathlib import Path

import pytest

from shaft.infer import InferEngineConfig, InferGenerationConfig, InferPipelineConfig, InferStageConfig, ShaftInferPipeline


def _fixture_images(*, count: int = 2) -> list[Path]:
    fixture_dir = Path(__file__).parent / "fixtures" / "infer_images"
    if not fixture_dir.exists():
        pytest.skip(f"fixture image directory not found: {fixture_dir}")
    candidates = sorted(
        [path for path in fixture_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if len(candidates) < count:
        pytest.skip(f"Not enough fixture images under {fixture_dir}, need>={count}, got={len(candidates)}")
    return candidates[:count]


def _build_smoke_pipeline(*, multi_stage: bool) -> ShaftInferPipeline:
    cfg = InferPipelineConfig(
        engines={
            "smoke": InferEngineConfig(
                model_type="smoke_vlm",
                model_name_or_path="unused",
                backend="hf_local",
                device="cpu",
                generation=InferGenerationConfig(
                    max_new_tokens=24,
                    do_sample=False,
                    temperature=0.0,
                ),
            )
        },
        stages=[
            InferStageConfig(
                name="stage1",
                engine="smoke",
                output_key="stage1_out",
                user_prompt_template="请识别图像并输出简短结构化文本。",
                codec="text",
                max_retries=0,
                fail_fast=True,
            ),
            *(
                [
                    InferStageConfig(
                        name="stage2",
                        engine="smoke",
                        output_key="stage2_out",
                        user_prompt_template="基于 stage1 输出继续整理：{stage1_out}",
                        codec="text",
                        max_retries=0,
                        fail_fast=True,
                    )
                ]
                if multi_stage
                else []
            ),
        ],
    )
    return ShaftInferPipeline.from_config(cfg)


@pytest.mark.integration
def test_infer_single_stage_with_val_images() -> None:
    images = _fixture_images(count=1)
    pipeline = _build_smoke_pipeline(multi_stage=False)
    outputs = pipeline.run(image_path=str(images[0]), inputs={})

    assert "stage1_out" in outputs
    assert isinstance(outputs["stage1_out"], str)
    assert len(outputs["__trace__"]) == 1
    assert outputs["__trace__"][0].success is True


@pytest.mark.integration
def test_infer_multi_stage_with_val_images() -> None:
    images = _fixture_images(count=2)
    pipeline = _build_smoke_pipeline(multi_stage=True)
    outputs = pipeline.run(
        image_path=str(images[0]),
        inputs={"aux_image_path": str(images[1])},
    )

    assert "stage1_out" in outputs
    assert "stage2_out" in outputs
    assert isinstance(outputs["stage2_out"], str)
    assert len(outputs["__trace__"]) == 2
    assert all(trace.success for trace in outputs["__trace__"])
