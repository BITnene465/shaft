from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from accelerate import init_empty_weights
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from shaft.infer import (
    InferEngineConfig,
    InferGenerationConfig,
    ShaftInferEngine,
    ShaftInferRequest,
)
from shaft.data import (
    DPOCollator,
    SFTCollator,
    SFTDataset,
    SFTRecord,
    ShaftDynamicBatchPlanner,
    ShaftDynamicBatchPlanningSpec,
    ShaftSamplePlan,
    ShaftSFTSampleCostProvider,
)
from shaft.model import MODEL_REGISTRY, build_model_meta
from shaft.template import ShaftChatRenderer, build_template


class _CountingProcessor:
    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.tokenizer = wrapped.tokenizer
        self.call_count = 0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.wrapped(*args, **kwargs)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model_type", "template_name", "model_path"),
    [
        ("qwen3vl", "qwen3vl", Path("models/Qwen3-VL-4B-Instruct")),
        ("qwen36vl", "qwen35vl", Path("models/Qwen3.6-27B")),
    ],
)
def test_qwen_vl_runtime_cost_matches_real_processor_and_collator(
    tmp_path: Path,
    model_type: str,
    template_name: str,
    model_path: Path,
) -> None:
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    tokenizer = processor.tokenizer
    model_adapter = build_model_meta(model_type).resolve_adapter(
        model_name_or_path=str(model_path)
    )
    template = build_template(template_name)

    for index, image_size in enumerate(((64, 64), (128, 512))):
        image_path = tmp_path / f"cost-{index}.png"
        Image.new("RGB", image_size, color=(index * 20, 30, 40)).save(image_path)
        messages = None
        if index == 1:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "first question"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
                {"role": "user", "content": [{"type": "text", "text": "second question"}]},
            ]
        plan = ShaftSamplePlan(
            {"integration": 1},
            {"integration": 1.0},
            strategy="concat",
            shuffle=False,
        )
        dataset = SFTDataset(
            {
                "integration": [
                    SFTRecord(
                        image_path=str(image_path),
                        target_text="one two three four",
                        dataset_name="integration",
                        messages=messages,
                        user_prompt="Return a compact answer.",
                    )
                ]
            },
            sample_plan=plan,
        )
        ref = plan.ref_at(0)
        common_kwargs = {
            "model_adapter": model_adapter,
            "template": template,
            "processor": processor,
            "tokenizer": tokenizer,
            "min_pixels": 16384,
            "max_pixels": 262144,
            "max_length": 256,
            "add_eos_token": True,
            "loss_scale_name": "default",
        }
        estimated = ShaftSFTSampleCostProvider(
            dataset=dataset,
            **common_kwargs,
        )(ref)
        actual = SFTCollator(
            include_targets_in_inputs=True,
            **common_kwargs,
        )([dataset[ref]])
        shifted_valid = actual["labels"][:, 1:].ne(-100)
        actual_loss_weight = (
            float(actual["loss_scale"][:, 1:][shifted_valid].sum())
            if "loss_scale" in actual
            else float(shifted_valid.sum())
        )

        assert estimated.llm_tokens == int(actual["attention_mask"].sum())
        assert estimated.supervised_tokens == int(shifted_valid.sum())
        assert estimated.loss_weight_sum == pytest.approx(actual_loss_weight)
        assert estimated.vision_patches == int(
            actual["image_grid_thw"].prod(dim=-1).sum()
        )


@pytest.mark.integration
def test_qwen3vl_dynamic_planner_hard_caps_match_heterogeneous_real_batches(
    tmp_path: Path,
) -> None:
    model_path = Path("models/Qwen3-VL-4B-Instruct")
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path=str(model_path)
    )
    template = build_template("qwen3vl")
    records: list[SFTRecord] = []
    for index, (image_size, prompt_length) in enumerate(
        (
            ((64, 64), 8),
            ((128, 256), 80),
            ((256, 128), 24),
            ((64, 512), 160),
        )
    ):
        image_path = tmp_path / f"heterogeneous-{index}.png"
        Image.new("RGB", image_size, color=(index * 30, 40, 50)).save(image_path)
        records.append(
            SFTRecord(
                image_path=str(image_path),
                target_text=f"answer-{index}",
                dataset_name="integration",
                sample_id=f"sample-{index}",
                user_prompt="x" * prompt_length,
            )
        )
    plan = ShaftSamplePlan(
        {"integration": len(records)},
        {"integration": 1.0},
        strategy="concat",
        shuffle=False,
    )
    dataset = SFTDataset({"integration": records}, sample_plan=plan)
    common_kwargs = {
        "model_adapter": model_adapter,
        "template": template,
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "min_pixels": 16384,
        "max_pixels": 262144,
        "max_length": 512,
        "add_eos_token": True,
        "loss_scale_name": "default",
    }
    cost_provider = ShaftSFTSampleCostProvider(
        dataset=dataset,
        **common_kwargs,
    )
    costs = tuple(cost_provider(plan.ref_at(index)) for index in range(len(plan)))
    max_padded_tokens = 3 * max(cost.llm_tokens for cost in costs)
    max_vision_patches = sum(cost.vision_patches for cost in costs)
    planning_spec = ShaftDynamicBatchPlanningSpec.from_plan(
        plan,
        optimizer_step_count=1,
        data_world_size=1,
        gradient_accumulation_steps=2,
        max_samples_per_microbatch=3,
        max_padded_tokens=max_padded_tokens,
        max_vision_patches=max_vision_patches,
        target_samples=4,
        target_supervised_tokens=None,
        planning_window=4,
        seed=7,
        rank_balance=True,
    )
    planner = ShaftDynamicBatchPlanner(
        plan=plan,
        cost_provider=cost_provider,
        spec=planning_spec,
    )
    collator = SFTCollator(include_targets_in_inputs=True, **common_kwargs)

    (optimizer_batch,) = tuple(planner.iter_optimizer_steps())
    local_batches = [
        local_batch
        for microstep in optimizer_batch.microsteps
        for local_batch in microstep.rank_microbatches
    ]
    assert sorted(len(batch.sample_refs) for batch in local_batches) == [2, 2]
    for local_batch in local_batches:
        actual = collator([dataset[ref] for ref in local_batch.sample_refs])
        assert actual["input_ids"].numel() == local_batch.padded_llm_tokens
        assert int(actual["image_grid_thw"].prod(dim=-1).sum()) == (
            local_batch.vision_patches
        )
        assert len(local_batch.sample_refs) <= planning_spec.max_samples_per_microbatch
        assert local_batch.padded_llm_tokens <= planning_spec.max_padded_tokens
        assert local_batch.vision_patches <= int(planning_spec.max_vision_patches)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model_type", "template_name", "model_path"),
    [
        ("qwen3vl", "qwen3vl", Path("models/Qwen3-VL-4B-Instruct")),
        ("qwen36vl", "qwen35vl", Path("models/Qwen3.6-27B")),
        ("qwen36vl", "qwen35vl_thinking", Path("models/Qwen3.6-27B")),
    ],
)
def test_qwen_vl_sft_multiround_supervision_uses_one_processor_call(
    tmp_path: Path,
    model_type: str,
    template_name: str,
    model_path: Path,
) -> None:
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")
    if not MODEL_REGISTRY.has(model_type):
        pytest.skip(f"Model adapter is not registered: {model_type}")

    image = Image.new("RGB", (64, 64), color=(240, 240, 240))
    base_processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    model_adapter = build_model_meta(model_type).resolve_adapter(
        model_name_or_path=str(model_path)
    )
    template = build_template(template_name)
    item = {
        "dataset_name": "integration",
        "sample_id": "multi-round",
        "image_path": str(tmp_path / "image.png"),
        "image": image,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "first"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer one"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ],
        "target_text": "answer two",
        "system_prompt": "",
        "user_prompt": "",
        "extra": {},
    }

    outputs = {}
    processors = {}
    for loss_scale_name in ("default", "last_round"):
        processor = _CountingProcessor(base_processor)
        processors[loss_scale_name] = processor
        collator = SFTCollator(
            model_adapter=model_adapter,
            template=template,
            processor=processor,
            tokenizer=base_processor.tokenizer,
            loss_scale_name=loss_scale_name,
        )
        outputs[loss_scale_name] = collator([item])

    assert processors["default"].call_count == 1
    assert processors["last_round"].call_count == 1
    assert torch.equal(outputs["default"]["input_ids"], outputs["last_round"]["input_ids"])
    assert torch.equal(outputs["default"]["pixel_values"], outputs["last_round"]["pixel_values"])
    assert int(outputs["default"]["labels"].ne(-100).sum()) > int(
        outputs["last_round"]["labels"].ne(-100).sum()
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model_type", "template_name", "model_path"),
    [
        ("qwen3vl", "qwen3vl", Path("models/Qwen3-VL-4B-Instruct")),
        ("qwen36vl", "qwen35vl", Path("models/Qwen3.6-27B")),
    ],
)
def test_qwen_vl_dpo_reuses_one_processed_prompt_for_both_completions(
    tmp_path: Path,
    model_type: str,
    template_name: str,
    model_path: Path,
) -> None:
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")
    if not MODEL_REGISTRY.has(model_type):
        pytest.skip(f"Model adapter is not registered: {model_type}")

    image = Image.new("RGB", (64, 64), color=(240, 240, 240))
    base_processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    processor = _CountingProcessor(base_processor)
    model_adapter = build_model_meta(model_type).resolve_adapter(
        model_name_or_path=str(model_path)
    )
    collator = DPOCollator(
        model_adapter=model_adapter,
        template=build_template(template_name),
        processor=processor,
        tokenizer=base_processor.tokenizer,
    )
    item = {
        "dataset_name": "integration",
        "sample_id": "dpo-multi-round",
        "image_path": str(tmp_path / "image.png"),
        "image": image,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "first"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer one"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ],
        "chosen_text": "preferred answer",
        "rejected_text": "rejected answer",
        "system_prompt": "",
        "user_prompt": "",
        "extra": {},
    }

    output = collator([item])

    assert processor.call_count == 1
    assert output["input_ids"].shape[0] == 2
    assert output["completion_mask"].shape == output["input_ids"].shape
    assert output["image_grid_thw"].shape[0] == 2
    midpoint = output["pixel_values"].shape[0] // 2
    assert midpoint > 0
    assert torch.equal(output["pixel_values"][:midpoint], output["pixel_values"][midpoint:])


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
        renderer=ShaftChatRenderer.from_components(
            processor=processor,
            tokenizer=getattr(processor, "tokenizer", None),
        ),
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
