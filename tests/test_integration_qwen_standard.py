from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

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
    ShaftBatchPlanner,
    ShaftBatchPlanningSpec,
    ShaftSamplePlan,
    ShaftSFTSampleCostProvider,
)
from shaft.model import MODEL_REGISTRY, build_model_meta
from shaft.observability import TRAINING_EFFICIENCY_SCHEMA_VERSION
from shaft.template import ShaftChatRenderer, build_template
from shaft.training import checkpoint_has_batch_planning_state
from tests.support.qwen_training_gate import (
    prepare_qwen_training_dataset,
    prepare_tiny_qwen35_training_assets,
    write_qwen_training_gate_config,
)


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


def _run_qwen_training_gate(repo_root: Path, config_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            "scripts/train.py",
            "sft",
            "--config",
            str(config_path),
        ],
        cwd=repo_root,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "Qwen training release gate failed.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _assert_nested_state_equal(left, right, *, path: str = "root") -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right), path
        assert left.dtype == right.dtype, path
        assert tuple(left.shape) == tuple(right.shape), path
        assert torch.equal(left.cpu(), right.cpu()), path
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray), path
        np.testing.assert_array_equal(left, right, err_msg=path)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        assert set(left) == set(right), path
        for key in sorted(left, key=str):
            _assert_nested_state_equal(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert isinstance(left, type(right)) or isinstance(right, type(left)), path
        assert len(left) == len(right), path
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            _assert_nested_state_equal(
                left_value,
                right_value,
                path=f"{path}[{index}]",
            )
        return
    assert left == right, path


def _normalized_trainer_state(checkpoint_dir: Path) -> dict:
    payload = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    best_checkpoint = payload.get("best_model_checkpoint")
    if best_checkpoint:
        payload["best_model_checkpoint"] = Path(best_checkpoint).name
    timing_keys = (
        "runtime",
        "samples_per_second",
        "steps_per_second",
        "jit_compilation_time",
    )
    payload["log_history"] = [
        {
            key: value
            for key, value in entry.items()
            if not key.startswith("efficiency/")
            and not any(key.endswith(suffix) for suffix in timing_keys)
        }
        for entry in payload.get("log_history", [])
    ]
    return payload


def _assert_checkpoint_state_equivalent(
    fresh_checkpoint: Path,
    resumed_checkpoint: Path,
    *,
    weight_filename: str,
) -> None:
    fresh_completion = fresh_checkpoint / "shaft_batch_planning_complete.json"
    resumed_completion = resumed_checkpoint / "shaft_batch_planning_complete.json"
    assert fresh_completion.is_file() == resumed_completion.is_file()
    if fresh_completion.is_file():
        assert checkpoint_has_batch_planning_state(fresh_checkpoint)
        assert checkpoint_has_batch_planning_state(resumed_checkpoint)
        assert json.loads(fresh_completion.read_text(encoding="utf-8")) == json.loads(
            resumed_completion.read_text(encoding="utf-8")
        )
    else:
        assert not checkpoint_has_batch_planning_state(fresh_checkpoint)
        assert not checkpoint_has_batch_planning_state(resumed_checkpoint)
    fresh_weights = (fresh_checkpoint / weight_filename).read_bytes()
    resumed_weights = (resumed_checkpoint / weight_filename).read_bytes()
    assert hashlib.sha256(fresh_weights).digest() == hashlib.sha256(
        resumed_weights
    ).digest()

    for filename in ("optimizer.pt", "scheduler.pt"):
        _assert_nested_state_equal(
            torch.load(fresh_checkpoint / filename, map_location="cpu", weights_only=True),
            torch.load(resumed_checkpoint / filename, map_location="cpu", weights_only=True),
            path=filename,
        )
    fresh_rng_files = tuple(sorted(path.name for path in fresh_checkpoint.glob("rng_state*.pth")))
    resumed_rng_files = tuple(
        sorted(path.name for path in resumed_checkpoint.glob("rng_state*.pth"))
    )
    assert fresh_rng_files == resumed_rng_files
    assert fresh_rng_files
    for filename in fresh_rng_files:
        _assert_nested_state_equal(
            torch.load(fresh_checkpoint / filename, map_location="cpu", weights_only=False),
            torch.load(resumed_checkpoint / filename, map_location="cpu", weights_only=False),
            path=filename,
        )
    assert _normalized_trainer_state(fresh_checkpoint) == _normalized_trainer_state(
        resumed_checkpoint
    )
    assert _restore_efficiency_snapshot_set(fresh_checkpoint) == (
        _restore_efficiency_snapshot_set(resumed_checkpoint)
    )


def _restore_efficiency_snapshot_set(checkpoint_dir: Path) -> dict:
    global_step = int(checkpoint_dir.name.rsplit("-", 1)[-1])
    with tempfile.TemporaryDirectory(prefix="shaft-efficiency-restore-") as directory:
        output_path = Path(directory) / "restored.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=2",
                "tests/support/distributed_efficiency_checkpoint_validate.py",
                str(checkpoint_dir),
                str(global_step),
                str(output_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "Efficiency checkpoint restore validation failed.\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return json.loads(output_path.read_text(encoding="utf-8"))


def _assert_full_hf_export_reloads(export_dir: Path, *, expected_type: str) -> None:
    processor = AutoProcessor.from_pretrained(
        export_dir,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_kwargs = {
        "local_files_only": True,
        "dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForImageTextToText.from_pretrained(export_dir, **load_kwargs)
    assert type(model).__name__ == expected_type
    assert next(model.parameters()).device.type == device.type
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in processor(text=["hello"], return_tensors="pt").items()
    }
    with torch.no_grad():
        logits = model(**inputs).logits
    assert torch.isfinite(logits).all()
    del logits, inputs, model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _assert_standard_qwen_peft_export_reloads(
    *,
    base_model_path: Path,
    export_dir: Path,
) -> None:
    processor = AutoProcessor.from_pretrained(
        export_dir,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base_model, export_dir, is_trainable=False)
    inputs = {
        key: value.to("cuda:0") if torch.is_tensor(value) else value
        for key, value in processor(text=["hello"], return_tensors="pt").items()
    }
    with torch.no_grad():
        logits = model(**inputs).logits
    assert torch.isfinite(logits).all()
    del logits, inputs, model, base_model, processor
    torch.cuda.empty_cache()


def _validate_qwen_peft_export(
    repo_root: Path,
    *,
    export_path: Path,
    model_type: str,
    model_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export.py",
            "validate",
            "--path",
            str(export_path),
            "--finetune-mode",
            "lora",
            "--model-type",
            model_type,
            "--model-name-or-path",
            str(model_path),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "Qwen PEFT export validation failed.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


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
            media_snapshot_id="integration-media-v1",
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
def test_qwen3vl_bounded_planner_hard_caps_match_heterogeneous_real_batches(
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
    dataset = SFTDataset(
        {"integration": records},
        sample_plan=plan,
        media_snapshot_id="integration-media-v1",
    )
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
    max_tokens_per_microbatch = 2 * max(cost.llm_tokens for cost in costs)
    vision_patch_budget = sum(cost.vision_patches for cost in costs)
    planning_spec = ShaftBatchPlanningSpec(
        data_world_size=2,
        buffer_size=4,
        per_device_microbatch_size=2,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
        resource_budgets=(("vision_patches", vision_patch_budget),),
        seed=7,
        sample_schedule_fingerprint=plan.schedule.fingerprint,
        cost_fingerprint=cost_provider.fingerprint,
    )
    planner = ShaftBatchPlanner(
        schedule=plan.schedule,
        cost_provider=cost_provider,
        spec=planning_spec,
    )
    collator = SFTCollator(include_targets_in_inputs=True, **common_kwargs)

    microstep = planner.next_global_microbatch()
    local_batches = microstep.rank_microbatches
    assert len(local_batches) == 2
    assert all(batch.sample_refs for batch in local_batches)
    for local_batch in local_batches:
        actual = collator([dataset[ref] for ref in local_batch.sample_refs])
        assert actual["input_ids"].numel() == local_batch.padded_llm_tokens
        assert int(actual["image_grid_thw"].prod(dim=-1).sum()) == (
            local_batch.vision_patches
        )
        assert len(local_batch.sample_refs) == planning_spec.per_device_microbatch_size
        assert local_batch.padded_llm_tokens <= planning_spec.max_tokens_per_microbatch
        assert local_batch.vision_patches <= int(
            planning_spec.resource_budget("vision_patches")
        )


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
def test_qwen35_qwen36_two_rank_train_save_and_exact_resume_release_gate(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    if os.environ.get("SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE") != "1":
        pytest.skip("Set SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE=1 to run the CUDA release gate.")
    if torch.cuda.device_count() < 2:
        pytest.skip("The Qwen training release gate requires two visible CUDA devices.")
    processor_source = Path("models/Qwen3.6-27B")
    if not (processor_source / "preprocessor_config.json").is_file():
        pytest.skip(f"Qwen3.6 processor assets not found: {processor_source}")

    model_dir, dataset_path = prepare_tiny_qwen35_training_assets(
        tmp_path,
        processor_source=processor_source,
    )
    qwen35_output = tmp_path / "qwen35-fixed-fresh"
    qwen35_config = write_qwen_training_gate_config(
        tmp_path / "qwen35-fixed.yaml",
        model_type="qwen35vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=qwen35_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
    )
    _run_qwen_training_gate(repo_root, qwen35_config)
    qwen35_resumed_output = tmp_path / "qwen35-fixed-resumed"
    qwen35_resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen35-fixed-resumed.yaml",
        model_type="qwen35vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=qwen35_resumed_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=qwen35_output / "checkpoint-1",
    )
    _run_qwen_training_gate(repo_root, qwen35_resumed_config)

    fresh_output = tmp_path / "qwen36-fresh"
    fresh_config = write_qwen_training_gate_config(
        tmp_path / "qwen36-fresh.yaml",
        model_type="qwen36vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=fresh_output,
        layout="varlen",
        packing="greedy",
        steps=2,
        save_steps=1,
    )
    _run_qwen_training_gate(repo_root, fresh_config)

    resumed_output = tmp_path / "qwen36-resumed"
    resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen36-resumed.yaml",
        model_type="qwen36vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=resumed_output,
        layout="varlen",
        packing="greedy",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=fresh_output / "checkpoint-1",
    )
    _run_qwen_training_gate(repo_root, resumed_config)

    _assert_checkpoint_state_equivalent(
        qwen35_output / "checkpoint-2",
        qwen35_resumed_output / "checkpoint-2",
        weight_filename="model.safetensors",
    )
    _assert_checkpoint_state_equivalent(
        fresh_output / "checkpoint-2",
        resumed_output / "checkpoint-2",
        weight_filename="model.safetensors",
    )
    _assert_full_hf_export_reloads(
        qwen35_output / "best",
        expected_type="Qwen3_5ForConditionalGeneration",
    )
    _assert_full_hf_export_reloads(
        fresh_output / "best",
        expected_type="Qwen3_5ForConditionalGeneration",
    )

    for output_dir, expected_steps in (
        (qwen35_output, 2),
        (qwen35_resumed_output, 2),
        (fresh_output, 2),
        (resumed_output, 2),
    ):
        summary = json.loads(
            (output_dir / "shaft_training_efficiency.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["schema_version"] == TRAINING_EFFICIENCY_SCHEMA_VERSION
        assert summary["complete_history"] is True
        assert summary["aggregate"]["optimizer_steps"] == expected_steps
        assert summary["aggregate"]["device_timing_steps"] == expected_steps
        assert summary["aggregate"]["device_training_seconds"] > 0
        assert summary["aggregate"]["critical_path_seconds"] >= summary["aggregate"][
            "device_training_seconds"
        ]

    fresh_contract = json.loads(
        (fresh_output / "shaft_training_efficiency.json").read_text(encoding="utf-8")
    )["contract"]
    resumed_contract = json.loads(
        (resumed_output / "shaft_training_efficiency.json").read_text(
            encoding="utf-8"
        )
    )["contract"]
    assert fresh_contract == resumed_contract


@pytest.mark.integration
@pytest.mark.manual
def test_qwen35_qwen36_moe_two_rank_train_save_and_exact_resume_release_gate(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    if os.environ.get("SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE") != "1":
        pytest.skip("Set SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE=1 to run the CUDA release gate.")
    if torch.cuda.device_count() < 2:
        pytest.skip("The Qwen training release gate requires two visible CUDA devices.")
    processor_source = Path("models/Qwen3.6-27B")
    if not (processor_source / "preprocessor_config.json").is_file():
        pytest.skip(f"Qwen3.6 processor assets not found: {processor_source}")

    model_dir, dataset_path = prepare_tiny_qwen35_training_assets(
        tmp_path,
        processor_source=processor_source,
        moe=True,
    )
    qwen35_output = tmp_path / "qwen35-moe-fixed-fresh"
    qwen35_config = write_qwen_training_gate_config(
        tmp_path / "qwen35-moe-fixed.yaml",
        model_type="qwen35vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=qwen35_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
    )
    _run_qwen_training_gate(repo_root, qwen35_config)
    qwen35_resumed_output = tmp_path / "qwen35-moe-fixed-resumed"
    qwen35_resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen35-moe-fixed-resumed.yaml",
        model_type="qwen35vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=qwen35_resumed_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=qwen35_output / "checkpoint-1",
    )
    _run_qwen_training_gate(repo_root, qwen35_resumed_config)

    fresh_output = tmp_path / "qwen36-moe-fresh"
    fresh_config = write_qwen_training_gate_config(
        tmp_path / "qwen36-moe-fresh.yaml",
        model_type="qwen36vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=fresh_output,
        layout="varlen",
        packing="greedy",
        steps=2,
        save_steps=1,
    )
    _run_qwen_training_gate(repo_root, fresh_config)

    resumed_output = tmp_path / "qwen36-moe-resumed"
    resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen36-moe-resumed.yaml",
        model_type="qwen36vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=resumed_output,
        layout="varlen",
        packing="greedy",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=fresh_output / "checkpoint-1",
    )
    _run_qwen_training_gate(repo_root, resumed_config)

    _assert_checkpoint_state_equivalent(
        qwen35_output / "checkpoint-2",
        qwen35_resumed_output / "checkpoint-2",
        weight_filename="model.safetensors",
    )
    _assert_checkpoint_state_equivalent(
        fresh_output / "checkpoint-2",
        resumed_output / "checkpoint-2",
        weight_filename="model.safetensors",
    )
    _assert_full_hf_export_reloads(
        qwen35_output / "best",
        expected_type="Qwen3_5MoeForConditionalGeneration",
    )
    _assert_full_hf_export_reloads(
        fresh_output / "best",
        expected_type="Qwen3_5MoeForConditionalGeneration",
    )

    for output_dir, expected_steps in (
        (qwen35_output, 2),
        (qwen35_resumed_output, 2),
        (fresh_output, 2),
        (resumed_output, 2),
    ):
        summary = json.loads(
            (output_dir / "shaft_training_efficiency.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["complete_history"] is True
        assert summary["aggregate"]["optimizer_steps"] == expected_steps
        assert summary["aggregate"]["device_timing_steps"] == expected_steps
        assert summary["contract"]["model_plan_fingerprint"]

    fresh_contract = json.loads(
        (fresh_output / "shaft_training_efficiency.json").read_text(encoding="utf-8")
    )["contract"]
    resumed_contract = json.loads(
        (resumed_output / "shaft_training_efficiency.json").read_text(
            encoding="utf-8"
        )
    )["contract"]
    assert fresh_contract == resumed_contract


@pytest.mark.integration
@pytest.mark.manual
def test_qwen3vl_two_rank_lora_varlen_and_export_release_gate(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    if os.environ.get("SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE") != "1":
        pytest.skip("Set SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE=1 to run the CUDA release gate.")
    if torch.cuda.device_count() < 2:
        pytest.skip("The Qwen training release gate requires two visible CUDA devices.")
    model_path = Path("models/Qwen3-VL-4B-Instruct").resolve()
    if not (model_path / "config.json").is_file():
        pytest.skip(f"Qwen3VL model assets not found: {model_path}")

    dataset_path = prepare_qwen_training_dataset(tmp_path)
    output_dir = tmp_path / "qwen3vl-lora-varlen-fresh"
    config_path = write_qwen_training_gate_config(
        tmp_path / "qwen3vl-lora-varlen.yaml",
        model_type="qwen3vl",
        model_dir=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        layout="varlen",
        packing="greedy",
        steps=2,
        save_steps=1,
        finetune_mode="lora",
    )
    _run_qwen_training_gate(repo_root, config_path)

    resumed_output = tmp_path / "qwen3vl-lora-varlen-resumed"
    resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen3vl-lora-varlen-resumed.yaml",
        model_type="qwen3vl",
        model_dir=model_path,
        dataset_path=dataset_path,
        output_dir=resumed_output,
        layout="varlen",
        packing="greedy",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=output_dir / "checkpoint-1",
        finetune_mode="lora",
    )
    _run_qwen_training_gate(repo_root, resumed_config)

    _assert_checkpoint_state_equivalent(
        output_dir / "checkpoint-2",
        resumed_output / "checkpoint-2",
        weight_filename="adapter_model.safetensors",
    )

    export_path = output_dir / "best"
    _validate_qwen_peft_export(
        repo_root,
        export_path=export_path,
        model_type="qwen3vl",
        model_path=model_path,
    )
    _assert_standard_qwen_peft_export_reloads(
        base_model_path=model_path,
        export_dir=export_path,
    )
    summary = json.loads(
        (output_dir / "shaft_training_efficiency.json").read_text(encoding="utf-8")
    )
    assert summary["schema_version"] == TRAINING_EFFICIENCY_SCHEMA_VERSION
    assert summary["complete_history"] is True
    assert summary["aggregate"]["optimizer_steps"] == 2
    assert summary["aggregate"]["device_timing_steps"] == 2
    assert summary["aggregate"]["logical_segments"] > summary["aggregate"][
        "physical_packs"
    ]
    assert (export_path / "adapter_config.json").is_file()
    assert (export_path / "adapter_model.safetensors").is_file()

    reload_output = tmp_path / "qwen3vl-lora-varlen-reloaded"
    reload_config = write_qwen_training_gate_config(
        tmp_path / "qwen3vl-lora-varlen-reloaded.yaml",
        model_type="qwen3vl",
        model_dir=model_path,
        dataset_path=dataset_path,
        output_dir=reload_output,
        layout="varlen",
        packing="greedy",
        steps=1,
        save_steps=None,
        init_from_checkpoint=export_path,
        finetune_mode="lora",
    )
    _run_qwen_training_gate(repo_root, reload_config)
    assert (reload_output / "best" / "adapter_model.safetensors").is_file()
