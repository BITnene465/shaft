from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from types import MethodType

from PIL import Image
import pytest
import torch
from transformers import TrainingArguments

from shaft.config import load_config
from shaft.config.opd import normalize_opd_runtime_config
from shaft.model import ShaftProcessorSequenceField, build_model_tokenizer_processor
from shaft.opd import (
    HFLocalOPDRolloutBackend,
    HTTPRemoteOPDTeacherProvider,
    LocalHFOPDTeacherProvider,
    OPDCollator,
    OPDDataset,
    OPDExecutionRegistry,
    OPDRolloutBackend,
    OPDRolloutRequest,
    OPDTeacherProvider,
    ShaftOPDTrainer,
    VLLMOPDRolloutBackend,
    build_opd_execution_runtime,
    load_jsonl_opd_records,
    resolve_opd_execution_plan,
)
from shaft.opd.loss import (
    OPDTeacherDistribution,
    opd_distribution_loss,
    resolve_opd_objective_plan,
)
from shaft.opd.input_abi import (
    ShaftOPDInputABI,
    build_opd_input_abi,
    validate_opd_input_abi_compatibility,
)
from shaft.opd.remote_teacher import (
    CONTENT_TYPE,
    OPDTeacherIdentity,
    PROTOCOL_VERSION,
    decode_teacher_distribution,
    decode_teacher_score_request,
    encode_teacher_distribution,
    encode_teacher_score_request,
    teacher_request_idempotency_key,
)
from shaft.opd.teacher import OPDTeacherScoreRequest
from shaft.opd.teacher_service import (
    OPDTeacherService,
    create_opd_teacher_app,
    read_bounded_request_body,
)
from shaft.opd.telemetry import OPDTelemetryContract, OPDTelemetryMonitor
from tests.support.configs import write_config_yaml
from tests.support.opd import write_opd_config


pytestmark = pytest.mark.component


def _test_opd_input_abi(
    *,
    token_to_id_fingerprint: str = "b" * 64,
    processor_abi_fingerprint: str = "c" * 64,
    logits_vocab_size: int = 7,
) -> ShaftOPDInputABI:
    return ShaftOPDInputABI(
        token_to_id_fingerprint=token_to_id_fingerprint,
        token_count=logits_vocab_size,
        special_token_ids=(("eos_token_id", (2,)),),
        logits_vocab_size=logits_vocab_size,
        processor_abi_fingerprint=processor_abi_fingerprint,
        required_model_input_names=("attention_mask", "input_ids", "use_cache"),
        optional_model_input_names=(),
        forward_accepted_input_names=("attention_mask", "input_ids"),
        forward_required_input_names=(),
        forward_accepts_kwargs=True,
    )


def test_opd_config_is_owned_by_opd_domain_and_fails_closed(tmp_path: Path) -> None:
    config = load_config(write_opd_config(tmp_path))
    assert config.algorithm.name == "opd"
    assert config.opd.rollout.backend == "hf_local"
    assert config.opd.teacher.provider == "hf_local"
    assert config.opd.teacher.model_type == "smoke_vlm"
    assert config.opd.objective.mode == "full_vocab"
    assert config.opd.objective.divergence == "reverse_kl"

    invalid = (tmp_path / "config_opd.yaml").read_text(encoding="utf-8").replace(
        "source_type: jsonl_opd",
        "source_type: jsonl_sft",
    )
    with pytest.raises(ValueError, match="requires one of.*jsonl_opd"):
        load_config(write_config_yaml(tmp_path, invalid, filename="invalid_source.yaml"))

    invalid = (tmp_path / "config_opd.yaml").read_text(encoding="utf-8").replace(
        "divergence: reverse_kl",
        "divergence: ''",
    )
    with pytest.raises(ValueError, match="must be explicitly set"):
        load_config(write_config_yaml(tmp_path, invalid, filename="invalid_loss.yaml"))

    invalid_remote = load_config(write_opd_config(tmp_path))
    invalid_remote.opd.teacher.provider = "http"
    invalid_remote.opd.teacher.remote.endpoint = "https://teacher.invalid"
    invalid_remote.opd.teacher.remote.artifact_fingerprint = "mutable-tag"
    with pytest.raises(ValueError, match="must be a SHA-256 digest"):
        normalize_opd_runtime_config(invalid_remote)

    invalid = (tmp_path / "config_opd.yaml").read_text(encoding="utf-8").replace(
        "teacher:\n    model_type:",
        "teacher:\n    provider: external\n    model_type:",
    )
    with pytest.raises(ValueError, match="provider must be 'hf_local' or 'http'"):
        load_config(write_config_yaml(tmp_path, invalid, filename="invalid_provider.yaml"))

    vllm = (tmp_path / "config_opd.yaml").read_text(encoding="utf-8").replace(
        "rollout:\n    max_new_tokens:",
        "rollout:\n    backend: vllm\n    max_new_tokens:",
    )
    assert (
        load_config(write_config_yaml(tmp_path, vllm, filename="vllm_backend.yaml"))
        .opd.rollout.backend
        == "vllm"
    )

    telemetry = (tmp_path / "config_opd.yaml").read_text(encoding="utf-8").replace(
        "efficiency:\n    enabled: false",
        "efficiency:\n    enabled: true",
    )
    assert load_config(
        write_config_yaml(tmp_path, telemetry, filename="telemetry.yaml")
    ).train.efficiency.enabled


def test_opd_objective_config_selects_one_registered_mode(tmp_path: Path) -> None:
    path = write_opd_config(tmp_path)
    base = path.read_text(encoding="utf-8")

    topk = base.replace(
        "objective:\n    divergence: reverse_kl",
        "objective:\n    mode: topk_tail\n    divergence: reverse_kl\n    top_k: 4",
    )
    config = load_config(write_config_yaml(tmp_path, topk, filename="topk.yaml"))
    plan = resolve_opd_objective_plan(config.opd.objective)
    assert plan.mode == "topk_tail"
    assert plan.top_k == 4

    missing_k = topk.replace("\n    top_k: 4", "")
    with pytest.raises(ValueError, match="requires opd.objective.top_k"):
        load_config(write_config_yaml(tmp_path, missing_k, filename="missing_k.yaml"))

    invalid_full = base.replace(
        "objective:\n    divergence: reverse_kl",
        "objective:\n    mode: full_vocab\n    divergence: reverse_kl\n    top_k: 4",
    )
    with pytest.raises(ValueError, match="full_vocab.*top_k"):
        load_config(write_config_yaml(tmp_path, invalid_full, filename="full_k.yaml"))

    invalid_chunk = base.replace(
        "objective:\n    divergence: reverse_kl",
        "objective:\n    divergence: reverse_kl\n    token_chunk_size: 0",
    )
    with pytest.raises(ValueError, match="token_chunk_size must be > 0"):
        load_config(write_config_yaml(tmp_path, invalid_chunk, filename="chunk.yaml"))


@pytest.mark.parametrize(
    ("distributed_block", "expected_strategy"),
    [
        (
            "  distributed:\n"
            "    strategy: fsdp\n"
            "    fsdp:\n"
            "      activation_checkpointing: false\n",
            "fsdp",
        ),
        (
            "  distributed:\n"
            "    strategy: deepspeed\n"
            "    deepspeed:\n"
            "      config:\n"
            "        zero_optimization:\n"
            "          stage: 2\n",
            "deepspeed",
        ),
    ],
)
def test_opd_config_accepts_backend_native_distributed_strategies(
    tmp_path: Path,
    distributed_block: str,
    expected_strategy: str,
) -> None:
    base = write_opd_config(tmp_path).read_text(encoding="utf-8")
    configured = base.replace("eval:\n", f"{distributed_block}eval:\n")
    config = load_config(
        write_config_yaml(tmp_path, configured, filename=f"{expected_strategy}.yaml")
    )
    assert config.train.distributed.strategy == expected_strategy


def test_opd_execution_registry_resolves_open_runtime_axes_without_pipeline_branches(
    tmp_path: Path,
) -> None:
    config = load_config(write_opd_config(tmp_path))

    class AlternateRollout(OPDRolloutBackend):
        name = "alternate"
        exact_resume_supported = False

        def generate(self, **kwargs):
            raise AssertionError(kwargs)

    class AlternateTeacher(OPDTeacherProvider):
        name = "alternate"
        exact_resume_supported = False

        def validate_student_model(self, student_model):
            _ = student_model

        def prepare(self, device):
            _ = device

        def validate_input_abi(self, input_abi):
            return input_abi.fingerprint

        def score(self, model_inputs):
            raise AssertionError(model_inputs)

    registry = OPDExecutionRegistry()
    registry.register_rollout_backend("alternate", AlternateRollout)
    registry.register_teacher_provider("alternate", AlternateTeacher)
    config.opd.rollout.backend = "alternate"
    config.opd.teacher.provider = "alternate"

    plan = registry.resolve(config.opd)
    assert plan.rollout_backend_type is AlternateRollout
    assert plan.teacher_provider_type is AlternateTeacher
    with pytest.raises(ValueError, match="does not support exact resume"):
        plan.validate_checkpointing(checkpointing_requested=True)

    class MalformedRollout(AlternateRollout):
        name = "malformed"
        exact_resume_supported = "yes"

    with pytest.raises(TypeError, match="exact_resume_supported as a boolean"):
        registry.register_rollout_backend("malformed", MalformedRollout)


def test_opd_execution_plan_rejects_unknown_runtime_components(tmp_path: Path) -> None:
    config = load_config(write_opd_config(tmp_path))
    config.opd.rollout.backend = "missing"
    with pytest.raises(ValueError, match="Unknown OPD rollout backend.*hf_local"):
        resolve_opd_execution_plan(config.opd)

    config.opd.rollout.backend = "hf_local"
    config.opd.teacher.provider = "missing"
    with pytest.raises(ValueError, match="Unknown OPD teacher provider.*hf_local"):
        resolve_opd_execution_plan(config.opd)


def test_opd_source_is_prompt_only_and_preserves_image_order(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(first)
    Image.new("RGB", (2, 2), color=(0, 255, 0)).save(second)
    path = tmp_path / "opd.jsonl"
    row = {
        "sample_id": "ordered",
        "images": [first.name, second.name],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "compare"},
                    {"type": "image"},
                ],
            }
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    records = load_jsonl_opd_records(path, dataset_name="ordered")
    assert records[0].image_paths == (str(first.resolve()), str(second.resolve()))
    sample = OPDDataset(records)[0]
    assert sample["image_paths"] == (str(first.resolve()), str(second.resolve()))
    assert sample["images"][0].getpixel((0, 0)) == (255, 0, 0)
    assert sample["images"][1].getpixel((0, 0)) == (0, 255, 0)

    row["messages"].append(
        {"role": "assistant", "content": [{"type": "text", "text": "target"}]}
    )
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt-only"):
        load_jsonl_opd_records(path, dataset_name="invalid")


def test_opd_collator_emits_rollout_inputs_without_arbitrary_metadata(
    tmp_path: Path,
) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifacts = build_model_tokenizer_processor(config)
    image = Image.new("RGB", (4, 4), color=(1, 2, 3))
    collator = OPDCollator(
        model_adapter=artifacts.model_adapter,
        template=artifacts.template,
        processor=artifacts.processor,
        tokenizer=artifacts.tokenizer,
        max_length=96,
        max_prompt_length=92,
        add_eos_token=False,
    )
    output = collator(
        [
            {
                "sample_id": "x",
                "dataset_name": "opd",
                "image_paths": ("x.png",),
                "image": image,
                "images": (image,),
                "user_prompt": "describe",
                "messages": None,
                "extra": {"must_not_leak": True},
            }
        ]
    )
    assert set(output) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "_shaft_sample_ids",
        "_shaft_rollout_prompt_ids",
        "_shaft_rollout_request_ids",
    }
    assert output["_shaft_sample_ids"] == ["x"]
    assert output["_shaft_rollout_prompt_ids"] == [
        tuple(output["input_ids"][0][output["attention_mask"][0].bool()].tolist())
    ]


def test_opd_collator_retains_ordered_media_only_for_remote_rollout(tmp_path: Path) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifacts = build_model_tokenizer_processor(config)
    first = Image.new("RGB", (4, 4), color=(1, 2, 3))
    second = Image.new("RGB", (4, 4), color=(4, 5, 6))
    collator = OPDCollator(
        model_adapter=artifacts.model_adapter,
        template=artifacts.template,
        processor=artifacts.processor,
        tokenizer=artifacts.tokenizer,
        max_length=96,
        max_prompt_length=92,
        add_eos_token=False,
        retain_rollout_media=True,
    )
    output = collator(
        [
            {
                "sample_id": "ordered",
                "dataset_name": "opd",
                "image_paths": ("first.png", "second.png"),
                "images": (first, second),
                "user_prompt": "compare",
                "messages": None,
            }
        ]
    )
    assert output["_shaft_rollout_images"] == [(first, second)]
    generation_prompt_ids = output["_shaft_rollout_generation_prompt_ids"]
    assert len(generation_prompt_ids) == 1
    assert generation_prompt_ids[0]


def test_opd_collator_assembles_policy_declared_sequence_fields_without_field_names(
    tmp_path: Path,
) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifacts = build_model_tokenizer_processor(config)

    class _SequenceProcessor:
        def __init__(self, base):
            self.base = base
            self.tokenizer = base.tokenizer

        def __getattr__(self, name):
            return getattr(self.base, name)

        def __call__(self, **kwargs):
            output = dict(self.base(**kwargs))
            output["alternate_sequence"] = output["attention_mask"].to(dtype=torch.long) * 7
            return output

    processor_policy = replace(
        artifacts.model_adapter.processor_policy,
        processor_sequence_fields=(
            ShaftProcessorSequenceField(
                name="alternate_sequence",
                padding_value=-1,
                continuation_value=0,
            ),
        ),
    )
    model_adapter = replace(
        artifacts.model_adapter,
        processor_policy=processor_policy,
    )
    collator = OPDCollator(
        model_adapter=model_adapter,
        template=artifacts.template,
        processor=_SequenceProcessor(artifacts.processor),
        tokenizer=artifacts.tokenizer,
        max_length=96,
        max_prompt_length=92,
        add_eos_token=False,
    )
    image = Image.new("RGB", (4, 4), color=(1, 2, 3))
    output = collator(
        [
            {
                "sample_id": "short",
                "dataset_name": "opd",
                "image_paths": ("short.png",),
                "images": (image,),
                "user_prompt": "short",
                "messages": None,
            },
            {
                "sample_id": "long",
                "dataset_name": "opd",
                "image_paths": ("long.png",),
                "images": (image,),
                "user_prompt": "a much longer prompt",
                "messages": None,
            },
        ]
    )

    assert tuple(output["alternate_sequence"].shape) == tuple(output["input_ids"].shape)
    assert torch.all(output["alternate_sequence"][output["attention_mask"].bool()] == 7)
    assert torch.all(output["alternate_sequence"][~output["attention_mask"].bool()] == -1)


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "jsd"])
def test_opd_loss_is_shifted_completion_only_and_teacher_is_detached(
    divergence: str,
) -> None:
    torch.manual_seed(3)
    student = torch.randn(1, 4, 7, requires_grad=True)
    teacher = student.detach().clone().requires_grad_(True)
    completion_mask = torch.tensor([[False, False, True, False]])

    baseline = opd_distribution_loss(
        student_logits=student,
        teacher_logits=teacher,
        completion_mask=completion_mask,
        divergence=divergence,
    )
    assert float(baseline.detach()) == pytest.approx(0.0, abs=1e-6)

    shifted_teacher = teacher.detach().clone()
    shifted_teacher[:, 1, 0] += 4.0
    counted = opd_distribution_loss(
        student_logits=student,
        teacher_logits=shifted_teacher,
        completion_mask=completion_mask,
        divergence=divergence,
    )
    assert float(counted.detach()) > 0.0
    counted.backward()
    assert student.grad is not None
    assert teacher.grad is None

    uncounted_teacher = student.detach().clone()
    uncounted_teacher[:, 2, 0] += 4.0
    uncounted = opd_distribution_loss(
        student_logits=student.detach(),
        teacher_logits=uncounted_teacher,
        completion_mask=completion_mask,
        divergence=divergence,
    )
    assert float(uncounted) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "jsd"])
def test_opd_full_vocab_token_chunks_match_value_and_student_gradient(
    divergence: str,
) -> None:
    torch.manual_seed(31)
    baseline_student = torch.randn(7, 13, dtype=torch.float64, requires_grad=True)
    chunked_student = baseline_student.detach().clone().requires_grad_(True)
    teacher = torch.randn(7, 13, dtype=torch.float64)
    distribution = OPDTeacherDistribution.from_dense_logits(teacher)

    baseline = resolve_opd_objective_plan(
        type(
            "Objective",
            (),
            {
                "mode": "full_vocab",
                "divergence": divergence,
                "temperature": 1.7,
                "top_k": None,
                "token_chunk_size": None,
            },
        )()
    ).build().compute(baseline_student, distribution)
    chunked = resolve_opd_objective_plan(
        type(
            "Objective",
            (),
            {
                "mode": "full_vocab",
                "divergence": divergence,
                "temperature": 1.7,
                "top_k": None,
                "token_chunk_size": 2,
            },
        )()
    ).build().compute(chunked_student, distribution)

    baseline.numerator.backward()
    chunked.numerator.backward()
    assert torch.allclose(baseline.numerator, chunked.numerator, atol=1e-10, rtol=1e-10)
    assert torch.equal(baseline.denominator, chunked.denominator)
    assert torch.allclose(
        baseline_student.grad,
        chunked_student.grad,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "jsd"])
def test_opd_topk_tail_at_full_vocabulary_matches_dense_objective(
    divergence: str,
) -> None:
    torch.manual_seed(41)
    dense_student = torch.randn(5, 11, dtype=torch.float64, requires_grad=True)
    topk_student = dense_student.detach().clone().requires_grad_(True)
    teacher = torch.randn(5, 11, dtype=torch.float64)
    temperature = 0.8
    dense_distribution = OPDTeacherDistribution.from_dense_logits(teacher)
    topk_distribution = OPDTeacherDistribution.from_topk_logits(
        teacher,
        top_k=teacher.shape[-1],
        temperature=temperature,
    )

    dense = resolve_opd_objective_plan(
        type(
            "Objective",
            (),
            {
                "mode": "full_vocab",
                "divergence": divergence,
                "temperature": temperature,
                "top_k": None,
                "token_chunk_size": 2,
            },
        )()
    ).build().compute(dense_student, dense_distribution)
    sparse = resolve_opd_objective_plan(
        type(
            "Objective",
            (),
            {
                "mode": "topk_tail",
                "divergence": divergence,
                "temperature": temperature,
                "top_k": teacher.shape[-1],
                "token_chunk_size": 2,
            },
        )()
    ).build().compute(topk_student, topk_distribution)

    dense.numerator.backward()
    sparse.numerator.backward()
    assert torch.allclose(dense.numerator, sparse.numerator, atol=1e-9, rtol=1e-9)
    assert torch.allclose(dense_student.grad, topk_student.grad, atol=1e-9, rtol=1e-9)


def test_opd_topk_tail_keeps_normalized_tail_bucket() -> None:
    teacher = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    distribution = OPDTeacherDistribution.from_topk_logits(
        teacher,
        top_k=2,
        temperature=1.0,
    )
    assert distribution.topk_log_probs is not None
    assert distribution.tail_log_probs is not None
    total = distribution.topk_log_probs.exp().sum(dim=-1) + distribution.tail_log_probs.exp()
    assert torch.allclose(total, torch.ones_like(total), atol=1e-7, rtol=1e-7)
    assert distribution.topk_token_ids.tolist() == [[0, 1]]


def test_opd_remote_teacher_protocol_round_trips_dense_and_topk_without_pickle(
    tmp_path: Path,
) -> None:
    config = load_config(write_opd_config(tmp_path))
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "use_cache": False,
    }
    request = OPDTeacherScoreRequest(
        model_inputs=inputs,
        causal_position_mask=torch.tensor([[False, True]], dtype=torch.bool),
        request_ids=("draw-1",),
        objective_plan=resolve_opd_objective_plan(config.opd.objective),
    )
    encoded = encode_teacher_score_request(request)
    decoded = decode_teacher_score_request(encoded, max_bytes=len(encoded))
    assert decoded.request_ids == request.request_ids
    assert decoded.objective_plan == request.objective_plan
    assert torch.equal(decoded.model_inputs["input_ids"], inputs["input_ids"])
    assert decoded.model_inputs["use_cache"] is False
    assert torch.equal(decoded.causal_position_mask, request.causal_position_mask)

    non_finite = replace(
        request,
        model_inputs={**inputs, "scale": float("nan")},
    )
    with pytest.raises(ValueError, match="non-finite floats"):
        encode_teacher_score_request(non_finite)

    dense = OPDTeacherDistribution.from_dense_logits(torch.randn(1, 7))
    dense_payload = encode_teacher_distribution(dense)
    dense_round_trip = decode_teacher_distribution(
        dense_payload,
        max_bytes=len(dense_payload),
    )
    assert dense_round_trip.kind == "dense_logits"
    assert torch.equal(dense_round_trip.dense_logits, dense.dense_logits)

    topk = OPDTeacherDistribution.from_topk_logits(
        torch.randn(2, 7),
        top_k=3,
        temperature=0.7,
    )
    topk_payload = encode_teacher_distribution(topk)
    topk_round_trip = decode_teacher_distribution(
        topk_payload,
        max_bytes=len(topk_payload),
    )
    assert topk_round_trip.kind == "topk_tail"
    assert torch.equal(topk_round_trip.topk_token_ids, topk.topk_token_ids)
    assert torch.equal(topk_round_trip.topk_log_probs, topk.topk_log_probs)
    assert torch.equal(topk_round_trip.tail_log_probs, topk.tail_log_probs)


def test_opd_input_abi_allows_equivalent_qwen_aliases_and_different_templates(
    tmp_path: Path,
) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifacts = build_model_tokenizer_processor(config)
    student = replace(
        artifacts,
        model_adapter=replace(
            artifacts.model_adapter,
            model_type="qwen35vl",
            template_type="qwen35vl",
        ),
        template=type("Qwen35Template", (), {})(),
    )
    teacher_processor = copy.deepcopy(artifacts.processor)

    def _different_chat_template(self):
        _ = self
        return {
            "processor_class": "DifferentProductAliasProcessor",
            "chat_template": "different raw chat template",
        }

    teacher_processor.to_dict = MethodType(_different_chat_template, teacher_processor)
    teacher = replace(
        artifacts,
        model_adapter=replace(
            artifacts.model_adapter,
            model_type="qwen38vl",
            template_type="qwen38vl",
        ),
        processor=teacher_processor,
        template=type("Qwen38Template", (), {"chat_template": "different"})(),
    )

    student_abi = build_opd_input_abi(student)
    teacher_abi = build_opd_input_abi(teacher)

    assert student_abi.fingerprint == teacher_abi.fingerprint
    assert (
        validate_opd_input_abi_compatibility(student=student_abi, teacher=teacher_abi)
        == student_abi.fingerprint
    )
    local_provider = LocalHFOPDTeacherProvider(config.opd.teacher, model=artifacts.model)
    assert local_provider.validate_input_abi(student_abi) == student_abi.fingerprint


def test_opd_input_abi_rejects_complete_token_to_id_drift(tmp_path: Path) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifacts = build_model_tokenizer_processor(config)
    teacher_tokenizer = copy.deepcopy(artifacts.tokenizer)
    original_vocab = teacher_tokenizer.get_vocab()

    def _drifted_vocab(self):
        _ = self
        drifted = dict(original_vocab)
        first, second = "<token_4>", "<token_5>"
        drifted[first], drifted[second] = drifted[second], drifted[first]
        return drifted

    teacher_tokenizer.get_vocab = MethodType(_drifted_vocab, teacher_tokenizer)
    student_abi = build_opd_input_abi(artifacts)
    teacher_abi = build_opd_input_abi(replace(artifacts, tokenizer=teacher_tokenizer))

    with pytest.raises(ValueError, match="complete token-to-ID mappings differ"):
        validate_opd_input_abi_compatibility(student=student_abi, teacher=teacher_abi)


def test_opd_input_abi_rejects_special_token_id_drift(tmp_path: Path) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifacts = build_model_tokenizer_processor(config)
    teacher_tokenizer = copy.deepcopy(artifacts.tokenizer)
    teacher_tokenizer.eos_token_id = 5

    student_abi = build_opd_input_abi(artifacts)
    teacher_abi = build_opd_input_abi(replace(artifacts, tokenizer=teacher_tokenizer))

    with pytest.raises(ValueError, match="special token IDs differ"):
        validate_opd_input_abi_compatibility(student=student_abi, teacher=teacher_abi)


def test_opd_input_abi_rejects_logits_vocab_drift(tmp_path: Path) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifacts = build_model_tokenizer_processor(config)
    teacher_model = type(artifacts.model)(
        type(artifacts.model.config)(vocab_size=artifacts.model.config.vocab_size + 1)
    )

    student_abi = build_opd_input_abi(artifacts)
    teacher_abi = build_opd_input_abi(replace(artifacts, model=teacher_model))

    with pytest.raises(ValueError, match="logits vocabulary dimensions differ"):
        validate_opd_input_abi_compatibility(student=student_abi, teacher=teacher_abi)


def test_opd_input_abi_rejects_processor_contract_drift(tmp_path: Path) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifacts = build_model_tokenizer_processor(config)
    teacher_policy = replace(
        artifacts.model_adapter.processor_policy,
        static_model_input_names=("position_ids",),
    )
    teacher = replace(
        artifacts,
        model_adapter=replace(
            artifacts.model_adapter,
            processor_policy=teacher_policy,
        ),
    )

    student_abi = build_opd_input_abi(artifacts)
    teacher_abi = build_opd_input_abi(teacher)

    with pytest.raises(ValueError, match="multimodal processor/input ABI differs"):
        validate_opd_input_abi_compatibility(student=student_abi, teacher=teacher_abi)


def test_opd_input_abi_rejects_processor_config_drift(tmp_path: Path) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifacts = build_model_tokenizer_processor(config)
    teacher_processor = copy.deepcopy(artifacts.processor)

    def _drifted_processor_config(self):
        _ = self
        return {
            "processor_class": "SmokeProcessor",
            "image_processor": {"patch_size": 32},
        }

    teacher_processor.to_dict = MethodType(_drifted_processor_config, teacher_processor)
    student_abi = build_opd_input_abi(artifacts)
    teacher_abi = build_opd_input_abi(replace(artifacts, processor=teacher_processor))

    with pytest.raises(ValueError, match="multimodal processor/input ABI differs"):
        validate_opd_input_abi_compatibility(student=student_abi, teacher=teacher_abi)


def test_opd_input_abi_rejects_teacher_forward_field_drift(tmp_path: Path) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifacts = build_model_tokenizer_processor(config)
    teacher_model = copy.deepcopy(artifacts.model)

    def _forward_without_kwargs(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
    ):
        return type(self).forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

    teacher_model.forward = MethodType(_forward_without_kwargs, teacher_model)
    student_abi = build_opd_input_abi(artifacts)
    teacher_abi = build_opd_input_abi(replace(artifacts, model=teacher_model))

    with pytest.raises(ValueError, match="teacher forward input fields are incompatible"):
        validate_opd_input_abi_compatibility(student=student_abi, teacher=teacher_abi)


def test_opd_http_teacher_binds_artifact_input_abi_and_scores(tmp_path: Path) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifact_fingerprint = "a" * 64
    input_abi = _test_opd_input_abi()
    config.opd.teacher.provider = "http"
    config.opd.teacher.remote.artifact_fingerprint = artifact_fingerprint
    config.opd.teacher.remote.max_response_bytes = 1024 * 1024

    identity = OPDTeacherIdentity(
        protocol_version=PROTOCOL_VERSION,
        artifact_fingerprint=artifact_fingerprint,
        model_type="smoke_vlm",
        input_abi=input_abi,
    )

    class FakeTransport:
        def __init__(self):
            self.keys = []

        def get_identity(self):
            return identity

        def score(self, payload, *, idempotency_key):
            self.keys.append(idempotency_key)
            decoded = decode_teacher_score_request(payload, max_bytes=len(payload))
            count = int(decoded.causal_position_mask.sum().item())
            return encode_teacher_distribution(
                OPDTeacherDistribution.from_dense_logits(torch.randn(count, 7))
            )

    transport = FakeTransport()
    provider = HTTPRemoteOPDTeacherProvider(
        config.opd.teacher,
        model=None,
        transport=transport,
    )
    assert provider.validate_input_abi(input_abi) == input_abi.fingerprint
    request = OPDTeacherScoreRequest(
        model_inputs={
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        },
        causal_position_mask=torch.tensor([[False, True]], dtype=torch.bool),
        request_ids=("draw-1",),
        objective_plan=resolve_opd_objective_plan(config.opd.objective),
    )
    distribution = provider.score(request)
    assert distribution.kind == "dense_logits"
    assert distribution.num_positions == 1
    provider.score(request)
    assert len(transport.keys) == 2
    assert transport.keys[0] == transport.keys[1]

    provider.config.remote.artifact_fingerprint = "d" * 64
    different = HTTPRemoteOPDTeacherProvider(
        provider.config,
        model=None,
        transport=transport,
    )
    with pytest.raises(ValueError, match="artifact identity differs"):
        different.validate_input_abi(input_abi)


@pytest.mark.parametrize(
    ("mode", "plugin_name", "prepare_name"),
    [
        ("fsdp", "fsdp_plugin", "prepare_fsdp"),
        ("deepspeed", "deepspeed_plugin", "prepare_deepspeed"),
    ],
)
def test_opd_local_teacher_uses_backend_native_eval_model_preparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    plugin_name: str,
    prepare_name: str,
) -> None:
    config = load_config(write_opd_config(tmp_path))
    model = torch.nn.Linear(2, 2)
    provider = LocalHFOPDTeacherProvider(config.opd.teacher, model=model)
    calls = []

    def prepare(candidate, accelerator):
        calls.append((candidate, accelerator))
        return candidate

    monkeypatch.setattr(f"trl.models.utils.{prepare_name}", prepare)
    state = type(
        "State",
        (),
        {
            "fsdp_plugin": object() if plugin_name == "fsdp_plugin" else None,
            "deepspeed_plugin": object() if plugin_name == "deepspeed_plugin" else None,
        },
    )()
    accelerator = type(
        "Accelerator",
        (),
        {"state": state, "device": torch.device("cpu")},
    )()
    provider.prepare(accelerator)
    assert calls == [(model, accelerator)], mode


def test_opd_teacher_service_validates_idempotency_and_scores_protocol_request(
    tmp_path: Path,
) -> None:
    config = load_config(write_opd_config(tmp_path))
    identity = OPDTeacherIdentity(
        protocol_version=PROTOCOL_VERSION,
        artifact_fingerprint="a" * 64,
        model_type="smoke_vlm",
        input_abi=_test_opd_input_abi(),
    )

    class FakeProvider:
        model = None

        def score(self, request):
            count = int(request.causal_position_mask.sum().item())
            return request.objective_plan.build_teacher_distribution(torch.randn(count, 7))

    request = OPDTeacherScoreRequest(
        model_inputs={
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        },
        causal_position_mask=torch.tensor([[False, True]], dtype=torch.bool),
        request_ids=("draw-1",),
        objective_plan=resolve_opd_objective_plan(config.opd.objective),
    )
    payload = encode_teacher_score_request(request)
    service = OPDTeacherService(
        identity=identity,
        provider=FakeProvider(),
        max_request_bytes=len(payload),
    )
    with pytest.raises(ValueError, match="idempotency key"):
        service.score(payload, idempotency_key="wrong")
    response = service.score(
        payload,
        idempotency_key=teacher_request_idempotency_key(payload),
    )
    distribution = decode_teacher_distribution(response, max_bytes=len(response))
    assert distribution.kind == "dense_logits"
    assert distribution.num_positions == 1


def test_opd_external_teacher_round_trips_over_live_http_loopback(
    tmp_path: Path,
) -> None:
    config = load_config(write_opd_config(tmp_path))
    identity = OPDTeacherIdentity(
        protocol_version=PROTOCOL_VERSION,
        artifact_fingerprint="a" * 64,
        model_type="smoke_vlm",
        input_abi=_test_opd_input_abi(),
    )

    class FakeProvider:
        model = None

        def score(self, request):
            count = int(request.causal_position_mask.sum().item())
            logits = torch.arange(count * 7, dtype=torch.float32).reshape(count, 7)
            return request.objective_plan.build_teacher_distribution(logits)

    service = OPDTeacherService(
        identity=identity,
        provider=FakeProvider(),
        max_request_bytes=1024 * 1024,
    )
    requests: list[tuple[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002, ANN001
            _ = format, args

        def do_GET(self):  # noqa: N802
            requests.append(("GET", self.path))
            payload = json.dumps(identity.to_dict()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):  # noqa: N802
            requests.append(("POST", self.path))
            length = int(self.headers["Content-Length"])
            payload = self.rfile.read(length)
            response = service.score(
                payload,
                idempotency_key=str(self.headers.get("Idempotency-Key", "")),
            )
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config.opd.teacher.provider = "http"
        config.opd.teacher.remote.endpoint = (
            f"http://127.0.0.1:{server.server_address[1]}"
        )
        config.opd.teacher.remote.artifact_fingerprint = identity.artifact_fingerprint
        provider = HTTPRemoteOPDTeacherProvider(config.opd.teacher, model=None)
        assert provider.validate_input_abi(identity.input_abi) == identity.input_abi.fingerprint
        request = OPDTeacherScoreRequest(
            model_inputs={
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            },
            causal_position_mask=torch.tensor([[False, True]], dtype=torch.bool),
            request_ids=("draw-live-http",),
            objective_plan=resolve_opd_objective_plan(config.opd.objective),
        )
        distribution = provider.score(request)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert distribution.kind == "dense_logits"
    assert distribution.num_positions == 1
    assert requests == [("GET", "/v1/identity"), ("POST", "/v1/score")]


def test_opd_teacher_service_rejects_chunked_oversize_body_before_full_read() -> None:
    class Request:
        consumed = 0

        async def stream(self):
            for chunk in (b"1234", b"5678", b"must-not-be-read"):
                self.consumed += 1
                yield chunk

    request = Request()
    with pytest.raises(ValueError, match="too large"):
        asyncio.run(read_bounded_request_body(request, max_bytes=6))
    assert request.consumed == 2


def test_opd_fastapi_teacher_route_injects_request_body() -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    config = type(
        "ObjectiveConfig",
        (),
        {
            "mode": "full_vocab",
            "divergence": "reverse_kl",
            "temperature": 1.0,
            "top_k": None,
            "token_chunk_size": None,
        },
    )()
    identity = OPDTeacherIdentity(
        protocol_version=PROTOCOL_VERSION,
        artifact_fingerprint="a" * 64,
        model_type="smoke_vlm",
        input_abi=_test_opd_input_abi(),
    )

    class Provider:
        model = None

        def score(self, request):
            count = int(request.causal_position_mask.sum().item())
            return request.objective_plan.build_teacher_distribution(torch.randn(count, 7))

    request = OPDTeacherScoreRequest(
        model_inputs={"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)},
        causal_position_mask=torch.tensor([[False, True]], dtype=torch.bool),
        request_ids=("fastapi-body",),
        objective_plan=resolve_opd_objective_plan(config),
    )
    payload = encode_teacher_score_request(request)
    service = OPDTeacherService(
        identity=identity,
        provider=Provider(),
        max_request_bytes=len(payload),
    )
    client = fastapi_testclient.TestClient(create_opd_teacher_app(service))
    response = client.post(
        "/v1/score",
        content=payload,
        headers={
            "Content-Type": CONTENT_TYPE,
            "Idempotency-Key": teacher_request_idempotency_key(payload),
        },
    )
    assert response.status_code == 200
    distribution = decode_teacher_distribution(
        response.content,
        max_bytes=len(response.content),
    )
    assert distribution.kind == "dense_logits"
    assert distribution.num_positions == 1


def test_opd_telemetry_commits_phases_and_exactly_resumes(tmp_path: Path) -> None:
    contract = OPDTelemetryContract(
        training_resume_fingerprint="training",
        rollout_backend="hf_local",
        teacher_provider="hf_local",
        teacher_artifact_fingerprint="teacher",
        objective_mode="full_vocab",
    )
    monitor = OPDTelemetryMonitor(
        output_dir=tmp_path,
        contract=contract,
        persist=True,
    )
    monitor.stage_microbatch(
        {"prompt_tokens": 4, "materialized_prompt_tokens": 6, "vision_patches": 8}
    )
    with monitor.phase("rollout_generate"):
        pass
    with monitor.phase("student_score"):
        pass
    monitor.record_completion_tokens(3)
    monitor.record_teacher_distribution(
        OPDTeacherDistribution.from_dense_logits(torch.randn(3, 7))
    )
    monitor.finish_training_step(0.1)
    monitor.start_optimizer_step()
    monitor.finish_optimizer_step()
    monitor.commit(global_step=1)

    checkpoint = tmp_path / "checkpoint-1"
    monitor.write_checkpoint_snapshot(checkpoint, global_step=1)
    resumed = OPDTelemetryMonitor.from_checkpoint(
        output_dir=tmp_path,
        checkpoint_dir=checkpoint,
        checkpoint_global_step=1,
        contract=contract,
        persist=True,
    )
    assert len(resumed.frames) == 1
    frame = resumed.frames[0]
    assert frame.prompt_tokens == 4
    assert frame.materialized_prompt_tokens == 6
    assert frame.completion_tokens == 3
    assert frame.vision_patches == 8
    assert frame.dense_teacher_elements == 21
    assert frame.update_applied

    incompatible = replace(contract, objective_mode="topk_tail")
    with pytest.raises(ValueError, match="exact resume validation failed"):
        OPDTelemetryMonitor.from_checkpoint(
            output_dir=tmp_path,
            checkpoint_dir=checkpoint,
            checkpoint_global_step=1,
            contract=incompatible,
            persist=True,
        )


def test_opd_telemetry_records_deferred_cuda_phase_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[str] = []

    class Event:
        next_index = 0

        def __init__(self, *, enable_timing: bool):
            assert enable_timing
            self.index = self.__class__.next_index
            self.__class__.next_index += 1

        def record(self) -> None:
            operations.append(f"record-{self.index}")

        def synchronize(self) -> None:
            operations.append(f"synchronize-{self.index}")

        def elapsed_time(self, other: "Event") -> float:
            operations.append(f"elapsed-{self.index}-{other.index}")
            return float(other.index - self.index)

    monkeypatch.setattr("shaft.opd.telemetry.torch.cuda.Event", Event)
    contract = OPDTelemetryContract(
        training_resume_fingerprint="training",
        rollout_backend="hf_local",
        teacher_provider="hf_local",
        teacher_artifact_fingerprint="teacher",
        objective_mode="full_vocab",
        timing_mode="wall+cuda_events",
    )
    monitor = OPDTelemetryMonitor(
        output_dir=tmp_path,
        contract=contract,
        persist=False,
        device_timing=True,
    )
    monitor.stage_microbatch(
        {"prompt_tokens": 2, "materialized_prompt_tokens": 2, "vision_patches": 0}
    )
    with monitor.phase("student_score"):
        pass
    monitor.record_completion_tokens(1)
    monitor.finish_training_step(0.1)
    monitor.start_optimizer_step()
    monitor.finish_optimizer_step()
    monitor.commit(global_step=1)
    metrics = monitor.finalize(final_global_step=1)

    frame = monitor.frames[0]
    assert frame.device_student_score_seconds == pytest.approx(0.001)
    assert frame.device_optimizer_frame_seconds == pytest.approx(0.003)
    assert metrics["opd_efficiency/local_device_seconds"] == pytest.approx(0.003)
    assert operations[-1] == "elapsed-0-3"


def test_opd_telemetry_uses_slowest_rank_as_optimizer_step_critical_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = OPDTelemetryContract(
        training_resume_fingerprint="training",
        rollout_backend="hf_local",
        teacher_provider="hf_local",
        teacher_artifact_fingerprint="teacher",
        objective_mode="full_vocab",
    )
    monitor = OPDTelemetryMonitor(
        output_dir=tmp_path,
        contract=contract,
        persist=False,
    )
    monitor.stage_microbatch(
        {"prompt_tokens": 2, "materialized_prompt_tokens": 2, "vision_patches": 0}
    )
    monitor.record_completion_tokens(3)
    monitor.finish_training_step(0.1)
    monitor.start_optimizer_step()
    monitor.finish_optimizer_step()
    monitor.commit(global_step=1)

    local_frame = monitor.frames[0]
    remote_frame = replace(
        local_frame,
        completion_tokens=5,
        rollout_generate_seconds=0.2,
        backward_and_trainer_overhead_seconds=0.3,
    )
    monkeypatch.setattr(
        "shaft.opd.telemetry._all_gather_object",
        lambda _value: [[asdict(local_frame)], [asdict(remote_frame)]],
    )

    metrics = monitor.finalize(final_global_step=1)

    assert metrics["opd_efficiency/completion_tokens_per_second"] == pytest.approx(
        8 / remote_frame.critical_path_seconds
    )
    assert metrics["opd_efficiency/rollout_seconds"] == pytest.approx(0.2)


def test_opd_collator_truncates_prompt_to_preserve_completion_budget(
    tmp_path: Path,
) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifacts = build_model_tokenizer_processor(config)
    image = Image.new("RGB", (4, 4))
    collator = OPDCollator(
        model_adapter=artifacts.model_adapter,
        template=artifacts.template,
        processor=artifacts.processor,
        tokenizer=artifacts.tokenizer,
        max_length=68,
        max_prompt_length=64,
        add_eos_token=False,
    )
    output = collator(
        [
            {
                "sample_id": "oversize",
                "dataset_name": "opd",
                "image_paths": ("x.png",),
                "image": image,
                "images": (image,),
                "messages": None,
                "user_prompt": "long prompt",
            }
        ]
    )

    assert int(output["attention_mask"].sum().item()) == 64
    assert output["_shaft_sample_ids"] == ["oversize"]


def test_opd_completion_mask_keeps_first_eos_when_pad_and_eos_are_shared() -> None:
    mask = HFLocalOPDRolloutBackend.completion_mask(
        torch.tensor([[9, 2, 2, 2]]),
        eos_token_id=2,
        pad_token_id=2,
    )
    assert mask.tolist() == [[True, True, False, False]]


def test_opd_teacher_request_accepts_model_declared_logit_tail(tmp_path: Path) -> None:
    objective_plan = resolve_opd_objective_plan(
        load_config(write_opd_config(tmp_path)).opd.objective
    )
    request = OPDTeacherScoreRequest(
        model_inputs={"input_ids": torch.arange(84).reshape(1, 84)},
        causal_position_mask=torch.tensor([[True, True]], dtype=torch.bool),
        request_ids=("sample",),
        objective_plan=objective_plan,
    )
    assert tuple(request.causal_position_mask.shape) == (1, 2)


def test_opd_teacher_request_rejects_logit_span_beyond_input_sequence(
    tmp_path: Path,
) -> None:
    objective_plan = resolve_opd_objective_plan(
        load_config(write_opd_config(tmp_path)).opd.objective
    )
    with pytest.raises(ValueError, match="logit tail within input_ids"):
        OPDTeacherScoreRequest(
            model_inputs={"input_ids": torch.arange(3).reshape(1, 3)},
            causal_position_mask=torch.ones((1, 3), dtype=torch.bool),
            request_ids=("sample",),
            objective_plan=objective_plan,
        )


def test_opd_local_rollout_returns_training_safe_non_inference_tensors(tmp_path: Path) -> None:
    config = load_config(write_opd_config(tmp_path))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.config = type("Config", (), {"use_cache": False})()

        def generate(self, input_ids, **kwargs):
            _ = kwargs
            completion = torch.tensor([[7, 2]], device=input_ids.device)
            return torch.cat((input_ids, completion), dim=-1)

    model = Model()
    processor = type(
        "Processor",
        (),
        {"tokenizer": type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 2})()},
    )()
    accelerator = type("Accelerator", (), {"unwrap_model": lambda self, value: value})()
    request = OPDRolloutRequest(
        model=model,
        model_inputs={
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
        generation_prompt_token_ids=((3, 4),),
        prompt_token_ids=((3, 4),),
        ordered_images=(None,),
        sample_ids=("sample",),
        request_ids=("draw",),
        model_version=0,
        accelerator=accelerator,
        processing_class=processor,
    )
    result = HFLocalOPDRolloutBackend(config.opd.rollout).generate(request)
    assert not torch.is_inference(result.sequences)
    assert not torch.is_inference(result.attention_mask)
    assert not torch.is_inference(result.completion_mask)


def test_opd_vllm_rollout_requires_pre_wrap_prepare(tmp_path: Path) -> None:
    config = load_config(write_opd_config(tmp_path))
    processor = type(
        "Processor",
        (),
        {"tokenizer": type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 2})()},
    )()
    request = OPDRolloutRequest(
        model=torch.nn.Linear(2, 2),
        model_inputs={
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
        generation_prompt_token_ids=((3, 4),),
        prompt_token_ids=((3, 4),),
        ordered_images=(None,),
        sample_ids=("sample",),
        request_ids=("draw-1",),
        model_version=0,
        accelerator=type("Accelerator", (), {"device": torch.device("cpu")})(),
        processing_class=processor,
    )

    with pytest.raises(RuntimeError, match="not prepared before distributed model wrapping"):
        VLLMOPDRolloutBackend(config.opd.rollout).generate(request)


def test_opd_vllm_rollout_syncs_each_student_version_and_binds_request_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(write_opd_config(tmp_path))
    config.opd.rollout.backend = "vllm"

    class FakeGeneration:
        instances = []

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.generation_kwargs = {}
            self.sync_count = 0
            self.calls = []
            self.__class__.instances.append(self)

        def sync_weights(self):
            self.sync_count += 1

        def generate(self, *, prompts, images, num_generations):
            self.calls.append(
                {
                    "prompts": prompts,
                    "images": images,
                    "num_generations": num_generations,
                    "seed": self.generation_kwargs["seed"],
                }
            )
            return [[3, 4] for _ in prompts], [[7, 2] for _ in prompts], None, None

    monkeypatch.setattr("shaft.opd.rollout._load_vllm_generation_type", lambda: FakeGeneration)
    backend = VLLMOPDRolloutBackend(config.opd.rollout, seed=13)
    model = torch.nn.Linear(2, 2)
    processor = type(
        "Processor",
        (),
        {"tokenizer": type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 2})()},
    )()
    accelerator = type(
        "Accelerator",
        (),
        {
            "device": torch.device("cpu"),
            "unwrap_model": lambda self, value: getattr(value, "module", value),
        },
    )()
    backend.prepare(
        model=model,
        accelerator=accelerator,
        processing_class=processor,
    )

    class RuntimeDistributedWrapper(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.module = module

    runtime_model = RuntimeDistributedWrapper(model)

    def request(*, version: int, request_id: str) -> OPDRolloutRequest:
        return OPDRolloutRequest(
            model=runtime_model,
            model_inputs={
                "input_ids": torch.tensor([[0, 3, 4]], dtype=torch.long),
                "attention_mask": torch.tensor([[0, 1, 1]], dtype=torch.long),
            },
            generation_prompt_token_ids=((99,),),
            prompt_token_ids=((3, 4),),
            ordered_images=(("image",),),
            sample_ids=("sample",),
            request_ids=(request_id,),
            model_version=version,
            accelerator=accelerator,
            processing_class=processor,
        )

    first = backend.generate(request(version=4, request_id="draw-1"))
    backend.generate(request(version=4, request_id="draw-2"))
    backend.generate(request(version=5, request_id="draw-1"))
    with pytest.raises(RuntimeError, match="does not reference the prepared student"):
        backend.generate(
            replace(
                request(version=6, request_id="draw-3"),
                model=torch.nn.Linear(2, 2),
            )
        )

    generation = FakeGeneration.instances[0]
    assert generation.init_kwargs["model"] is model
    assert generation.sync_count == 2
    assert [call["seed"] for call in generation.calls] == [
        backend._request_seed(request(version=4, request_id="draw-1")),
        backend._request_seed(request(version=4, request_id="draw-2")),
        backend._request_seed(request(version=5, request_id="draw-1")),
    ]
    assert generation.calls[0]["prompts"] == [[99]]
    assert generation.calls[0]["images"] == [["image"]]
    assert first.sequences.tolist() == [[0, 3, 4, 7, 2]]
    assert first.completion_mask.tolist() == [[False, False, False, True, True]]


def test_opd_vllm_rollout_rejects_remote_prompt_token_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(write_opd_config(tmp_path))

    class DriftedGeneration:
        def __init__(self, **kwargs):
            _ = kwargs
            self.generation_kwargs = {}

        def sync_weights(self):
            return None

        def generate(self, *, prompts, images, num_generations):
            _ = images, num_generations
            return [[*prompts[0], 9]], [[2]], None, None

    monkeypatch.setattr(
        "shaft.opd.rollout._load_vllm_generation_type",
        lambda: DriftedGeneration,
    )
    backend = VLLMOPDRolloutBackend(config.opd.rollout, seed=1)
    processor = type(
        "Processor",
        (),
        {"tokenizer": type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 2})()},
    )()
    model = torch.nn.Linear(2, 2)
    accelerator = type(
        "Accelerator",
        (),
        {
            "device": torch.device("cpu"),
            "unwrap_model": lambda self, value: getattr(value, "module", value),
        },
    )()
    backend.prepare(
        model=model,
        accelerator=accelerator,
        processing_class=processor,
    )
    request = OPDRolloutRequest(
        model=model,
        model_inputs={
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
        generation_prompt_token_ids=((3, 4),),
        prompt_token_ids=((3, 4),),
        ordered_images=(("image",),),
        sample_ids=("sample",),
        request_ids=("draw-1",),
        model_version=0,
        accelerator=accelerator,
        processing_class=processor,
    )
    with pytest.raises(ValueError, match="differ from the local processor contract"):
        backend.generate(request)


def test_opd_gradient_accumulation_uses_one_optimizer_window_token_denominator(
    tmp_path: Path,
) -> None:
    config = load_config(write_opd_config(tmp_path))
    artifacts = build_model_tokenizer_processor(config)
    student = copy.deepcopy(artifacts.model)
    teacher = copy.deepcopy(artifacts.model)
    with torch.no_grad():
        next(teacher.parameters()).add_(0.25)

    def _generate(self, input_ids, **kwargs):
        _ = kwargs
        marker = int(input_ids[0, -1].item())
        completion = (
            torch.tensor([[2]], dtype=input_ids.dtype, device=input_ids.device)
            if marker == 10
            else torch.tensor([[5, 6, 2]], dtype=input_ids.dtype, device=input_ids.device)
        )
        return torch.cat((input_ids, completion), dim=-1)

    student.generate = MethodType(_generate, student)
    expected_student = copy.deepcopy(student)
    expected_teacher = copy.deepcopy(teacher)
    execution_runtime = build_opd_execution_runtime(
        resolve_opd_execution_plan(config.opd),
        rollout_config=config.opd.rollout,
        teacher_config=config.opd.teacher,
        teacher_model=teacher,
    )
    assert isinstance(execution_runtime.rollout_backend, HFLocalOPDRolloutBackend)
    assert isinstance(execution_runtime.teacher_provider, LocalHFOPDTeacherProvider)
    trainer = ShaftOPDTrainer(
        model=student,
        execution_runtime=execution_runtime,
        objective_plan=resolve_opd_objective_plan(config.opd.objective),
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer"),
            use_cpu=True,
            report_to="none",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            remove_unused_columns=False,
        ),
        processing_class=artifacts.processor,
        model_adapter=artifacts.model_adapter,
    )
    trainer.current_gradient_accumulation_steps = 2

    first = {
        "input_ids": torch.tensor([[1, 10]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "_shaft_sample_ids": ["short"],
    }
    second = {
        "input_ids": torch.tensor([[1, 11]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "_shaft_sample_ids": ["long"],
    }

    expected_components = []
    for sequences, completion_mask in (
        (
            torch.tensor([[1, 10, 2]], dtype=torch.long),
            torch.tensor([[False, False, True]]),
        ),
        (
            torch.tensor([[1, 11, 5, 6, 2]], dtype=torch.long),
            torch.tensor([[False, False, True, True, True]]),
        ),
    ):
        student_outputs = expected_student(input_ids=sequences, use_cache=False)
        with torch.no_grad():
            teacher_outputs = expected_teacher(input_ids=sequences, use_cache=False)
        _, components = opd_distribution_loss(
            student_logits=student_outputs.logits,
            teacher_logits=teacher_outputs.logits,
            completion_mask=completion_mask,
            divergence=config.opd.objective.divergence,
            temperature=config.opd.objective.temperature,
            return_components=True,
        )
        expected_components.append(components)
    expected_loss = sum(component.numerator for component in expected_components) / sum(
        component.denominator for component in expected_components
    )
    expected_loss.backward()

    trainer.accelerator.gradient_state._set_sync_gradients(False)
    first_report = trainer.training_step(student, first)
    trainer.accelerator.gradient_state._set_sync_gradients(True)
    final_report = trainer.training_step(student, second)

    assert float(first_report) == 0.0
    assert float(final_report) == pytest.approx(float(expected_loss.detach()), rel=1e-6)
    expected_parameters = dict(expected_student.named_parameters())
    for name, parameter in student.named_parameters():
        expected_gradient = expected_parameters[name].grad
        assert parameter.grad is not None, name
        assert expected_gradient is not None, name
        assert torch.allclose(parameter.grad, expected_gradient, atol=1e-7, rtol=1e-6), name
