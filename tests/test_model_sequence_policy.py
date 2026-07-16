from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from shaft.data import ShaftVarlenLayoutPlan, ShaftVarlenSegmentLayout
from shaft.model import (
    ShaftMediaSegmentManifest,
    ShaftMediaSlice,
    ShaftProcessorMediaManifest,
    build_model_meta,
)
from shaft.model.sequence import (
    Qwen35VLSequenceExecutionPolicy,
    Qwen3VLSequenceExecutionPolicy,
)


def _tiny_qwen3vl_config(*, deepstack: bool = False):
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "attention_dropout": 0.0,
            "use_cache": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [0] if deepstack else [],
            "num_position_embeddings": 16,
        },
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )


def _trusted_qwen_core() -> object:
    def get_rope_index(
        self,
        *,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        video_grid_thw=None,
        attention_mask=None,
    ):
        _ = self, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask
        positions = torch.arange(input_ids.shape[-1], dtype=input_ids.dtype)
        mrope = torch.stack(
            (positions + 10, positions + 20, positions + 30),
            dim=0,
        ).unsqueeze(1)
        return mrope, torch.zeros((1, 1), dtype=input_ids.dtype)

    core_type = type(
        "Qwen3VLModel",
        (),
        {
            "__module__": "transformers.models.qwen3_vl.modeling_qwen3_vl",
            "get_rope_index": get_rope_index,
        },
    )
    core = core_type()
    core.config = SimpleNamespace(
        _attn_implementation="sdpa",
        image_token_id=120,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    return core


def _trusted_qwen35_core() -> object:
    core = _trusted_qwen_core()
    trusted_type = type(
        "Qwen3_5Model",
        (type(core),),
        {"__module__": "transformers.models.qwen3_5.modeling_qwen3_5"},
    )
    trusted = trusted_type()
    trusted.config = core.config
    trusted.forward = lambda **kwargs: kwargs
    trusted.language_model = SimpleNamespace(forward=lambda **kwargs: kwargs)
    return trusted


def _varlen_inputs() -> dict[str, object]:
    layout = ShaftVarlenLayoutPlan(
        global_microstep=4,
        plan_fingerprint="plan-v4",
        local_batch_id=0,
        pack_lengths=(7,),
        segments=(
            ShaftVarlenSegmentLayout(
                processor_row_index=0,
                pack_index=0,
                segment_index=0,
                start=0,
                stop=4,
            ),
            ShaftVarlenSegmentLayout(
                processor_row_index=1,
                pack_index=0,
                segment_index=1,
                start=4,
                stop=7,
            ),
        ),
    )
    manifest = ShaftProcessorMediaManifest(
        segments=(
            ShaftMediaSegmentManifest(
                processor_row_index=0,
                image_grids=ShaftMediaSlice(0, 1),
                image_patches=ShaftMediaSlice(0, 4),
            ),
            ShaftMediaSegmentManifest(
                processor_row_index=1,
                image_grids=ShaftMediaSlice(1, 2),
                image_patches=ShaftMediaSlice(4, 8),
            ),
        ),
        image_grid_count=2,
        image_patch_count=8,
    )
    return {
        "input_ids": torch.tensor([[11, 120, 13, 14, 21, 22, 120]], dtype=torch.long),
        "labels": torch.tensor([[-100, 120, 13, 14, -100, 22, 120]], dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 1, 0, 0, 0, 0, 1]], dtype=torch.long),
        "pixel_values": torch.zeros((8, 8), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long),
        "_shaft_varlen_layout": layout,
        "_shaft_media_manifest": manifest,
    }


def test_qwen3vl_varlen_policy_builds_isolated_four_axis_positions() -> None:
    policy = Qwen3VLSequenceExecutionPolicy()
    model = SimpleNamespace(model=_trusted_qwen_core())

    prepared = policy.prepare_training_inputs(model=model, inputs=_varlen_inputs())

    assert prepared["position_ids"].shape == (4, 1, 7)
    assert prepared["position_ids"][0, 0].tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert prepared["position_ids"][1, 0].tolist() == [10, 11, 12, 13, 10, 11, 12]
    assert "attention_mask" not in prepared
    assert "_shaft_varlen_layout" not in prepared
    assert "_shaft_media_manifest" not in prepared
    assert prepared["use_cache"] is False


def test_qwen35vl_hybrid_varlen_adds_full_and_linear_attention_boundaries() -> None:
    policy = Qwen35VLSequenceExecutionPolicy()
    prepared = policy.prepare_training_inputs(
        model=SimpleNamespace(model=_trusted_qwen35_core()),
        inputs=_varlen_inputs(),
    )

    assert prepared["position_ids"][0, 0].tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert prepared["seq_idx"].tolist() == [[0, 0, 0, 0, 1, 1, 1]]
    assert prepared["cu_seq_lens_q"].tolist() == [0, 4, 7]
    assert prepared["cu_seq_lens_k"].tolist() == [0, 4, 7]
    assert prepared["max_length_q"] == 4
    assert prepared["max_length_k"] == 4


def test_qwen35vl_and_qwen36vl_adapters_share_verified_hybrid_policy() -> None:
    with patch.object(
        Qwen35VLSequenceExecutionPolicy,
        "_package_version",
        return_value="test-version",
    ):
        for model_type in ("qwen35vl", "qwen36vl"):
            adapter = build_model_meta(model_type).resolve_adapter(
                model_name_or_path="models/Qwen3.6-27B"
            )
            assert isinstance(
                adapter.sequence_execution_policy,
                Qwen35VLSequenceExecutionPolicy,
            )
            contract = adapter.build_sequence_execution_contract(
                layout="varlen",
                device_type="cuda",
                attention_implementation="flash_attention_2",
                torch_dtype="bf16",
                distributed_strategy="ddp",
            )
            assert contract.capability_signature[0].startswith(
                "shaft-qwen35vl-hybrid-sequence-execution"
            )
            assert contract.capability_signature[-3:] == (
                "flash-attn=test-version",
                "flash-linear-attention=test-version",
                "causal-conv1d=test-version",
            )


def test_qwen35vl_hybrid_varlen_fails_closed_without_cuda_or_ddp() -> None:
    policy = Qwen35VLSequenceExecutionPolicy()
    with pytest.raises(ValueError, match="requires CUDA kernels"):
        policy.build_contract(
            layout="varlen",
            device_type="cpu",
            attention_implementation="eager",
            torch_dtype="fp32",
            distributed_strategy="ddp",
        )
    with pytest.raises(ValueError, match="supports DDP only"):
        policy.build_contract(
            layout="varlen",
            device_type="cuda",
            attention_implementation="flash_attention_2",
            torch_dtype="bf16",
            distributed_strategy="fsdp",
        )


def test_qwen35vl_hybrid_varlen_fails_closed_when_isolation_kernel_is_missing() -> None:
    policy = Qwen35VLSequenceExecutionPolicy()

    def package_version(package: str) -> str:
        return "missing" if package == "causal-conv1d" else "test-version"

    with patch.object(
        Qwen35VLSequenceExecutionPolicy,
        "_package_version",
        side_effect=package_version,
    ):
        with pytest.raises(ImportError, match="causal-conv1d"):
            policy.build_contract(
                layout="varlen",
                device_type="cuda",
                attention_implementation="flash_attention_2",
                torch_dtype="bf16",
                distributed_strategy="ddp",
            )


def test_qwen35vl_runtime_adapter_filters_boundaries_only_from_media_calls() -> None:
    policy = Qwen35VLSequenceExecutionPolicy()
    core = _trusted_qwen35_core()
    observed = {}

    def get_image_features(pixel_values=None, **kwargs):
        observed.update(kwargs)
        return pixel_values

    core.get_image_features = get_image_features
    model = SimpleNamespace(model=core)
    contract = SimpleNamespace(layout="varlen")

    policy.configure_runtime(model=model, contract=contract)
    result = core.get_image_features(
        pixel_values="pixels",
        seq_idx="private",
        cu_seq_lens_q="private",
        output_attentions=True,
    )

    assert result == "pixels"
    assert observed == {"output_attentions": True}
    assert core._shaft_sequence_kwarg_filter_v2 is True


def test_qwen35vl_runtime_adapter_fails_closed_on_upstream_api_drift() -> None:
    policy = Qwen35VLSequenceExecutionPolicy()
    model = SimpleNamespace(model=_trusted_qwen35_core())

    with pytest.raises(ValueError, match="media feature methods"):
        policy.configure_runtime(
            model=model,
            contract=SimpleNamespace(layout="varlen"),
        )


def test_qwen3vl_varlen_policy_rejects_untrusted_or_wrong_model_core() -> None:
    policy = Qwen3VLSequenceExecutionPolicy()

    with pytest.raises(ValueError, match="trusted Transformers Qwen3VLModel"):
        policy.prepare_training_inputs(
            model=SimpleNamespace(model=SimpleNamespace()),
            inputs=_varlen_inputs(),
        )


def test_qwen3vl_varlen_policy_rejects_missing_media_manifest() -> None:
    policy = Qwen3VLSequenceExecutionPolicy()
    inputs = _varlen_inputs()
    inputs.pop("_shaft_media_manifest")

    with pytest.raises(ValueError, match="image-only media manifest"):
        policy.prepare_training_inputs(
            model=SimpleNamespace(model=_trusted_qwen_core()),
            inputs=inputs,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("use_cache", True, "use_cache=True"),
        ("past_key_values", object(), "past_key_values"),
        ("position_ids", torch.zeros((4, 1, 7), dtype=torch.long), "position_ids"),
        (
            "attention_mask",
            torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.long),
            "attention_mask",
        ),
    ),
)
def test_qwen3vl_varlen_policy_rejects_unsafe_prefill_inputs(
    field: str,
    value: object,
    message: str,
) -> None:
    inputs = _varlen_inputs()
    inputs[field] = value

    with pytest.raises(ValueError, match=message):
        Qwen3VLSequenceExecutionPolicy().prepare_training_inputs(
            model=SimpleNamespace(model=_trusted_qwen_core()),
            inputs=inputs,
        )


def test_qwen3vl_varlen_policy_rejects_image_run_manifest_mismatch() -> None:
    inputs = _varlen_inputs()
    inputs["mm_token_type_ids"] = torch.zeros((1, 7), dtype=torch.long)

    with pytest.raises(ValueError, match="image modality runs"):
        Qwen3VLSequenceExecutionPolicy().prepare_training_inputs(
            model=SimpleNamespace(model=_trusted_qwen_core()),
            inputs=inputs,
        )


def test_qwen3vl_varlen_policy_closes_cuda_backend_holes() -> None:
    policy = Qwen3VLSequenceExecutionPolicy()
    model = SimpleNamespace(model=_trusted_qwen_core())

    with pytest.raises(ValueError, match="flash_attention_2"):
        policy.build_contract(
            layout="varlen",
            device_type="cuda",
            attention_implementation="sdpa",
            torch_dtype="bf16",
            distributed_strategy="ddp",
        )

    cpu_contract = policy.build_contract(
        layout="varlen",
        device_type="cpu",
        attention_implementation="eager",
        torch_dtype="fp32",
        distributed_strategy="ddp",
    )
    policy.validate_runtime(
        model=model,
        contract=cpu_contract,
    )

    wrong_backend_model = SimpleNamespace(model=_trusted_qwen_core())
    wrong_backend_model.model.config = SimpleNamespace(_attn_implementation="sdpa")
    with pytest.raises(ValueError, match="did not retain"):
        policy.validate_runtime(
            model=wrong_backend_model,
            contract=policy.build_contract(
                layout="varlen",
                device_type="cuda",
                attention_implementation="flash_attention_2",
                torch_dtype="bf16",
                distributed_strategy="ddp",
            ),
        )


def test_model_adapter_owns_its_sequence_execution_policy() -> None:
    qwen_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    smoke_adapter = build_model_meta("smoke_vlm").resolve_adapter(
        model_name_or_path="models/Smoke-VLM"
    )

    assert isinstance(qwen_adapter.sequence_execution_policy, Qwen3VLSequenceExecutionPolicy)
    with pytest.raises(ValueError, match="does not support varlen"):
        smoke_adapter.build_sequence_execution_contract(
            layout="varlen",
            device_type="cpu",
            attention_implementation="eager",
            torch_dtype="fp32",
            distributed_strategy="ddp",
        )


def test_model_adapter_builds_stable_model_owned_sequence_contract() -> None:
    adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )

    first = adapter.build_sequence_execution_contract(
        layout="varlen",
        device_type="cuda",
        attention_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        distributed_strategy="ddp",
    )
    second = adapter.build_sequence_execution_contract(
        layout="VARLEN",
        device_type="CUDA",
        attention_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        distributed_strategy="DDP",
    )

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.capability_signature[0] == "shaft-qwen3vl-sequence-execution-v1"
    assert any(item.startswith("transformers=") for item in first.capability_signature)
    assert any(item.startswith("flash-attn=") for item in first.capability_signature)

    with pytest.raises(ValueError, match="does not support varlen"):
        build_model_meta("smoke_vlm").resolve_adapter(
            model_name_or_path="models/Smoke-VLM"
        ).build_sequence_execution_contract(
            layout="varlen",
            device_type="cpu",
            attention_implementation="eager",
            torch_dtype="fp32",
            distributed_strategy="ddp",
        )


def test_real_tiny_qwen3vl_cpu_packed_segments_match_separate_forwards() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    torch.manual_seed(17)
    config = _tiny_qwen3vl_config()
    model = Qwen3VLModel(config).eval()
    first_ids = torch.tensor([[3, 120, 4, 5]], dtype=torch.long)
    first_types = torch.tensor([[0, 1, 0, 0]], dtype=torch.long)
    second_ids = torch.tensor([[6, 7, 120]], dtype=torch.long)
    second_types = torch.tensor([[0, 0, 1]], dtype=torch.long)
    image_grids = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long)
    layout = ShaftVarlenLayoutPlan(
        global_microstep=0,
        plan_fingerprint="real-qwen-oracle",
        local_batch_id=0,
        pack_lengths=(7,),
        segments=(
            ShaftVarlenSegmentLayout(0, 0, 0, 0, 4),
            ShaftVarlenSegmentLayout(1, 0, 1, 4, 7),
        ),
    )
    manifest = ShaftProcessorMediaManifest(
        segments=(
            ShaftMediaSegmentManifest(
                0,
                ShaftMediaSlice(0, 1),
                ShaftMediaSlice(0, 4),
            ),
            ShaftMediaSegmentManifest(
                1,
                ShaftMediaSlice(1, 2),
                ShaftMediaSlice(4, 8),
            ),
        ),
        image_grid_count=2,
        image_patch_count=8,
    )
    packed_inputs = Qwen3VLSequenceExecutionPolicy().prepare_training_inputs(
        model=model,
        inputs={
            "input_ids": torch.cat((first_ids, second_ids), dim=1),
            "mm_token_type_ids": torch.cat((first_types, second_types), dim=1),
            "pixel_values": torch.zeros((8, 8), dtype=torch.float32),
            "image_grid_thw": image_grids,
            "_shaft_varlen_layout": layout,
            "_shaft_media_manifest": manifest,
        },
    )

    def separate_hidden(
        input_ids: torch.Tensor,
        token_types: torch.Tensor,
        image_grid: torch.Tensor,
    ) -> torch.Tensor:
        mrope, _ = model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=token_types,
            image_grid_thw=image_grid,
        )
        scalar = torch.arange(input_ids.shape[-1], dtype=torch.long).view(1, 1, -1)
        return model.language_model(
            input_ids=input_ids,
            position_ids=torch.cat((scalar, mrope), dim=0),
            use_cache=False,
        ).last_hidden_state

    with torch.no_grad():
        packed_hidden = model.language_model(
            input_ids=packed_inputs["input_ids"],
            position_ids=packed_inputs["position_ids"],
            use_cache=False,
        ).last_hidden_state
        first_hidden = separate_hidden(first_ids, first_types, image_grids[0:1])
        second_hidden = separate_hidden(second_ids, second_types, image_grids[1:2])

    torch.testing.assert_close(packed_hidden[:, :4], first_hidden, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(packed_hidden[:, 4:], second_hidden, atol=1e-5, rtol=1e-5)


def test_real_tiny_qwen3vl_full_multimodal_loss_and_gradients_match() -> None:
    from transformers import Qwen3VLForConditionalGeneration

    torch.manual_seed(23)
    model = Qwen3VLForConditionalGeneration(
        _tiny_qwen3vl_config(deepstack=True)
    ).eval()
    policy = Qwen3VLSequenceExecutionPolicy()
    first_ids = torch.tensor([[3, 120, 4, 5]], dtype=torch.long)
    second_ids = torch.tensor([[6, 7, 120]], dtype=torch.long)
    first_types = torch.tensor([[0, 1, 0, 0]], dtype=torch.long)
    second_types = torch.tensor([[0, 0, 1]], dtype=torch.long)
    image_grids = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long)
    pixel_values = torch.randn((8, 12), dtype=torch.float32)

    def prepare(
        *,
        input_ids: torch.Tensor,
        token_types: torch.Tensor,
        pixels: torch.Tensor,
        grids: torch.Tensor,
        fingerprint: str,
    ) -> dict[str, object]:
        segment_lengths = tuple(int(row.shape[-1]) for row in input_ids.split(1, dim=0))
        if len(segment_lengths) != 1:
            raise AssertionError("The standalone oracle expects one processor row.")
        length = segment_lengths[0]
        return policy.prepare_training_inputs(
            model=model,
            inputs={
                "input_ids": input_ids,
                "mm_token_type_ids": token_types,
                "pixel_values": pixels,
                "image_grid_thw": grids,
                "_shaft_varlen_layout": ShaftVarlenLayoutPlan(
                    0,
                    fingerprint,
                    0,
                    (length,),
                    (ShaftVarlenSegmentLayout(0, 0, 0, 0, length),),
                ),
                "_shaft_media_manifest": ShaftProcessorMediaManifest(
                    (
                        ShaftMediaSegmentManifest(
                            0,
                            ShaftMediaSlice(0, 1),
                            ShaftMediaSlice(0, 4),
                        ),
                    ),
                    1,
                    4,
                ),
            },
        )

    packed = policy.prepare_training_inputs(
        model=model,
        inputs={
            "input_ids": torch.cat((first_ids, second_ids), dim=1),
            "mm_token_type_ids": torch.cat((first_types, second_types), dim=1),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grids,
            "_shaft_varlen_layout": ShaftVarlenLayoutPlan(
                0,
                "full-multimodal-packed",
                0,
                (7,),
                (
                    ShaftVarlenSegmentLayout(0, 0, 0, 0, 4),
                    ShaftVarlenSegmentLayout(1, 0, 1, 4, 7),
                ),
            ),
            "_shaft_media_manifest": ShaftProcessorMediaManifest(
                (
                    ShaftMediaSegmentManifest(
                        0,
                        ShaftMediaSlice(0, 1),
                        ShaftMediaSlice(0, 4),
                    ),
                    ShaftMediaSegmentManifest(
                        1,
                        ShaftMediaSlice(1, 2),
                        ShaftMediaSlice(4, 8),
                    ),
                ),
                2,
                8,
            ),
        },
    )
    first = prepare(
        input_ids=first_ids,
        token_types=first_types,
        pixels=pixel_values[:4],
        grids=image_grids[:1],
        fingerprint="full-multimodal-first",
    )
    second = prepare(
        input_ids=second_ids,
        token_types=second_types,
        pixels=pixel_values[4:],
        grids=image_grids[1:],
        fingerprint="full-multimodal-second",
    )

    labels_first = torch.tensor([[-100, 120, 4, 5]], dtype=torch.long)
    labels_second = torch.tensor([[-100, 7, 120]], dtype=torch.long)
    scales_first = torch.tensor([[0.0, 0.5, 1.0, 1.0]])
    scales_second = torch.tensor([[0.0, 2.0, 0.25]])
    packed_labels = torch.cat((labels_first, labels_second), dim=1)
    packed_scales = torch.cat((scales_first, scales_second), dim=1)

    def weighted_terms(
        logits: torch.Tensor,
        labels: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = labels[:, 1:]
        per_token = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(targets)
        valid = targets.ne(-100)
        shifted_scales = scales[:, 1:]
        return (
            (per_token * shifted_scales * valid).sum(),
            (shifted_scales * valid).sum(),
        )

    with torch.no_grad():
        packed_logits = model(**packed).logits
        first_logits = model(**first).logits
        second_logits = model(**second).logits
    torch.testing.assert_close(
        packed_logits[:, :4],
        first_logits,
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        packed_logits[:, 4:],
        second_logits,
        atol=2e-6,
        rtol=2e-6,
    )

    model.zero_grad(set_to_none=True)
    packed_logits = model(**packed).logits
    packed_numerator, packed_denominator = weighted_terms(
        packed_logits,
        packed_labels,
        packed_scales,
    )
    (packed_numerator / packed_denominator).backward()
    packed_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    model.zero_grad(set_to_none=True)
    first_numerator, first_denominator = weighted_terms(
        model(**first).logits,
        labels_first,
        scales_first,
    )
    second_numerator, second_denominator = weighted_terms(
        model(**second).logits,
        labels_second,
        scales_second,
    )
    separate_numerator = first_numerator + second_numerator
    separate_denominator = first_denominator + second_denominator
    (separate_numerator / separate_denominator).backward()
    separate_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    torch.testing.assert_close(packed_numerator, separate_numerator)
    torch.testing.assert_close(packed_denominator, separate_denominator)
    assert packed_gradients.keys() == separate_gradients.keys()
    for name in packed_gradients:
        torch.testing.assert_close(
            packed_gradients[name],
            separate_gradients[name],
            atol=2e-6,
            rtol=2e-5,
            msg=lambda message, name=name: f"{name}: {message}",
        )
    vision_gradient = packed_gradients["model.visual.patch_embed.proj.weight"]
    assert torch.isfinite(vision_gradient).all()
    assert bool(vision_gradient.abs().max() > 0)
