from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from shaft.data import ShaftVarlenLayoutPlan, ShaftVarlenSegmentLayout
from shaft.model import (
    Qwen35VLSequenceExecutionPolicy,
    Qwen3VLSequenceExecutionPolicy,
    ShaftMediaSegmentManifest,
    ShaftMediaSlice,
    ShaftProcessorMediaManifest,
)


pytestmark = [pytest.mark.smoke, pytest.mark.manual]


def _require_flash_attn_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in the current environment.")
    if importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash-attn is not installed in the current environment.")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 0):
        pytest.skip("FlashAttention 2 requires an Ampere-or-newer CUDA device.")


def test_flash_attn_cuda_kernel_smoke() -> None:
    _require_flash_attn_cuda()
    from flash_attn import flash_attn_func

    q = torch.randn(2, 32, 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(2, 32, 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(2, 32, 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)

    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    loss = out.float().mean()
    loss.backward()

    assert tuple(out.shape) == (2, 32, 4, 64)
    assert out.dtype == torch.float16
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()


def test_transformers_flash_attention_2_smoke() -> None:
    _require_flash_attn_cuda()

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    config._attn_implementation = "flash_attention_2"

    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.float16)
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")

    with torch.inference_mode():
        outputs = model(input_ids=input_ids)

    assert getattr(model.config, "_attn_implementation", None) == "flash_attention_2"
    assert tuple(outputs.logits.shape) == (1, 16, config.vocab_size)


def test_qwen3vl_flash_attention_2_isolates_reset_varlen_segments() -> None:
    _require_flash_attn_cuda()
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    torch.manual_seed(17)
    config = Qwen3VLConfig(
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
            "_attn_implementation": "flash_attention_2",
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
            "deepstack_visual_indexes": [],
            "num_position_embeddings": 16,
            "_attn_implementation": "flash_attention_2",
        },
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    config._attn_implementation = "flash_attention_2"
    model = Qwen3VLModel(config).to(device="cuda", dtype=torch.bfloat16).eval()
    first_ids = torch.tensor([[3, 120, 4, 5]], dtype=torch.long)
    first_types = torch.tensor([[0, 1, 0, 0]], dtype=torch.long)
    second_ids = torch.tensor([[6, 120, 7]], dtype=torch.long)
    second_types = torch.tensor([[0, 1, 0]], dtype=torch.long)
    image_grids = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long)
    prepared = Qwen3VLSequenceExecutionPolicy().prepare_training_inputs(
        model=model,
        inputs={
            "input_ids": torch.cat((first_ids, second_ids), dim=1),
            "mm_token_type_ids": torch.cat((first_types, second_types), dim=1),
            "pixel_values": torch.zeros((8, 8), dtype=torch.float32),
            "image_grid_thw": image_grids,
            "_shaft_varlen_layout": ShaftVarlenLayoutPlan(
                global_microstep=0,
                plan_fingerprint="qwen-fa2-isolation",
                local_batch_id=0,
                pack_lengths=(7,),
                segments=(
                    ShaftVarlenSegmentLayout(0, 0, 0, 0, 4),
                    ShaftVarlenSegmentLayout(1, 0, 1, 4, 7),
                ),
            ),
            "_shaft_media_manifest": ShaftProcessorMediaManifest(
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
            ),
        },
    )

    def standalone_positions(
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
        return torch.cat((scalar, mrope), dim=0)

    packed_ids = prepared["input_ids"].to("cuda")
    packed_positions = prepared["position_ids"].to("cuda")
    first_ids = first_ids.to("cuda")
    second_ids = second_ids.to("cuda")
    with torch.inference_mode():
        packed_hidden = model.language_model(
            input_ids=packed_ids,
            position_ids=packed_positions,
            use_cache=False,
        ).last_hidden_state
        first_hidden = model.language_model(
            input_ids=first_ids,
            position_ids=standalone_positions(
                first_ids.cpu(),
                first_types,
                image_grids[0:1],
            ).to("cuda"),
            use_cache=False,
        ).last_hidden_state
        second_hidden = model.language_model(
            input_ids=second_ids,
            position_ids=standalone_positions(
                second_ids.cpu(),
                second_types,
                image_grids[1:2],
            ).to("cuda"),
            use_cache=False,
        ).last_hidden_state

    torch.testing.assert_close(packed_hidden[:, :4], first_hidden, atol=0.02, rtol=0.02)
    torch.testing.assert_close(packed_hidden[:, 4:], second_hidden, atol=0.02, rtol=0.02)


def test_qwen35_hybrid_kernels_isolate_packed_segments() -> None:
    _require_flash_attn_cuda()
    if importlib.util.find_spec("fla") is None or importlib.util.find_spec(
        "causal_conv1d"
    ) is None:
        pytest.skip("Qwen3.5 packed isolation kernels are not installed.")
    from transformers import Qwen3_5Config
    from transformers import Qwen3_5ForConditionalGeneration

    torch.manual_seed(29)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "max_position_embeddings": 128,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "use_cache": False,
            "_attn_implementation": "flash_attention_2",
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
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
            "_attn_implementation": "flash_attention_2",
        },
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    config._attn_implementation = "flash_attention_2"
    model = Qwen3_5ForConditionalGeneration(config).to(
        device="cuda", dtype=torch.bfloat16
    ).eval()
    policy = Qwen35VLSequenceExecutionPolicy()
    contract = policy.build_contract(
        layout="varlen",
        device_type="cuda",
        attention_implementation="flash_attention_2",
        torch_dtype="bf16",
        distributed_strategy="ddp",
    )
    policy.configure_runtime(model=model, contract=contract)
    policy.validate_runtime(model=model, contract=contract)

    def prepare(ids: torch.Tensor, types: torch.Tensor, lengths: tuple[int, ...]):
        stops = []
        cursor = 0
        segments = []
        manifests = []
        for row, length in enumerate(lengths):
            start = cursor
            cursor += length
            stops.append(cursor)
            segments.append(ShaftVarlenSegmentLayout(row, 0, row, start, cursor))
            manifests.append(
                ShaftMediaSegmentManifest(
                    row,
                    ShaftMediaSlice(row, row + 1),
                    ShaftMediaSlice(row * 4, (row + 1) * 4),
                )
            )
        return policy.prepare_training_inputs(
            model=model,
            inputs={
                "input_ids": ids,
                "mm_token_type_ids": types,
                "pixel_values": torch.zeros((len(lengths) * 4, 12)),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(lengths)),
                "_shaft_varlen_layout": ShaftVarlenLayoutPlan(
                    0,
                    f"qwen35-{lengths}",
                    0,
                    (sum(lengths),),
                    tuple(segments),
                ),
                "_shaft_media_manifest": ShaftProcessorMediaManifest(
                    tuple(manifests),
                    len(lengths),
                    len(lengths) * 4,
                ),
            },
        )

    first_ids = torch.tensor([[3, 120, 4, 5]], dtype=torch.long)
    second_ids = torch.tensor([[6, 7, 120]], dtype=torch.long)
    first_types = torch.tensor([[0, 1, 0, 0]], dtype=torch.long)
    second_types = torch.tensor([[0, 0, 1]], dtype=torch.long)
    packed = prepare(
        torch.cat((first_ids, second_ids), dim=1),
        torch.cat((first_types, second_types), dim=1),
        (4, 3),
    )
    first = prepare(first_ids, first_types, (4,))
    second = prepare(second_ids, second_types, (3,))

    def language_inputs(values):
        names = (
            "input_ids",
            "position_ids",
            "seq_idx",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
            "use_cache",
        )
        return {
            name: value.to("cuda") if torch.is_tensor(value) else value
            for name in names
            if (value := values.get(name)) is not None
        }

    with torch.inference_mode():
        packed_hidden = model.model.language_model(
            **language_inputs(packed)
        ).last_hidden_state
        first_hidden = model.model.language_model(
            **language_inputs(first)
        ).last_hidden_state
        second_hidden = model.model.language_model(
            **language_inputs(second)
        ).last_hidden_state

    torch.testing.assert_close(packed_hidden[:, :4], first_hidden, atol=0.03, rtol=0.03)
    torch.testing.assert_close(packed_hidden[:, 4:], second_hidden, atol=0.03, rtol=0.03)

    def cuda_inputs(values):
        return {
            name: value.to("cuda") if torch.is_tensor(value) else value
            for name, value in values.items()
        }

    def loss_numerator(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shifted_labels = labels[..., 1:]
        return F.cross_entropy(
            logits[..., :-1, :].float().reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

    packed_labels = torch.tensor(
        [[-100, -100, -100, 5, -100, -100, 7]],
        device="cuda",
    )
    first_labels = packed_labels[:, :4]
    second_labels = packed_labels[:, 4:]

    model.eval()
    model.zero_grad(set_to_none=True)
    packed_logits = model(**cuda_inputs(packed)).logits
    packed_loss = loss_numerator(packed_logits, packed_labels) / 2.0
    packed_loss.backward()
    selected_names = (
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
    )
    packed_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if name in selected_names
    }

    model.zero_grad(set_to_none=True)
    first_logits = model(**cuda_inputs(first)).logits
    second_logits = model(**cuda_inputs(second)).logits
    standalone_loss = (
        loss_numerator(first_logits, first_labels)
        + loss_numerator(second_logits, second_labels)
    ) / 2.0
    standalone_loss.backward()

    torch.testing.assert_close(
        packed_logits[:, :4], first_logits, atol=0.04, rtol=0.04
    )
    torch.testing.assert_close(
        packed_logits[:, 4:], second_logits, atol=0.04, rtol=0.04
    )
    torch.testing.assert_close(packed_loss, standalone_loss, atol=0.03, rtol=0.03)
    assert set(packed_gradients) == set(selected_names)
    for name, parameter in model.named_parameters():
        if name in packed_gradients:
            assert parameter.grad is not None
            torch.testing.assert_close(
                packed_gradients[name],
                parameter.grad,
                atol=0.05,
                rtol=0.05,
            )


def test_qwen35_moe_hybrid_kernels_and_media_assembly_isolate_segments() -> None:
    _require_flash_attn_cuda()
    if importlib.util.find_spec("fla") is None or importlib.util.find_spec(
        "causal_conv1d"
    ) is None:
        pytest.skip("Qwen3.5 MoE packed isolation kernels are not installed.")
    from transformers import Qwen3_5MoeConfig
    from transformers import Qwen3_5MoeForConditionalGeneration

    torch.manual_seed(41)
    config = Qwen3_5MoeConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "max_position_embeddings": 128,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "use_cache": False,
            "_attn_implementation": "flash_attention_2",
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
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
            "_attn_implementation": "flash_attention_2",
        },
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    config._attn_implementation = "flash_attention_2"
    model = Qwen3_5MoeForConditionalGeneration(config).to(
        device="cuda", dtype=torch.bfloat16
    ).eval()
    policy = Qwen35VLSequenceExecutionPolicy()
    contract = policy.build_contract(
        layout="varlen",
        device_type="cuda",
        attention_implementation="flash_attention_2",
        torch_dtype="bf16",
        distributed_strategy="ddp",
    )
    policy.configure_runtime(model=model, contract=contract)
    policy.validate_runtime(model=model, contract=contract)

    def prepare(ids: torch.Tensor, types: torch.Tensor, lengths: tuple[int, ...]):
        cursor = 0
        segments = []
        manifests = []
        for row, length in enumerate(lengths):
            start = cursor
            cursor += length
            segments.append(ShaftVarlenSegmentLayout(row, 0, row, start, cursor))
            manifests.append(
                ShaftMediaSegmentManifest(
                    row,
                    ShaftMediaSlice(row, row + 1),
                    ShaftMediaSlice(row * 4, (row + 1) * 4),
                )
            )
        return policy.prepare_training_inputs(
            model=model,
            inputs={
                "input_ids": ids,
                "mm_token_type_ids": types,
                "pixel_values": torch.zeros((len(lengths) * 4, 12)),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(lengths)),
                "_shaft_varlen_layout": ShaftVarlenLayoutPlan(
                    0,
                    f"qwen35-moe-{lengths}",
                    0,
                    (sum(lengths),),
                    tuple(segments),
                ),
                "_shaft_media_manifest": ShaftProcessorMediaManifest(
                    tuple(manifests),
                    len(lengths),
                    len(lengths) * 4,
                ),
            },
        )

    first_ids = torch.tensor([[3, 120, 4, 5]], dtype=torch.long)
    second_ids = torch.tensor([[6, 120, 7]], dtype=torch.long)
    first_types = torch.tensor([[0, 1, 0, 0]], dtype=torch.long)
    second_types = torch.tensor([[0, 1, 0]], dtype=torch.long)
    packed = prepare(
        torch.cat((first_ids, second_ids), dim=1),
        torch.cat((first_types, second_types), dim=1),
        (4, 3),
    )
    first = prepare(first_ids, first_types, (4,))
    second = prepare(second_ids, second_types, (3,))

    def cuda_inputs(values):
        return {
            name: value.to("cuda") if torch.is_tensor(value) else value
            for name, value in values.items()
        }

    with torch.inference_mode():
        packed_logits = model(**cuda_inputs(packed)).logits
        first_logits = model(**cuda_inputs(first)).logits
        second_logits = model(**cuda_inputs(second)).logits

    torch.testing.assert_close(
        packed_logits[:, :4], first_logits, atol=0.05, rtol=0.05
    )
    torch.testing.assert_close(
        packed_logits[:, 4:], second_logits, atol=0.05, rtol=0.05
    )

    def loss_numerator(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shifted_labels = labels[..., 1:]
        return F.cross_entropy(
            logits[..., :-1, :].float().reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

    packed_labels = torch.tensor(
        [[-100, -100, -100, 5, -100, -100, 7]],
        device="cuda",
    )
    first_labels = packed_labels[:, :4]
    second_labels = packed_labels[:, 4:]
    selected_names = (
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    )

    model.zero_grad(set_to_none=True)
    packed_logits = model(**cuda_inputs(packed)).logits
    packed_loss = loss_numerator(packed_logits, packed_labels) / 2.0
    packed_loss.backward()
    packed_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if name in selected_names and parameter.grad is not None
    }

    model.zero_grad(set_to_none=True)
    first_logits = model(**cuda_inputs(first)).logits
    second_logits = model(**cuda_inputs(second)).logits
    standalone_loss = (
        loss_numerator(first_logits, first_labels)
        + loss_numerator(second_logits, second_labels)
    ) / 2.0
    standalone_loss.backward()

    torch.testing.assert_close(packed_loss, standalone_loss, atol=0.05, rtol=0.05)
    assert set(packed_gradients) == set(selected_names)
    for name, parameter in model.named_parameters():
        if name in packed_gradients:
            assert parameter.grad is not None
            torch.testing.assert_close(
                packed_gradients[name],
                parameter.grad,
                atol=0.08,
                rtol=0.08,
            )
