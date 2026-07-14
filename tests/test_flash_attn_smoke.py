from __future__ import annotations

import importlib.util

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from shaft.data import ShaftVarlenLayoutPlan, ShaftVarlenSegmentLayout
from shaft.model import (
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
    second_ids = torch.tensor([[6, 7, 120]], dtype=torch.long)
    second_types = torch.tensor([[0, 0, 1]], dtype=torch.long)
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
