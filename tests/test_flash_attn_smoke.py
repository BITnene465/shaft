from __future__ import annotations

import importlib.util

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


def _require_flash_attn_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in the current environment.")
    if importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash-attn is not installed in the current environment.")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 0):
        pytest.skip("FlashAttention 2 requires an Ampere-or-newer CUDA device.")


@pytest.mark.manual
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


@pytest.mark.manual
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
