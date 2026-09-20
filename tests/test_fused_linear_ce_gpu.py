from contextlib import nullcontext
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from shaft.model.descriptor import ResolvedModelDescriptor
from shaft.config.training import TrainLigerConfig
from shaft.model.qwen35vl import QWEN35VL_META
from shaft.training.linear_ce import linear_causal_cross_entropy
from shaft.training.loss import causal_lm_cross_entropy
from shaft.training.sft_trainer import ShaftSFTTrainer
from tests.support.training import build_training_args
from tests.test_training_numerics import collate, model, reference_loss, rows  # noqa: F401


pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("Fused linear CE runtime tests require CUDA.")
    pytest.importorskip("liger_kernel")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_liger_loss_gradients_and_update(weighted, bf16, tied):
    torch.manual_seed(12)
    head = torch.nn.Linear(64, 257, bias=False, device="cuda")
    embedding = torch.nn.Embedding(257, 64, device="cuda")
    if tied:
        embedding.weight = head.weight
    ref_head, ref_embedding = deepcopy((head, embedding))
    ids = torch.randint(0, 257, (3, 17), device="cuda")
    labels = ids.clone()
    labels[0, :8] = -100
    labels[1, 12:] = -100
    scales = torch.rand(3, 17, device="cuda") if weighted else None
    losses = []
    for fused, current_head, current_embedding in (
        (False, ref_head, ref_embedding),
        (True, head, embedding),
    ):
        parameters = list(
            dict.fromkeys([*current_head.parameters(), *current_embedding.parameters()])
        )
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        with torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext():
            hidden = current_embedding(ids)
            kwargs = dict(labels=labels, loss_scale=scales, normalization_denominator=73)
            loss = (
                linear_causal_cross_entropy(hidden_states=hidden, lm_head=current_head, **kwargs)
                if fused
                else causal_lm_cross_entropy(logits=current_head(hidden), **kwargs)
            )
        loss.backward()
        losses.append(loss.detach())
        optimizer.step()
    atol, rtol = (3e-4, 0.02) if bf16 else (2e-6, 2e-4)
    torch.testing.assert_close(losses[0], losses[1], atol=atol, rtol=rtol)
    for actual, expected in (
        (head.weight, ref_head.weight),
        (embedding.weight, ref_embedding.weight),
    ):
        torch.testing.assert_close(actual.grad, expected.grad, atol=atol, rtol=rtol)
        if bf16:
            # Adam's first update is approximately sign(g): near-zero BF16
            # cancellation can reverse that sign without a meaningful gradient
            # error. Check gradient relative L2 and stable-sign updates separately,
            # rather than widening the parameter tolerance to 2 * learning_rate.
            relative_error = (actual.grad - expected.grad).norm() / expected.grad.norm()
            assert relative_error < 0.01
            stable = expected.grad.abs() > expected.grad.abs().max() * 0.01
            torch.testing.assert_close(actual[stable], expected[stable], atol=2e-5, rtol=2e-4)
        else:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("liger", [False, True])
def test_qwen35_trainer_fused_gradient_accumulation(model, weighted, liger, tmp_path):  # noqa: F811
    model.cuda().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    items = rows(4, weighted)
    batch = {key: value.cuda() for key, value in collate(items).items()}
    expected = reference_loss(model, batch)
    expected.backward()
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    assert any("visual" in name and bool(grad.abs().sum()) for name, grad in gradients.items())
    adapter = QWEN35VL_META.resolve_adapter(
        model_name_or_path="tiny-qwen35",
        descriptor=ResolvedModelDescriptor.from_payload(model.config.to_dict(), source="test"),
    )
    trainer = ShaftSFTTrainer(
        model=model,
        model_adapter=adapter,
        liger_config=TrainLigerConfig(fused_linear_ce=True, rms_norm=liger, swiglu=liger),
        args=build_training_args(tmp_path, use_cpu=False, average_tokens_across_devices=True),
    )
    denominator = batch["labels"][:, 1:].ne(-100).float()
    if weighted:
        denominator *= batch["loss_scale"][:, 1:]
    denominator = denominator.sum()
    for size in (1, 2, 4):
        model.zero_grad(set_to_none=True)
        actual = 0
        for start in range(0, 4, size):
            microbatch = {
                key: value.cuda() for key, value in collate(items[start : start + size]).items()
            }
            loss = trainer.compute_loss(model, microbatch, num_items_in_batch=denominator)
            actual += loss.detach()
            loss.backward()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, gradients[name], atol=2e-6, rtol=2e-4)
    # Explicit outputs / evaluation must still return normal HF logits.
    _, outputs = trainer.compute_loss(model, batch, return_outputs=True)
    assert outputs.logits.shape == (4, 12, 128)


def test_liger_module_coverage_preserves_parameters_and_hf_classes(model):  # noqa: F811
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP
    from shaft.model.qwen35_loss import Qwen35VLTrainingObjectivePolicy

    norm_class, mlp_class = hf.Qwen3_5RMSNorm, hf.Qwen3_5MLP
    original_norm_forward = norm_class.forward
    original_mlp_forward = mlp_class.forward
    parameters = {name: id(p) for name, p in model.named_parameters()}
    state = {name: value.clone() for name, value in model.state_dict().items()}
    policy = Qwen35VLTrainingObjectivePolicy()
    policy.enable_liger_kernels(model, rms_norm=True, swiglu=True)
    policy.enable_liger_kernels(model, rms_norm=True, swiglu=True)
    patched_norms = []
    for name, module in model.named_modules():
        if isinstance(module, norm_class):
            assert module.forward.__func__ is LigerRMSNorm.forward
            assert module.offset == 1.0
            assert module.casting_mode == "gemma"
            assert module.in_place is False
            patched_norms.append(name)
        if isinstance(module, mlp_class):
            assert module.forward.__func__ is LigerQwen3MoeSwiGLUMLP.forward
    assert any(name.endswith("q_norm") for name in patched_norms)
    assert any(name.endswith("k_norm") for name in patched_norms)
    assert {name: id(p) for name, p in model.named_parameters()} == parameters
    assert state.keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    assert hf.Qwen3_5RMSNorm is norm_class
    assert norm_class.forward is original_norm_forward
    assert hf.Qwen3_5MLP is mlp_class
    assert mlp_class.forward is original_mlp_forward


def test_four_rank_fused_ce_global_denominator_and_empty_rank():
    if torch.cuda.device_count() < 4:
        pytest.skip("Four visible idle CUDA devices are required for the DDP fused CE gate.")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "tests/support/fused_ce_ddp_probe.py",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS rank=") == 8


@pytest.mark.parametrize("liger", [False, True])
def test_fused_ce_runs_through_hf_train_loop_without_logits(model, tmp_path, liger):  # noqa: F811
    adapter = QWEN35VL_META.resolve_adapter(
        model_name_or_path="tiny-qwen35",
        descriptor=ResolvedModelDescriptor.from_payload(model.config.to_dict(), source="test"),
    )
    trainer = ShaftSFTTrainer(
        model=model,
        model_adapter=adapter,
        liger_config=TrainLigerConfig(fused_linear_ce=True, rms_norm=liger, swiglu=liger),
        args=build_training_args(
            tmp_path,
            use_cpu=False,
            bf16=True,
            max_steps=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            average_tokens_across_devices=True,
            remove_unused_columns=False,
            save_strategy="no",
            logging_strategy="no",
            disable_tqdm=True,
        ),
        train_dataset=rows(4),
        data_collator=collate,
    )

    def forbidden(*args):
        raise AssertionError("HF train loop materialized full logits")

    hook = model.lm_head.register_forward_pre_hook(forbidden)
    trainer.train()
    hook.remove()
    assert trainer.state.global_step == 2
