"""CPU numerical gate, independent of release weights and optional CUDA kernels.

Uses real tiny HF Qwen3.5 vision + hybrid attention layers, but synthetic already-
processed inputs. This does not certify the tokenizer/processor or GPU kernels.
"""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import SequentialSampler
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5

from shaft.training.sft_trainer import ShaftSFTTrainer
from tests.support.training import build_training_args


pytestmark = pytest.mark.contract


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")
    # Explicit CPU reference kernels, even when a developer has FLA installed.
    for name in (
        "FusedRMSNormGated",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    ):
        monkeypatch.setattr(modeling_qwen3_5, name, None)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(465)
            config = Qwen3_5Config(
                text_config={
                    "vocab_size": 128,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 16,
                    "linear_key_head_dim": 16,
                    "linear_value_head_dim": 16,
                    "linear_num_key_heads": 1,
                    "linear_num_value_heads": 2,
                    "linear_conv_kernel_dim": 4,
                    "layer_types": ["linear_attention", "full_attention"],
                    "max_position_embeddings": 128,
                    "use_cache": False,
                    "attention_dropout": 0.0,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 10000.0,
                        "mrope_section": [2, 1, 1],
                        "mrope_interleaved": True,
                    },
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
                    "num_position_embeddings": 16,
                },
                image_token_id=120,
                video_token_id=121,
                vision_start_token_id=122,
                vision_end_token_id=123,
            )
            config._attn_implementation = "eager"
            yield Qwen3_5ForConditionalGeneration(config).float()
    finally:
        torch.set_num_threads(previous_threads)


def rows(count, weighted=False):
    result = []
    for index, length in enumerate((5, 9, 6, 12)[:count]):
        ids = torch.tensor([3, 122, 120, 123] + list(range(10, 10 + length - 4)))
        labels = ids.clone()
        labels[:4] = -100
        row = {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": ids.eq(120).long(),
            "pixel_values": torch.linspace(-1, 1, 48).reshape(4, 12) + index / 10,
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        if weighted:
            row["loss_scale"] = torch.linspace(0.25, 1.75, length)
        result.append(row)
    return result


def collate(items):
    result = {}
    for key in items[0]:
        values = [item[key] for item in items]
        result[key] = (
            torch.cat(values)
            if key in {"pixel_values", "image_grid_thw"}
            else pad_sequence(
                values, batch_first=True, padding_value=-100 if key == "labels" else 0
            )
        )
    return result


def reference_loss(model, batch):
    inputs = {key: value for key, value in batch.items() if key not in {"labels", "loss_scale"}}
    logits = model(**inputs).logits
    labels = batch["labels"][:, 1:]
    weights = labels.ne(-100).float()
    if "loss_scale" in batch:
        weights = weights * batch["loss_scale"][:, 1:]
    losses = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(labels)
    return (losses * weights).sum() / weights.sum()


def test_qwen35_padded_logits_match_single_samples(model):
    model.eval()
    items = rows(4)
    with torch.no_grad():
        batch = collate(items)
        batch.pop("labels")
        batched = model(**batch).logits
        for index, row in enumerate(items):
            single = collate([row])
            single.pop("labels")
            expected = model(**single).logits[0]
            torch.testing.assert_close(
                batched[index, : len(row["input_ids"])],
                expected,
                atol=2e-6,
                rtol=2e-5,
            )


def test_qwen35_loss_only_forward_preserves_hf_export_and_logits(model, tmp_path):
    from shaft.model.qwen35_loss import Qwen35VLTrainingObjectivePolicy

    batch = collate(rows(2))
    labels = batch.pop("labels")
    original_keys = tuple(model.state_dict())
    expected = model(**batch).logits
    Qwen35VLTrainingObjectivePolicy().enable_fused_linear_ce(model)
    torch.testing.assert_close(model(**batch).logits, expected)
    seen = []

    def objective(*, hidden_states, lm_head):
        seen.append(hidden_states.shape)
        logits = F.linear(hidden_states[:, :-1], lm_head.weight)
        return F.cross_entropy(logits.reshape(-1, 128), labels[:, 1:].reshape(-1))

    def forbidden(*args):
        raise AssertionError("Loss-only forward must not call LM head.forward")

    hook = model.lm_head.register_forward_pre_hook(forbidden)
    output = model(**batch, shaft_loss=objective)
    output.loss.backward()
    hook.remove()
    assert output.logits is None and seen == [(2, 9, 32)]
    assert tuple(model.state_dict()) == original_keys
    model.save_pretrained(tmp_path)
    reloaded = Qwen3_5ForConditionalGeneration.from_pretrained(tmp_path)
    assert reloaded.config.architectures == ["Qwen3_5ForConditionalGeneration"]
    model.eval()
    torch.testing.assert_close(reloaded(**batch).logits, model(**batch).logits)


@pytest.mark.parametrize("count", [3, 4], ids=["incomplete-window", "full-window"])
@pytest.mark.parametrize("weighted", [False, True], ids=["token-mean", "weighted-mean"])
def test_qwen35_optimizer_update_matches_reference_across_microbatches(
    model,
    tmp_path,
    count,
    weighted,
):
    items = rows(count, weighted)
    initial = deepcopy(model.state_dict())
    # Independent autograd CE + AdamW oracle: no Shaft loss or Trainer used here.
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss = reference_loss(model, collate(items))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert any("visual" in name and bool(grad.abs().sum()) for name, grad in gradients.items())
    assert any("linear_attn" in name and bool(grad.abs().sum()) for name, grad in gradients.items())
    optimizer.step()
    expected_parameters = deepcopy(model.state_dict())
    expected_optimizer = deepcopy(optimizer.state_dict())

    for batch_size, accumulation in ((1, 4), (2, 2), (4, 1)):
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        actual_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        captured = {}

        def capture_gradients(*_):
            captured.update(
                {
                    name: parameter.grad.clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
            )

        hook = actual_optimizer.register_step_pre_hook(capture_gradients)
        args = build_training_args(
            tmp_path / f"batch-{batch_size}",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            max_steps=1,
            max_grad_norm=0.7,
            average_tokens_across_devices=True,
            remove_unused_columns=False,
            save_strategy="no",
            logging_strategy="no",
            disable_tqdm=True,
            dataloader_pin_memory=False,
            accelerator_config={
                "dispatch_batches": False,
                "split_batches": False,
                "even_batches": False,
            },
        )
        trainer = ShaftSFTTrainer(
            model=model,
            args=args,
            train_dataset=items,
            train_sampler=SequentialSampler(items),
            data_collator=collate,
            optimizers=(
                actual_optimizer,
                torch.optim.lr_scheduler.LambdaLR(actual_optimizer, lambda _: 1),
            ),
        )
        trainer.train()
        hook.remove()
        assert trainer.state.global_step == 1
        assert captured.keys() == gradients.keys()
        for name, expected in gradients.items():
            torch.testing.assert_close(captured[name], expected, atol=2e-6, rtol=2e-4)
        for name, actual in model.state_dict().items():
            torch.testing.assert_close(actual, expected_parameters[name], atol=2e-5, rtol=2e-4)
        actual_states = actual_optimizer.state_dict()["state"]
        assert actual_states.keys() == expected_optimizer["state"].keys()
        for parameter_id, expected_state in expected_optimizer["state"].items():
            for key, expected in expected_state.items():
                torch.testing.assert_close(
                    actual_states[parameter_id][key],
                    expected,
                    atol=2e-7,
                    rtol=2e-4,
                )
