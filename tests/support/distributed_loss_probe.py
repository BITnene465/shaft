from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers.modeling_outputs import CausalLMOutput

from shaft.training.loss import causal_lm_cross_entropy
from shaft.training.sft_trainer import ShaftSFTTrainer
from tests.support.training import build_training_args


class _ProbeModel(torch.nn.Module):
    def __init__(self, *, vocab_size: int = 7) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids, labels=None, **kwargs):
        _ = labels, kwargs
        return CausalLMOutput(logits=self.embedding(input_ids))


def _batch(
    input_ids: list[int],
    labels: list[int],
    loss_scale: list[float],
) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "loss_scale": torch.tensor([loss_scale], dtype=torch.float32),
    }


def _rank_batches() -> tuple[tuple[dict[str, torch.Tensor], ...], ...]:
    return (
        (
            _batch([0, 1, 2, 3], [0, 1, 2, 3], [0.0, 0.5, 1.0, 0.0]),
            _batch([3, 2, 1, 0], [3, 2, 1, 0], [0.0, 1.0, 1.0, 1.0]),
        ),
        (
            _batch([1, 3, 5, 0], [1, 3, 5, 0], [0.0, 2.0, 0.0, 0.0]),
            _batch([6, 4, 2, 0], [6, 4, 2, 0], [0.0, 0.25, 1.25, 2.5]),
        ),
    )


def main(output_path: str) -> None:
    rank = int(os.environ["RANK"])
    torch.manual_seed(20260710)
    model = _ProbeModel()
    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    args = build_training_args(
        output_dir=Path(output_path).parent / f"trainer-rank-{rank}",
        gradient_accumulation_steps=2,
        average_tokens_across_devices=True,
    )
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        data_collator=lambda rows: rows,
        loss_name="causal_lm",
    )
    if not torch.distributed.is_initialized():
        raise RuntimeError("Trainer did not initialize the torchrun process group.")

    all_batches = _rank_batches()
    local_batches = list(all_batches[rank])
    denominator = trainer._get_num_items_in_batch(local_batches, torch.device("cpu"))
    assert denominator is not None

    ddp_model = DistributedDataParallel(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    for microstep, batch in enumerate(local_batches):
        sync_context = (
            ddp_model.no_sync() if microstep + 1 < len(local_batches) else _NullContext()
        )
        with sync_context:
            loss = trainer.compute_loss(
                ddp_model,
                batch,
                num_items_in_batch=denominator,
            )
            loss.backward()
    optimizer.step()

    if rank == 0:
        reference = _ProbeModel()
        reference.load_state_dict(initial_state)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        reference_optimizer.zero_grad(set_to_none=True)
        merged = {
            key: torch.cat(
                [batch[key] for rank_rows in all_batches for batch in rank_rows],
                dim=0,
            )
            for key in ("input_ids", "labels", "loss_scale")
        }
        reference_outputs = reference(
            input_ids=merged["input_ids"],
            labels=merged["labels"],
        )
        reference_loss = causal_lm_cross_entropy(
            logits=reference_outputs.logits,
            labels=merged["labels"],
            loss_scale=merged["loss_scale"],
        )
        reference_loss.backward()
        reference_optimizer.step()

        max_parameter_error = max(
            float((actual - expected).abs().max())
            for actual, expected in zip(
                ddp_model.module.state_dict().values(),
                reference.state_dict().values(),
                strict=True,
            )
        )
        Path(output_path).write_text(
            json.dumps(
                {
                    "global_denominator": float(denominator),
                    "reference_loss": float(reference_loss.detach()),
                    "max_parameter_error": max_parameter_error,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        _ = exc_type, exc_value, traceback
        return False


if __name__ == "__main__":
    main(sys.argv[1])
