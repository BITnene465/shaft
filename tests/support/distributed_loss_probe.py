from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers.utils import ModelOutput

from shaft.model.types import (
    ShaftAuxiliaryLossTerm,
    ShaftEvalAuxiliaryMetric,
    ShaftEvalAuxiliaryStatistic,
)
from shaft.training.loss import causal_lm_cross_entropy
from shaft.training.sft_trainer import ShaftSFTTrainer
from tests.support.training import build_training_args


class _ProbeModel(torch.nn.Module):
    def __init__(self, *, vocab_size: int = 7) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, vocab_size)
        self.router_scale = torch.nn.Parameter(torch.tensor(0.75))

    def forward(self, input_ids, labels=None, **kwargs):
        _ = labels, kwargs
        return _ProbeOutput(
            logits=self.embedding(input_ids),
            auxiliary_loss=self.router_scale
            * (input_ids.to(dtype=torch.float32).mean() + 1.0),
        )


@dataclass
class _ProbeOutput(ModelOutput):
    logits: torch.Tensor | None = None
    auxiliary_loss: torch.Tensor | None = None


class _ProbeAdapter:
    def __init__(self) -> None:
        self.local_eval_values: list[float] = []

    def prepare_sft_forward_inputs(self, *, model, inputs):
        _ = model
        return dict(inputs)

    def auxiliary_loss_names(self):
        return ("router_aux_loss",)

    def resolve_sft_auxiliary_loss_terms(self, *, model, outputs, inputs):
        _ = model, inputs
        return (
            ShaftAuxiliaryLossTerm(
                name="router_aux_loss",
                value=outputs.auxiliary_loss,
                coefficient=0.125,
            ),
        )

    def resolve_sft_eval_auxiliary_statistics(self, *, model, outputs, inputs):
        _ = model, outputs
        values = inputs["input_ids"].to(dtype=torch.float32).sum(
            dim=1,
            keepdim=True,
        )
        self.local_eval_values.extend(float(value) for value in values.flatten())
        return (
            ShaftEvalAuxiliaryStatistic(
                name="sample_value_mean",
                coefficient_key="router_aux_loss",
                coefficient=0.125,
                components={
                    "sum": values,
                    "count": torch.ones_like(values),
                },
            ),
        )

    def finalize_sft_eval_auxiliary_statistics(self, statistics):
        if len(statistics) != 1:
            raise AssertionError(statistics)
        statistic = statistics[0]
        value = statistic.components["sum"].sum() / statistic.components[
            "count"
        ].sum()
        return (
            ShaftEvalAuxiliaryMetric(
                name=statistic.name,
                value=value,
                coefficient_key=statistic.coefficient_key,
                coefficient=statistic.coefficient,
            ),
        )


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


def _stack_batches(*batches: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: torch.cat([batch[key] for batch in batches], dim=0)
        for key in ("input_ids", "labels", "loss_scale")
    }


def _rank_batches() -> tuple[tuple[dict[str, torch.Tensor], ...], ...]:
    first = _batch([0, 1, 2, 3], [0, 1, 2, 3], [0.0, 0.5, 1.0, 0.0])
    second = _batch([3, 2, 1, 0], [3, 2, 1, 0], [0.0, 1.0, 1.0, 1.0])
    third = _batch([1, 3, 5, 0], [1, 3, 5, 0], [0.0, 2.0, 0.0, 0.0])
    fourth = _batch([6, 4, 2, 0], [6, 4, 2, 0], [0.0, 0.25, 1.25, 2.5])
    fifth = _batch([2, 4, 6, 1], [2, 4, 6, 1], [0.0, 1.0, 0.0, 0.5])
    return (
        (first, second),
        (_stack_batches(third, fourth), fifth),
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
        per_device_eval_batch_size=1,
    )
    eval_rows = [
        {"input_ids": [0, 0, 0, 1], "labels": [0, 0, 0, 1]},
        {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
        {"input_ids": [5, 6, 6, 6], "labels": [5, 6, 6, 6]},
    ]

    def collate(rows):
        return {
            "input_ids": torch.tensor(
                [row["input_ids"] for row in rows],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [row["labels"] for row in rows],
                dtype=torch.long,
            ),
        }

    adapter = _ProbeAdapter()
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset=eval_rows,
        data_collator=collate,
        model_adapter=adapter,
        loss_name="causal_lm",
    )
    if not torch.distributed.is_initialized():
        raise RuntimeError("Trainer did not initialize the torchrun process group.")

    eval_metrics = trainer.evaluate()
    expected_eval_auxiliary = sum(sum(row["input_ids"]) for row in eval_rows) / len(
        eval_rows
    )
    local_eval_auxiliary = float(eval_metrics["eval_aux/sample_value_mean"])
    gathered_eval_auxiliary: list[float | None] = [None] * int(
        torch.distributed.get_world_size()
    )
    torch.distributed.all_gather_object(
        gathered_eval_auxiliary,
        local_eval_auxiliary,
    )
    gathered_local_eval_values: list[list[float] | None] = [None] * int(
        torch.distributed.get_world_size()
    )
    torch.distributed.all_gather_object(
        gathered_local_eval_values,
        adapter.local_eval_values,
    )
    normalized_local_values = [
        sorted(float(value) for value in values)
        for values in gathered_local_eval_values
        if values is not None
    ]
    if sorted(normalized_local_values) != [[1.0, 10.0], [1.0, 23.0]]:
        raise AssertionError(
            "The eval probe must expose one rank-exclusive contribution on each "
            f"rank: actual={gathered_local_eval_values}."
        )
    if any(
        abs(sum(values) / len(values) - expected_eval_auxiliary) <= 1e-7
        for values in normalized_local_values
    ):
        raise AssertionError(
            "Each rank-local eval mean must differ from the dataset-global mean: "
            f"local={gathered_local_eval_values}, global={expected_eval_auxiliary}."
        )
    if any(
        value is None or abs(float(value) - expected_eval_auxiliary) > 1e-6
        for value in gathered_eval_auxiliary
    ):
        raise AssertionError(
            "Every rank must observe the dataset-global eval auxiliary metric: "
            f"actual={gathered_eval_auxiliary}, expected={expected_eval_auxiliary}."
        )
    model.train()

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
        reference_auxiliary_loss = sum(
            reference(
                input_ids=batch["input_ids"],
            ).auxiliary_loss
            for rank_rows in all_batches
            for batch in rank_rows
        ) * (0.125 / 4.0)
        reference_loss = reference_loss + reference_auxiliary_loss
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
                    "rank_batch_sizes": [
                        [int(batch["input_ids"].shape[0]) for batch in rank_batches]
                        for rank_batches in all_batches
                    ],
                    "reference_loss": float(reference_loss.detach()),
                    "reference_auxiliary_loss": float(
                        reference_auxiliary_loss.detach()
                    ),
                    "eval_auxiliary": float(
                        eval_metrics["eval_aux/sample_value_mean"]
                    ),
                    "eval_auxiliary_by_rank": gathered_eval_auxiliary,
                    "eval_local_values_by_rank": gathered_local_eval_values,
                    "expected_eval_auxiliary": expected_eval_auxiliary,
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
