"""Four-rank CUDA numerics probe; launched by the GPU suite, not a training entry."""

# ruff: noqa: E402 -- cache isolation must precede torch/FLA imports.
import os
from pathlib import Path
import tempfile
from contextlib import nullcontext

cache = Path(tempfile.mkdtemp(prefix=f"shaft-ce-rank{os.environ.get('RANK', '0')}-", dir="/tmp"))
for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
    os.environ[key] = str(cache / key.lower())

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel

from shaft.model.descriptor import ResolvedModelDescriptor
from shaft.config.training import TrainLigerConfig
from shaft.model.qwen35vl import QWEN35VL_META
from shaft.training.sft_trainer import ShaftSFTTrainer
from tests.support.training import build_training_args
from tests.test_training_numerics import collate, model as model_fixture, reference_loss, rows


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    patch = pytest.MonkeyPatch()
    fixture = model_fixture.__wrapped__(patch)
    model = next(fixture).cuda().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    adapter = QWEN35VL_META.resolve_adapter(
        model_name_or_path="tiny-qwen35",
        descriptor=ResolvedModelDescriptor.from_payload(model.config.to_dict(), source="test"),
    )
    trainer = ShaftSFTTrainer(
        model=model,
        model_adapter=adapter,
        liger_config=TrainLigerConfig(fused_linear_ce=True, rms_norm=True, swiglu=True),
        args=build_training_args(
            cache, use_cpu=False, average_tokens_across_devices=True, gradient_accumulation_steps=2
        ),
    )
    assert torch.distributed.get_world_size() == 4
    wrapped = DistributedDataParallel(model, device_ids=[rank])
    for weighted in (False, True):
        items = rows(4, weighted)
        # Rank zero has no supervised tokens: it must still join every reduction.
        items[0]["labels"].fill_(-100)
        global_batch = {key: value.cuda() for key, value in collate(items * 2).items()}
        model.zero_grad(set_to_none=True)
        reference_loss(model, global_batch).backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        batches = [
            {key: value.cuda() for key, value in collate([items[rank]]).items()} for _ in range(2)
        ]
        denominator = trainer._get_num_items_in_batch(batches, torch.device("cuda", rank))
        for index, batch in enumerate(batches):
            with wrapped.no_sync() if index == 0 else nullcontext():
                trainer.compute_loss(wrapped, batch, num_items_in_batch=denominator).backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-6, rtol=2e-4)
        print(f"PASS rank={rank} weighted={weighted} denominator={float(denominator)}", flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    fixture.close()
    patch.undo()


if __name__ == "__main__":
    main()
