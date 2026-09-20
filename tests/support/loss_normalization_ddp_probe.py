"""Run with torchrun --nproc_per_node=2; real Gloo/DDP optimizer oracle."""
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from shaft.training.sft_trainer import ShaftSFTTrainer
from tests.support.training import build_training_args


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.weight = torch.nn.Parameter(torch.tensor([.1, -.2, .3, .4, -.5]))

    def forward(self, input_ids, **kwargs):
        return CausalLMOutput(logits=self.weight.expand(*input_ids.shape, 5))


def collate(rows):
    return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


def main():
    torch.set_num_threads(1)
    rank = int(os.environ['RANK'])
    for count in (6, 8):
        rows = []
        for i in range(count):
            labels = torch.full((6,), i % 5, dtype=torch.long)
            labels[:1+i%4] = -100
            rows.append(dict(input_ids=torch.ones(6, dtype=torch.long), labels=labels,
                             loss_scale=torch.linspace(.25, 1.75, 6)))
        for mode in ('global_token', 'rank_token', 'microbatch_token'):
            reference = Model()
            sums, masses = [], []
            for row in rows:
                y = row['labels'][1:]
                w = row['loss_scale'][1:] * y.ne(-100)
                sums.append((F.cross_entropy(reference.weight.expand(5,5), y, reduction='none') * w).sum())
                masses.append(w.sum())
            if mode == 'global_token':
                loss = sum(sums)/sum(masses)
            elif mode == 'rank_token':
                loss = sum(sum(sums[r::2])/sum(masses[r::2]) for r in range(2))/2
            else:
                loss = sum(s/n for s,n in zip(sums,masses))/count
            loss.backward()
            expected = reference.weight.detach() - .1*reference.weight.grad
            model = Model()
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            args = build_training_args(
                Path(os.environ['PROBE_OUT'])/f'{mode}-{count}',
                per_device_train_batch_size=1, gradient_accumulation_steps=4,
                max_steps=1, max_grad_norm=0., average_tokens_across_devices=True,
                remove_unused_columns=False, save_strategy='no', logging_strategy='no',
                disable_tqdm=True, dataloader_pin_memory=False, ddp_find_unused_parameters=False,
            )
            trainer = ShaftSFTTrainer(
                model=model, args=args, loss_normalization=mode,
                train_dataset=rows, train_sampler=SequentialSampler(rows),
                data_collator=collate,
                optimizers=(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)),
            )
            trainer.train()
            torch.testing.assert_close(model.weight, expected, atol=2e-6, rtol=2e-5)
            if rank == 0:
                print(f'PASS {mode} count={count}', flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
