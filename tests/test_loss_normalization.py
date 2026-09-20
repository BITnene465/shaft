from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import SequentialSampler

from shaft.config.training import TrainConfig
from shaft.training.loss_normalization import supervision_mass, validate_loss_normalization
from shaft.training.sft_trainer import ShaftSFTTrainer
from tests.support.training import build_training_args
from tests.test_training_numerics import model as model
from tests.test_training_numerics import rows, collate, reference_loss

pytestmark = pytest.mark.contract


@pytest.mark.parametrize('mode', ['global_token', 'rank_token', 'microbatch_token'])
@pytest.mark.parametrize('count', [3, 4])
@pytest.mark.parametrize('weighted', [False, True])
def test_real_trainer_update(model, tmp_path, mode, count, weighted):
    items = rows(count, weighted)
    initial = deepcopy(model.state_dict())
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    if mode == 'microbatch_token':
        reference = sum(reference_loss(model, collate([row])) for row in items) / count
    else:
        reference = reference_loss(model, collate(items))
    reference.backward()
    optimizer.step()
    expected = deepcopy(model.state_dict())
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    trainer = ShaftSFTTrainer(
        model=model, loss_normalization=mode,
        args=build_training_args(
            tmp_path, per_device_train_batch_size=1, gradient_accumulation_steps=4,
            max_steps=1, max_grad_norm=0., average_tokens_across_devices=True,
            remove_unused_columns=False, save_strategy='no', logging_strategy='no',
            disable_tqdm=True, dataloader_pin_memory=False,
        ),
        train_dataset=items, train_sampler=SequentialSampler(items), data_collator=collate,
        optimizers=(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)),
    )
    trainer.train()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize('mode', ['global_token', 'rank_token', 'microbatch_token'])
def test_window_contract(mode):
    batches = [collate([r]) for r in rows(3, True)]
    mass = sum(supervision_mass(b['labels'], b['loss_scale']) for b in batches)
    fake = SimpleNamespace(
        loss_normalization=mode, ignore_index=-100,
        args=SimpleNamespace(average_tokens_across_devices=True, world_size=2, n_gpu=1),
        accelerator=SimpleNamespace(gather=lambda x: torch.stack([x, x*2])),
    )
    actual = ShaftSFTTrainer._get_num_items_in_batch(fake, batches, torch.device('cpu'))
    expected = 3 if mode == 'microbatch_token' else mass * (3 if mode == 'global_token' else 1)
    torch.testing.assert_close(torch.as_tensor(actual), torch.as_tensor(expected))


def test_config_default_and_rejection():
    assert TrainConfig().loss_normalization == 'global_token'
    for invalid in ['lf', '', None, True, 'GLOBAL_TOKEN']:
        with pytest.raises(ValueError):
            validate_loss_normalization(invalid)


@pytest.mark.parametrize('mode', ['global_token', 'rank_token', 'microbatch_token'])
def test_yaml_roundtrip(tmp_path, mode):
    from shaft.config import load_config
    from tests.support.configs import write_config_yaml
    config = load_config(write_config_yaml(tmp_path, f'''
model:
  model_type: qwen35vl
  model_name_or_path: Qwen/Qwen3.5-0.8B
train:
  loss_normalization: {mode}
eval:
  enabled: false
data:
  datasets:
    - dataset_name: ds
      train_path: train.jsonl
'''))
    assert config.train.loss_normalization == mode


def test_zero_and_masked_supervision():
    labels = torch.tensor([[1, -100, -100]])
    assert supervision_mass(labels).item() == 0
    labels = torch.tensor([[1, 2, -100]])
    assert supervision_mass(labels, torch.tensor([[100., .25, 900.]])).item() == .25


@pytest.mark.parametrize('mode', ['global_token', 'rank_token', 'microbatch_token'])
def test_eval_keeps_token_normalized_components(model, tmp_path, mode):
    model.eval()
    batch = collate(rows(3, True))
    expected = reference_loss(model, batch).detach()
    trainer = ShaftSFTTrainer(
        model=model, loss_normalization=mode,
        args=build_training_args(tmp_path, average_tokens_across_devices=True),
    )
    model.eval()
    with torch.no_grad():
        trainer.compute_loss(model, batch)
    numerator, denominator = trainer._shaft_eval_primary_loss_accumulator
    torch.testing.assert_close(numerator/denominator, expected, atol=2e-6, rtol=2e-5)
