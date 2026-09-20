import pytest
import torch

from shaft.config.training import TrainLigerConfig
from shaft.model.descriptor import ResolvedModelDescriptor
from shaft.model.qwen35vl import QWEN35VL_META
from shaft.training.sft_trainer import ShaftSFTTrainer
from tests.support.training import build_training_args
from tests.test_training_numerics import model as model
from tests.test_training_numerics import rows, collate, reference_loss
from tests.test_fused_linear_ce_gpu import require_cuda as require_cuda

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize('mode', ['global_token', 'rank_token', 'microbatch_token'])
@pytest.mark.parametrize('weighted', [False, True])
def test_fused_normalization(model, tmp_path, mode, weighted):
    model.cuda().train()
    items = rows(3, weighted)
    batches = [{k:v.cuda() for k,v in collate([r]).items()} for r in items]
    if mode == 'microbatch_token':
        expected = sum(reference_loss(model, b) for b in batches)/len(batches)
    else:
        masses = []
        for b in batches:
            weights = b['labels'][:, 1:].ne(-100).float()
            if weighted:
                weights *= b['loss_scale'][:, 1:]
            masses.append(weights.sum())
        expected = sum(reference_loss(model, b)*n for b,n in zip(batches, masses))/sum(masses)
    expected.backward()
    gradients = {k:p.grad.clone() for k,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    adapter = QWEN35VL_META.resolve_adapter(
        model_name_or_path='tiny-qwen35',
        descriptor=ResolvedModelDescriptor.from_payload(model.config.to_dict(), source='test'),
    )
    trainer = ShaftSFTTrainer(
        model=model, model_adapter=adapter, loss_normalization=mode,
        liger_config=TrainLigerConfig(fused_linear_ce=True, rms_norm=True, swiglu=True),
        args=build_training_args(tmp_path, use_cpu=False, average_tokens_across_devices=True),
    )
    denominator = trainer._get_num_items_in_batch(batches, torch.device('cuda'))
    value = 0
    for b in batches:
        loss = trainer.compute_loss(model, b, num_items_in_batch=denominator)
        loss.backward()
        value += loss.detach()
    torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-5)
    for k,p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradients[k], atol=2e-6, rtol=2e-4)
