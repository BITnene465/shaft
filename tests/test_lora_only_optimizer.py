from __future__ import annotations

import unittest

import torch
from torch import nn

from vlm_structgen.core.config import ExperimentRuntimeConfig
from vlm_structgen.core.train.optim import build_optimizer


class _LoraOnlyDummyModel(nn.Module):
    def __init__(self, *, with_extra_trainable: bool) -> None:
        super().__init__()
        self.lora_a = nn.Parameter(torch.ones(4))
        self.extra = nn.Parameter(torch.ones(4), requires_grad=with_extra_trainable)


class LoraOnlyOptimizerTest(unittest.TestCase):
    def _build_config(self) -> ExperimentRuntimeConfig:
        config = ExperimentRuntimeConfig()
        config.finetune.mode = "lora"
        config.lora.enabled = True
        config.train.learning_rate = 1e-4
        config.train.lora_learning_rate = 2e-4
        return config

    def test_lora_only_optimizer_rejects_extra_trainable_params(self) -> None:
        model = _LoraOnlyDummyModel(with_extra_trainable=True)
        config = self._build_config()

        with self.assertRaises(ValueError):
            build_optimizer(model, config)

    def test_lora_only_optimizer_accepts_lora_only_params(self) -> None:
        model = _LoraOnlyDummyModel(with_extra_trainable=False)
        config = self._build_config()

        optimizer = build_optimizer(model, config)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["name"], "lora_params")


if __name__ == "__main__":
    unittest.main()
