from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from vlm_structgen.core.config import ExperimentRuntimeConfig
from vlm_structgen.core.modeling.builder import _finalize_model_for_runtime


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.visual = nn.Linear(4, 4, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class LoraOnlyBuilderTest(unittest.TestCase):
    def test_lora_mode_freezes_embeddings_and_wraps_lm_head(self) -> None:
        captured: dict[str, object] = {}

        fake_peft = types.ModuleType("peft")

        class FakeLoraConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeTaskType:
            CAUSAL_LM = "CAUSAL_LM"

        def fake_get_peft_model(model, lora_config):  # noqa: ANN001
            captured["target_modules"] = list(lora_config.target_modules)
            captured["task_type"] = lora_config.task_type
            return model

        fake_peft.LoraConfig = FakeLoraConfig
        fake_peft.TaskType = FakeTaskType
        fake_peft.get_peft_model = fake_get_peft_model

        config = ExperimentRuntimeConfig()
        config.finetune.mode = "lora"
        config.lora.enabled = True
        config.lora.lang_target_modules = []
        config.lora.vis_target_modules = []
        config.lora.proj_target_modules = []
        config.lora.lm_head_target_modules = ["lm_head"]
        config.model.freeze_vision_tower = True
        config.model.train_projector = False
        config.train.gradient_checkpointing = False

        model = _DummyModel()

        with patch.dict(sys.modules, {"peft": fake_peft}):
            finalized = _finalize_model_for_runtime(model, config)

        self.assertIs(finalized, model)
        self.assertEqual(captured["target_modules"], ["lm_head"])
        self.assertEqual(captured["task_type"], "CAUSAL_LM")
        self.assertFalse(model.embed_tokens.weight.requires_grad)
        self.assertFalse(model.lm_head.weight.requires_grad)
        self.assertFalse(model.visual.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
