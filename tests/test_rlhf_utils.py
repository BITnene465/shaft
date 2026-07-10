from __future__ import annotations

from types import SimpleNamespace
import warnings

import pytest
import torch

from shaft.algorithms.grpo_rewards import build_grpo_reward_functions
from shaft.algorithms.rlhf_utils import (
    build_ppo_value_and_reward_models,
    build_trl_dpo_config,
    build_trl_grpo_config,
    build_trl_ppo_config,
    validate_ppo_runtime_requirements,
)
from shaft.config import DPOConfig as ShaftDPOConfig
from shaft.config import GRPOConfig as ShaftGRPOConfig
from shaft.config import PPOConfig as ShaftPPOConfig
from shaft.config import GRPORewardConfig
from shaft.config import GRPORolloutConfig, GRPOVLLMConfig
from shaft.data import ShaftGroupedSampleSampler, ShaftSamplePlan
from shaft.model import build_model_meta
from shaft.training import ShaftDPOTrainer, ShaftGRPOTrainer, ShaftPPOTrainer
from tests.support.training import TinyModel as _TinyModel
from tests.support.training import build_training_args


pytestmark = pytest.mark.component


def test_grpo_trainer_uses_epoch_resumable_grouped_sample_refs() -> None:
    plan = ShaftSamplePlan(
        {"ds": 8},
        {"ds": 1.0},
        strategy="weighted",
        seed=3,
    )
    trainer = object.__new__(ShaftGRPOTrainer)
    trainer.sample_plan = plan
    trainer.num_generations = 2
    trainer.num_iterations = 1
    trainer.shuffle_dataset = True
    trainer.args = SimpleNamespace(
        generation_batch_size=4,
        steps_per_generation=2,
        seed=7,
    )

    sampler = trainer._get_train_sampler()
    sampler.set_epoch(2)

    assert isinstance(sampler, ShaftGroupedSampleSampler)
    assert {ref.context.plan_cycle for ref in sampler} == {2}


def test_build_trl_dpo_config_from_training_args() -> None:
    args = build_training_args(
        output_dir="/tmp/shaft_dpo_config_smoke",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dpo_args = build_trl_dpo_config(
            train_args=args,
            rlhf_config=ShaftDPOConfig(
                beta=0.2,
                label_smoothing=0.05,
                loss_type="sigmoid",
                precompute_ref_log_probs=True,
                use_weighting=True,
            ),
        )
    assert all("push_to_hub_token" not in str(w.message) for w in caught)
    assert dpo_args.beta == pytest.approx(0.2)
    assert dpo_args.label_smoothing == pytest.approx(0.05)
    assert dpo_args.loss_type == ["sigmoid"]
    assert dpo_args.precompute_ref_log_probs is True
    assert dpo_args.use_weighting is True


def test_build_trl_ppo_config_from_training_args() -> None:
    args = build_training_args(
        output_dir="/tmp/shaft_ppo_config_smoke",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ppo_args = build_trl_ppo_config(
            train_args=args,
            rlhf_config=ShaftPPOConfig(
                cliprange=0.2,
                cliprange_value=0.2,
                kl_coef=0.03,
                vf_coef=0.2,
                gamma=0.99,
                lam=0.95,
                whiten_rewards=True,
                response_length=64,
                temperature=0.7,
                num_ppo_epochs=2,
                num_mini_batches=1,
                local_rollout_forward_batch_size=8,
                num_sample_generations=0,
                stop_token="eos",
                train_value_backbone=False,
            ),
        )
    assert all("push_to_hub_token" not in str(w.message) for w in caught)
    assert ppo_args.cliprange == pytest.approx(0.2)
    assert ppo_args.cliprange_value == pytest.approx(0.2)
    assert ppo_args.kl_coef == pytest.approx(0.03)
    assert ppo_args.vf_coef == pytest.approx(0.2)
    assert ppo_args.response_length == 64
    assert ppo_args.temperature == pytest.approx(0.7)
    assert ppo_args.num_ppo_epochs == 2


def test_build_trl_grpo_config_from_training_args() -> None:
    args = build_training_args(
        output_dir="/tmp/shaft_grpo_config_smoke",
        gradient_accumulation_steps=2,
        max_steps=7,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grpo_args = build_trl_grpo_config(
            train_args=args,
            rlhf_config=ShaftGRPOConfig(
                beta=0.01,
                rollout=GRPORolloutConfig(
                    num_generations=4,
                    max_completion_length=96,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=16,
                    min_p=0.05,
                    repetition_penalty=1.1,
                    generation_kwargs={"frequency_penalty": 0.1},
                    cache_implementation="static",
                    use_transformers_paged=True,
                ),
                vllm=GRPOVLLMConfig(
                    enabled=True,
                    mode="colocate",
                    model_impl="transformers",
                    enable_sleep_mode=True,
                    gpu_memory_utilization=0.25,
                    max_model_length=4096,
                    tensor_parallel_size=1,
                ),
                reward_functions=[
                    GRPORewardConfig(name="parse_success", codec="json_any", weight=0.25),
                    GRPORewardConfig(name="exact_match", codec="json_any", weight=2.0),
                ],
            ),
        )
    assert all("push_to_hub_token" not in str(w.message) for w in caught)
    assert grpo_args.beta == pytest.approx(0.01)
    assert grpo_args.max_steps == 7
    assert grpo_args.num_generations == 4
    assert grpo_args.max_completion_length == 96
    assert grpo_args.temperature == pytest.approx(0.8)
    assert grpo_args.top_p == pytest.approx(0.95)
    assert grpo_args.top_k == 16
    assert grpo_args.min_p == pytest.approx(0.05)
    assert grpo_args.repetition_penalty == pytest.approx(1.1)
    assert grpo_args.generation_kwargs == {"frequency_penalty": 0.1}
    assert grpo_args.cache_implementation == "static"
    assert grpo_args.use_transformers_paged is True
    assert grpo_args.use_vllm is True
    assert grpo_args.vllm_mode == "colocate"
    assert grpo_args.vllm_model_impl == "transformers"
    assert grpo_args.vllm_enable_sleep_mode is True
    assert grpo_args.vllm_gpu_memory_utilization == pytest.approx(0.25)
    assert grpo_args.vllm_max_model_length == 4096
    assert grpo_args.vllm_tensor_parallel_size == 1
    assert grpo_args.reward_weights == [0.25, 2.0]


def test_build_trl_grpo_config_sets_bf16_model_init_kwargs() -> None:
    args = build_training_args(
        output_dir="/tmp/shaft_grpo_config_bf16",
        gradient_accumulation_steps=1,
    )
    args.bf16 = True

    grpo_args = build_trl_grpo_config(
        train_args=args,
        rlhf_config=ShaftGRPOConfig(
            reward_functions=[GRPORewardConfig(name="parse_success", codec="json_any")]
        ),
    )

    assert grpo_args.model_init_kwargs == {"dtype": "bfloat16"}


def test_shaft_rlhf_trainer_classes_are_importable() -> None:
    assert isinstance(ShaftDPOTrainer, type)
    assert isinstance(ShaftPPOTrainer, type)
    assert isinstance(ShaftGRPOTrainer, type)


def test_build_grpo_reward_functions_supports_exact_match_and_parse_success() -> None:
    reward_funcs = build_grpo_reward_functions(
        [
            GRPORewardConfig(name="parse_success", codec="json_any", weight=0.5),
            GRPORewardConfig(name="exact_match", codec="json_any", weight=2.0),
        ]
    )
    parse_reward, exact_reward = reward_funcs
    assert parse_reward.__name__ == "grpo_reward_parse_success"
    assert exact_reward.__name__ == "grpo_reward_exact_match"
    assert parse_reward(
        completions=['{"ok": 1}', "not-json"],
        target_text=['{"ok": 1}', '{"ok": 0}'],
    ) == [1.0, 0.0]
    assert parse_reward(
        completions=["["],
        target_text=["[]"],
    ) == [0.0]
    assert exact_reward(
        completions=['{"ok": 1}', '{"ok": 0}'],
        target_text=['{"ok": 1}', '{"ok": 1}'],
    ) == [1.0, 0.0]
    assert exact_reward(
        completions=["["],
        target_text=["[]"],
    ) == [0.0]
    assert exact_reward(
        completions=[
            [{"role": "assistant", "content": [{"type": "text", "text": '{"ok": 1}'}]}],
            [{"role": "assistant", "content": '{"ok": 0}'}],
        ],
        target_text=['{"ok": 1}', '{"ok": 1}'],
    ) == [1.0, 0.0]


def test_build_grpo_reward_functions_supports_grounding_iou() -> None:
    reward_func = build_grpo_reward_functions(
        [
            GRPORewardConfig(
                name="grounding_iou",
                codec="json_list",
                weight=2.0,
            )
        ]
    )[0]

    assert reward_func.__name__ == "grpo_reward_grounding_iou"

    rewards = reward_func(
        completions=[
            '[{"label":"icon","bbox_2d":[0,0,100,100]}]',
            '[{"label":"image","bbox_2d":[0,0,100,100]}, {"label":"icon","bbox_2d":[500,500,600,600]}]',
            "[",
            "not-json",
        ],
        target_text=[
            '[{"label":"icon","bbox_2d":[0,0,100,100]}]',
            '[{"label":"image","bbox_2d":[0,0,100,100]}]',
            "[]",
            '[{"label":"icon","bbox_2d":[0,0,100,100]}]',
        ],
    )

    assert rewards == [pytest.approx(1.0), pytest.approx(0.5), 0.0, 0.0]


def test_build_grpo_reward_functions_supports_grounding_det_f1() -> None:
    reward_func = build_grpo_reward_functions(
        [
            GRPORewardConfig(
                name="grounding_det_f1",
                codec="json_list",
                weight=1.0,
                params={"iou_threshold": 0.5},
            )
        ]
    )[0]

    assert reward_func.__name__ == "grpo_reward_grounding_det_f1"

    rewards = reward_func(
        completions=[
            '[{"label":"icon","bbox_2d":[0,0,100,100]}]',
            '[{"label":"image","bbox_2d":[0,0,100,100]}, {"label":"icon","bbox_2d":[500,500,600,600]}]',
            '[{"label":"icon","bbox_2d":[500,500,600,600]}]',
            "[",
        ],
        target_text=[
            '[{"label":"icon","bbox_2d":[0,0,100,100]}]',
            '[{"label":"image","bbox_2d":[0,0,100,100]}]',
            '[{"label":"icon","bbox_2d":[0,0,100,100]}]',
            "[]",
        ],
    )

    assert rewards == [pytest.approx(1.0), pytest.approx(2 / 3), 0.0, 0.0]


def test_ppo_requires_explicit_random_reward_opt_in() -> None:
    model = _TinyModel()
    with pytest.raises(ValueError, match="allow_untrained_reward_model"):
        build_ppo_value_and_reward_models(
            model=model,
            train_value_backbone=False,
            value_model_mode="shared_backbone",
            reward_model_mode="adapter_disabled_policy",
            allow_untrained_reward_model=False,
        )
    value_model, reward_model = build_ppo_value_and_reward_models(
        model=model,
        train_value_backbone=False,
        value_model_mode="copy_backbone",
        reward_model_mode="copy_backbone",
        allow_untrained_reward_model=True,
    )
    assert isinstance(value_model, torch.nn.Module)
    assert isinstance(reward_model, torch.nn.Module)


def test_ppo_multimodal_guard_requires_opt_in() -> None:
    meta = build_model_meta("smoke_vlm")
    with pytest.raises(ValueError, match="allow_text_only_multimodal_ppo"):
        validate_ppo_runtime_requirements(
            model_meta=meta,
            model=_TinyModel(),
            finetune_mode="lora",
            rlhf_config=ShaftPPOConfig(allow_text_only_multimodal_ppo=False),
        )
    validate_ppo_runtime_requirements(
        model_meta=meta,
        model=_TinyModel(),
        finetune_mode="lora",
        rlhf_config=ShaftPPOConfig(
            allow_text_only_multimodal_ppo=True,
            reward_model_mode="copy_backbone",
        ),
    )


def test_ppo_shared_value_backbone_keeps_policy_trainable() -> None:
    model = _TinyModel()
    value_model, reward_model = build_ppo_value_and_reward_models(
        model=model,
        train_value_backbone=False,
        value_model_mode="shared_backbone",
        reward_model_mode="copy_backbone",
        allow_untrained_reward_model=True,
    )
    assert value_model.backbone is not None
    assert any(param.requires_grad for param in model.parameters())
    assert all(not param.requires_grad for param in reward_model.score.parameters())


def test_ppo_rejects_full_finetune_mode() -> None:
    meta = build_model_meta("smoke_vlm")
    with pytest.raises(ValueError, match="finetune.mode='full'"):
        validate_ppo_runtime_requirements(
            model_meta=meta,
            model=_TinyModel(),
            finetune_mode="full",
            rlhf_config=ShaftPPOConfig(allow_text_only_multimodal_ppo=True),
        )
