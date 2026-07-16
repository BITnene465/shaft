from __future__ import annotations

import json
import random
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch
from accelerate.data_loader import skip_first_batches
from torch.utils.data import DataLoader

from shaft.algorithms import ShaftTrainerSpec
from shaft.algorithms.base import model_topology_signature, trainer_spec_contract
from shaft.algorithms.grpo_rewards import build_grpo_reward_functions
from shaft.algorithms.rlhf_utils import (
    build_ppo_value_and_reward_models,
    build_trl_dpo_config,
    build_trl_grpo_config,
    build_trl_ppo_config,
    resolve_grpo_grouped_sample_contract,
    resolve_grpo_checkpoint_step_cadence,
    validate_grpo_checkpoint_cadence,
    validate_grpo_rollout_checkpointability,
    validate_grpo_vllm_runtime_compatibility,
    validate_ppo_runtime_requirements,
)
from shaft.algorithms import rlhf_utils as rlhf_utils_module
from shaft.config import DPOConfig as ShaftDPOConfig
from shaft.config import GRPOConfig as ShaftGRPOConfig
from shaft.config import PPOConfig as ShaftPPOConfig
from shaft.config import GRPORewardConfig
from shaft.config import GRPORolloutConfig, GRPOVLLMConfig
from shaft.config import TrainConfig
from shaft.data import ShaftGroupedSampleContract, ShaftGroupedSampleSampler, ShaftSamplePlan
from shaft.model import build_model_meta
from shaft.training import ShaftDPOTrainer, ShaftGRPOTrainer, ShaftPPOTrainer
from shaft.training import trl_trainers as trl_trainer_module
from shaft.training.checkpointing import ShaftCheckpointCommitMixin
from tests.support.training import TinyModel as _TinyModel
from tests.support.training import build_training_args


pytestmark = pytest.mark.component


def _trainer_constructor_helper_v1(instance) -> None:
    instance.marker = 1


def _trainer_constructor_helper_v2(instance) -> None:
    instance.marker = 2


_TRAINER_CONSTRUCTOR_HELPER = _trainer_constructor_helper_v1


def test_trainer_spec_defers_the_constructor_boundary() -> None:
    constructor_calls: list[int] = []

    class _Trainer:
        def __init__(self, *, value: int) -> None:
            constructor_calls.append(value)

    spec = ShaftTrainerSpec(
        trainer_cls=_Trainer,
        kwargs={"value": 7},
        contract={"version": 1, "algorithm": "test"},
    )

    assert constructor_calls == []
    assert spec.implementation["type"].endswith("._Trainer")
    assert len(spec.implementation["semantic_sha256"]) == 64
    assert len(spec.fingerprint) == 64
    trainer = spec.build()
    assert isinstance(trainer, _Trainer)
    assert constructor_calls == [7]


def test_trainer_spec_fingerprint_binds_same_name_constructor_implementation() -> None:
    def _trainer_cls(marker: int):
        class _Trainer:
            def __init__(self) -> None:
                self.marker = marker

        return _Trainer

    first = ShaftTrainerSpec(
        trainer_cls=_trainer_cls(1),
        kwargs={},
        contract={"version": 1, "algorithm": "test"},
    )
    second = ShaftTrainerSpec(
        trainer_cls=_trainer_cls(2),
        kwargs={},
        contract={"version": 1, "algorithm": "test"},
    )

    assert first.implementation["type"] == second.implementation["type"]
    assert (
        first.implementation["semantic_sha256"]
        != second.implementation["semantic_sha256"]
    )
    assert first.fingerprint != second.fingerprint


def test_trainer_spec_fingerprint_binds_live_constructor_global_helper(monkeypatch) -> None:
    class _Trainer:
        def __init__(self) -> None:
            _TRAINER_CONSTRUCTOR_HELPER(self)

    original = ShaftTrainerSpec(
        trainer_cls=_Trainer,
        kwargs={},
        contract={"version": 1, "algorithm": "test"},
    )
    original_fingerprint = original.fingerprint
    monkeypatch.setattr(
        "tests.test_rlhf_utils._TRAINER_CONSTRUCTOR_HELPER",
        _trainer_constructor_helper_v2,
    )
    changed = ShaftTrainerSpec(
        trainer_cls=_Trainer,
        kwargs={},
        contract={"version": 1, "algorithm": "test"},
    )

    assert original.implementation["type"] == changed.implementation["type"]
    assert original_fingerprint != changed.fingerprint


def test_trainer_spec_contract_excludes_rank_local_args() -> None:
    class _Args:
        def __init__(self, *, local_rank: int, learning_rate: float) -> None:
            self.local_rank = local_rank
            self.learning_rate = learning_rate

        def to_dict(self) -> dict[str, object]:
            return {
                "local_rank": self.local_rank,
                "process_index": self.local_rank,
                "device": f"cuda:{self.local_rank}",
                "learning_rate": self.learning_rate,
                "per_device_train_batch_size": 1,
            }

    train_config = TrainConfig()
    rank_zero = trainer_spec_contract(
        algorithm="ppo",
        args=_Args(local_rank=0, learning_rate=1e-5),
        train_config=train_config,
    )
    rank_one = trainer_spec_contract(
        algorithm="ppo",
        args=_Args(local_rank=1, learning_rate=1e-5),
        train_config=train_config,
    )
    changed_learning_rate = trainer_spec_contract(
        algorithm="ppo",
        args=_Args(local_rank=1, learning_rate=2e-5),
        train_config=train_config,
    )

    assert rank_zero == rank_one
    assert rank_zero != changed_learning_rate


def test_model_topology_signature_ignores_values_but_binds_trainable_shape() -> None:
    class _Graph(torch.nn.Module):
        def __init__(self, *, with_activation: bool) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(3, 2, bias=True)
            if with_activation:
                self.activation = torch.nn.ReLU()

    first = torch.nn.Linear(3, 2, bias=True)
    second = torch.nn.Linear(3, 2, bias=True)
    with torch.no_grad():
        first.weight.fill_(1.0)
        second.weight.fill_(9.0)

    first_signature = model_topology_signature(first)
    assert first_signature == model_topology_signature(second)

    second.weight.requires_grad_(False)
    assert first_signature != model_topology_signature(second)
    assert first_signature != model_topology_signature(torch.nn.Linear(4, 2, bias=True))
    assert model_topology_signature(_Graph(with_activation=False)) != model_topology_signature(
        _Graph(with_activation=True)
    )


@pytest.mark.parametrize(
    ("trainer_cls", "base_cls"),
    [
        (ShaftDPOTrainer, trl_trainer_module._TRLDPOTrainer),
        (ShaftGRPOTrainer, trl_trainer_module._TRLGRPOTrainer),
    ],
)
def test_exact_resumable_rlhf_eval_preserves_training_rng(
    trainer_cls,
    base_cls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = object.__new__(trainer_cls)
    trainer.is_in_train = True
    trainer.eval_dataset = []
    if trainer_cls is ShaftDPOTrainer:
        trainer.eval_config = None
    else:
        trainer.online_eval_runner = None

    def _consume_rng(self, *args, **kwargs):
        _ = self, args, kwargs
        random.random()
        np.random.random()
        torch.rand(())
        return {"eval_loss": 0.0}

    monkeypatch.setattr(base_cls, "evaluate", _consume_rng)
    random.seed(29)
    np.random.seed(29)
    torch.manual_seed(29)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    assert trainer.evaluate(eval_dataset=[]) == {"eval_loss": 0.0}
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_grpo_trainer_uses_epoch_resumable_grouped_sample_refs() -> None:
    plan = ShaftSamplePlan(
        {"ds": 8},
        {"ds": 1.0},
        strategy="weighted",
        seed=3,
    )
    trainer = object.__new__(ShaftGRPOTrainer)
    trainer.sample_plan = plan
    trainer.grouped_sample_contract = ShaftGroupedSampleContract(
        mini_repeat_count=2,
        batch_size=2,
        iteration_count=1,
        steps_per_iteration=2,
    )
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


def test_grpo_checkpoint_cadence_rejects_mid_generation_save_and_resume(
    tmp_path,
) -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=1,
        steps_per_generation=2,
        num_iterations=1,
        save_strategy="steps",
        save_steps=1,
        max_steps=4,
        num_train_epochs=1.0,
        use_vllm=False,
    )

    assert resolve_grpo_checkpoint_step_cadence(args) == 2
    with pytest.raises(ValueError, match="save_steps=1"):
        validate_grpo_checkpoint_cadence(args, epoch_microsteps=4)

    args.save_steps = 2
    validate_grpo_checkpoint_cadence(args, epoch_microsteps=4)

    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generation-reuse cycle"):
        validate_grpo_checkpoint_cadence(
            args,
            epoch_microsteps=4,
            resume_checkpoint=checkpoint,
        )

    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 2}',
        encoding="utf-8",
    )
    validate_grpo_checkpoint_cadence(
        args,
        epoch_microsteps=4,
        resume_checkpoint=checkpoint,
    )


@pytest.mark.parametrize("global_step", [True, "2"])
def test_grpo_checkpoint_cadence_rejects_noncanonical_global_step(
    tmp_path,
    global_step: object,
) -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=1,
        steps_per_generation=2,
        num_iterations=1,
        save_strategy="no",
    )
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": global_step}),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=r"global_step must be a JSON integer"):
        validate_grpo_checkpoint_cadence(
            args,
            epoch_microsteps=4,
            resume_checkpoint=checkpoint,
        )


@pytest.mark.parametrize(
    "trainer_state",
    [
        '{"global_step": 2, "global_step": 4}',
        '{"global_step": NaN}',
        '{"global_step": Infinity}',
    ],
)
def test_grpo_checkpoint_cadence_rejects_ambiguous_json(
    tmp_path,
    trainer_state: str,
) -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=1,
        steps_per_generation=2,
        num_iterations=1,
        save_strategy="no",
    )
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        trainer_state,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="readable trainer_state|duplicate|non-finite"):
        validate_grpo_checkpoint_cadence(
            args,
            epoch_microsteps=4,
            resume_checkpoint=checkpoint,
        )


def test_grpo_checkpoint_cadence_accounts_for_gradient_accumulation() -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=2,
        steps_per_generation=3,
        num_iterations=2,
        save_strategy="no",
        save_steps=1,
    )

    # Generate every six local training steps; two microsteps are consumed by
    # each optimizer step, so only every third checkpoint is exactly resumable.
    assert resolve_grpo_checkpoint_step_cadence(args) == 3
    validate_grpo_checkpoint_cadence(args)


def test_grpo_checkpoint_cadence_proves_epoch_save_boundary() -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=1,
        steps_per_generation=2,
        num_iterations=1,
        save_strategy="epoch",
        save_steps=1,
        max_steps=-1,
        num_train_epochs=1.0,
        use_vllm=False,
    )

    # A complete grouped epoch always contains a whole number of generation
    # reuse cycles, even when not every optimizer boundary is safe.
    validate_grpo_checkpoint_cadence(args, epoch_microsteps=4)

    # A step-bounded run can stop and trigger its epoch save inside the first
    # grouped epoch; that partial boundary must still be rejected.
    args.max_steps = 1
    with pytest.raises(ValueError, match="partial epoch.*generation-reuse cycle"):
        validate_grpo_checkpoint_cadence(args, epoch_microsteps=4)


def test_grpo_checkpoint_cadence_accounts_for_short_epoch_accumulation() -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=2,
        steps_per_generation=3,
        num_iterations=1,
        save_strategy="steps",
        save_steps=3,
        max_steps=4,
        num_train_epochs=1.0,
        use_vllm=False,
    )

    # Epoch 0 consumes three microsteps in two optimizer updates. At global
    # step 3, epoch 1 has consumed two more microsteps: 5 % 3 == 2. The old
    # global_step * GA formula incorrectly accepted this checkpoint.
    with pytest.raises(ValueError, match="global_step=3.*microstep=5.*phase=2"):
        validate_grpo_checkpoint_cadence(args, epoch_microsteps=3)


def test_grpo_checkpoint_cadence_checks_later_save_targets_across_epochs() -> None:
    args = SimpleNamespace(
        gradient_accumulation_steps=3,
        steps_per_generation=2,
        num_iterations=1,
        save_strategy="steps",
        save_steps=2,
        max_steps=4,
        num_train_epochs=1.0,
        use_vllm=False,
    )

    # Eight microsteps form three optimizer updates per epoch. The first save
    # target is safe at microstep 6, but the second crosses the shortened epoch
    # tail and lands at microstep 11, inside the next two-step rollout cycle.
    with pytest.raises(ValueError, match="global_step=4.*microstep=11.*phase=1"):
        validate_grpo_checkpoint_cadence(args, epoch_microsteps=8)


def test_grpo_vllm_sampled_rollout_rejects_checkpoint_and_resume() -> None:
    args = SimpleNamespace(use_vllm=True, save_strategy="steps")
    with pytest.raises(ValueError, match="vLLM.*RNG state"):
        validate_grpo_rollout_checkpointability(args)

    args.save_strategy = "no"
    validate_grpo_rollout_checkpointability(args)
    with pytest.raises(ValueError, match="vLLM.*RNG state"):
        validate_grpo_rollout_checkpointability(args, resume_requested=True)


def _patch_grpo_dependency_metadata(
    monkeypatch: pytest.MonkeyPatch,
    *,
    requirements: list[str] | None,
    trl_version: str = "0.29.1",
    vllm_version: str | None = "0.12.1",
) -> None:
    monkeypatch.setattr(
        rlhf_utils_module.metadata,
        "requires",
        lambda distribution: requirements if distribution == "trl" else None,
    )

    def resolve_version(distribution: str) -> str:
        if distribution == "trl":
            return trl_version
        if distribution == "vllm" and vllm_version is not None:
            return vllm_version
        raise rlhf_utils_module.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(rlhf_utils_module.metadata, "version", resolve_version)


def test_grpo_vllm_compatibility_bypasses_metadata_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rlhf_utils_module.metadata,
        "requires",
        lambda _distribution: pytest.fail("disabled vLLM should not inspect metadata"),
    )
    monkeypatch.setattr(
        rlhf_utils_module.metadata,
        "version",
        lambda _distribution: pytest.fail("disabled vLLM should not inspect versions"),
    )

    validate_grpo_vllm_runtime_compatibility(SimpleNamespace(use_vllm=False))


def test_grpo_vllm_compatibility_rejects_non_boolean_enable_flag() -> None:
    with pytest.raises(ValueError, match=r"use_vllm must be a boolean.*'false'"):
        validate_grpo_vllm_runtime_compatibility(SimpleNamespace(use_vllm="false"))


def test_grpo_vllm_compatibility_accepts_trl_extra_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_grpo_dependency_metadata(
        monkeypatch,
        requirements=['vllm<0.13.0,>=0.10.2; extra == "vllm"'],
    )

    validate_grpo_vllm_runtime_compatibility(SimpleNamespace(use_vllm=True))


def test_grpo_vllm_compatibility_respects_active_platform_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_grpo_dependency_metadata(
        monkeypatch,
        requirements=[
            ('vllm<0.13.0,>=0.10.2; python_version >= "3.0" and extra == "vllm"'),
            ('vllm<0.9; python_version < "3.0" and extra == "vllm"'),
        ],
    )

    validate_grpo_vllm_runtime_compatibility(SimpleNamespace(use_vllm=True))


@pytest.mark.parametrize(
    ("requirements", "expected_spec"),
    [
        (None, "<missing>"),
        (["requests; extra == 'vllm'"], "<missing>"),
        (["vllm; extra == 'vllm'"], "<ambiguous:<missing>>"),
        (
            [
                "vllm<0.13; extra == 'vllm'",
                "vllm>=0.10; extra == 'vllm'",
            ],
            "<ambiguous:<0.13,>=0.10>",
        ),
    ],
)
def test_grpo_vllm_compatibility_rejects_missing_or_ambiguous_spec(
    monkeypatch: pytest.MonkeyPatch,
    requirements: list[str] | None,
    expected_spec: str,
) -> None:
    _patch_grpo_dependency_metadata(monkeypatch, requirements=requirements)

    with pytest.raises(ValueError) as exc_info:
        validate_grpo_vllm_runtime_compatibility(SimpleNamespace(use_vllm=True))

    message = str(exc_info.value)
    assert "trl_version=0.29.1" in message
    assert f"required_vllm_spec={expected_spec}" in message
    assert "installed_vllm_version=0.12.1" in message


def test_grpo_vllm_compatibility_rejects_malformed_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_requirement = "vllm>=???; extra == 'vllm'"
    _patch_grpo_dependency_metadata(
        monkeypatch,
        requirements=[raw_requirement],
    )

    with pytest.raises(ValueError) as exc_info:
        validate_grpo_vllm_runtime_compatibility(SimpleNamespace(use_vllm=True))

    message = str(exc_info.value)
    assert "required_vllm_spec=<malformed:" in message
    assert "installed_vllm_version=0.12.1" in message


@pytest.mark.parametrize(
    ("vllm_version", "expected_reason"),
    [
        (None, "not installed"),
        ("not-a-version", "malformed version"),
        ("0.19.1", "compatibility window"),
    ],
)
def test_grpo_vllm_compatibility_rejects_invalid_runtime_version(
    monkeypatch: pytest.MonkeyPatch,
    vllm_version: str | None,
    expected_reason: str,
) -> None:
    _patch_grpo_dependency_metadata(
        monkeypatch,
        requirements=['vllm<0.13.0,>=0.10.2; extra == "vllm"'],
        vllm_version=vllm_version,
    )

    with pytest.raises(ValueError) as exc_info:
        validate_grpo_vllm_runtime_compatibility(SimpleNamespace(use_vllm=True))

    message = str(exc_info.value)
    assert expected_reason in message
    assert "required_vllm_spec=<0.13.0,>=0.10.2" in message
    assert f"installed_vllm_version={vllm_version or '<missing>'}" in message


def test_grpo_boundary_resume_matches_next_generation_batch(tmp_path) -> None:
    plan = ShaftSamplePlan(
        {"ds": 4},
        {"ds": 1.0},
        strategy="concat",
        shuffle=False,
        seed=3,
    )
    contract = ShaftGroupedSampleContract(
        mini_repeat_count=2,
        batch_size=1,
        iteration_count=1,
        steps_per_iteration=2,
    )

    class _RefDataset:
        def __len__(self):
            return len(plan)

        def __getitem__(self, ref):
            return ref

    def build_loader(sampler):
        return DataLoader(
            _RefDataset(),
            batch_size=2,
            sampler=sampler,
            collate_fn=lambda refs: {
                "draw_ids": torch.tensor(
                    [ref.context.draw_id for ref in refs],
                    dtype=torch.long,
                ),
                "plan_cycles": tuple(ref.context.plan_cycle for ref in refs),
            },
        )

    def build_trainer(generation_calls):
        trainer = object.__new__(ShaftGRPOTrainer)
        trainer.model = SimpleNamespace(training=True)
        trainer.args = SimpleNamespace(steps_per_generation=2)
        trainer.num_iterations = 1
        trainer._step = 0
        trainer._buffered_inputs = None

        def generate(batch):
            generation_calls.append(
                (tuple(batch["draw_ids"].tolist()), tuple(batch["plan_cycles"]))
            )
            return {"x": batch["draw_ids"].clone()}

        trainer._generate_and_score_completions = generate
        return trainer

    prepare_inputs = ShaftGRPOTrainer._prepare_inputs.__wrapped__
    uninterrupted_calls = []
    uninterrupted_trainer = build_trainer(uninterrupted_calls)
    uninterrupted_iterator = iter(build_loader(ShaftGroupedSampleSampler(plan, contract=contract)))
    _ = prepare_inputs(uninterrupted_trainer, next(uninterrupted_iterator))
    uninterrupted_trainer._step += 1
    _ = prepare_inputs(uninterrupted_trainer, next(uninterrupted_iterator))
    uninterrupted_trainer._step += 1
    checkpoint_rng = torch.get_rng_state().clone()
    uninterrupted_next = prepare_inputs(
        uninterrupted_trainer,
        next(uninterrupted_iterator),
    )

    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 2}',
        encoding="utf-8",
    )
    cadence_args = SimpleNamespace(
        gradient_accumulation_steps=1,
        steps_per_generation=2,
        num_iterations=1,
        save_strategy="no",
        save_steps=1,
    )
    validate_grpo_checkpoint_cadence(
        cadence_args,
        epoch_microsteps=8,
        resume_checkpoint=checkpoint,
    )

    resumed_calls = []
    resumed_trainer = build_trainer(resumed_calls)
    resumed_sampler = ShaftGroupedSampleSampler(plan, contract=contract)
    resumed_loader = skip_first_batches(build_loader(resumed_sampler), 2)
    resumed_sampler.set_epoch(0)
    torch.set_rng_state(checkpoint_rng)
    resumed_next = prepare_inputs(resumed_trainer, next(iter(resumed_loader)))

    assert uninterrupted_calls[-1] == resumed_calls[0] == ((1, 1), (0, 0))
    assert torch.equal(uninterrupted_next["x"], resumed_next["x"])


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
    assert grpo_args.shuffle_dataset is False
    assert resolve_grpo_grouped_sample_contract(grpo_args) == ShaftGroupedSampleContract(
        mini_repeat_count=4,
        batch_size=1,
        iteration_count=1,
        steps_per_iteration=4,
    )


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
    assert ShaftDPOTrainer.requires_equal_rank_train_batch_cardinality is True
    assert ShaftGRPOTrainer.requires_equal_rank_train_batch_cardinality is False


def test_dpo_and_grpo_share_checkpoint_commit_protocol_but_ppo_does_not() -> None:
    assert issubclass(ShaftDPOTrainer, ShaftCheckpointCommitMixin)
    assert issubclass(ShaftGRPOTrainer, ShaftCheckpointCommitMixin)
    assert not issubclass(ShaftPPOTrainer, ShaftCheckpointCommitMixin)


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
