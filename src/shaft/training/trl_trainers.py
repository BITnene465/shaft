from __future__ import annotations

import os
from typing import Any

from transformers import Trainer as HFTrainer

from shaft.config.training import EvalConfig
from shaft.data.mixing import ShaftSamplePlan
from shaft.data.sampler import ShaftGroupedSampleContract, ShaftGroupedSampleSampler

from .distributed import barrier_if_distributed
from .optimizer_mixin import ShaftOptimizerMixin
from .online_eval import ShaftOnlineEvalRunner
from .train_sampler_mixin import ShaftTrainSamplerMixin

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

try:
    from trl import DPOTrainer as _TRLDPOTrainer
except Exception as exc:  # noqa: BLE001
    _TRLDPOTrainer = object
    _DPO_IMPORT_ERROR = exc
else:
    _DPO_IMPORT_ERROR = None

try:
    from trl.experimental.ppo import PPOTrainer as _TRLPPOTrainer
except Exception as exc:  # noqa: BLE001
    _TRLPPOTrainer = object
    _PPO_IMPORT_ERROR = exc
else:
    _PPO_IMPORT_ERROR = None

try:
    from trl import GRPOTrainer as _TRLGRPOTrainer
except Exception as exc:  # noqa: BLE001
    _TRLGRPOTrainer = object
    _GRPO_IMPORT_ERROR = exc
else:
    _GRPO_IMPORT_ERROR = None


class ShaftDPOTrainer(ShaftOptimizerMixin, ShaftTrainSamplerMixin, _TRLDPOTrainer):
    """TRL DPOTrainer wrapper with Shaft naming."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _DPO_IMPORT_ERROR is not None:
            raise ImportError(
                "TRL DPO trainer is unavailable. Install RLHF deps: `uv pip install -e \".[rlhf]\"`."
            ) from _DPO_IMPORT_ERROR
        super().__init__(*args, **kwargs)


class ShaftPPOTrainer(ShaftOptimizerMixin, _TRLPPOTrainer):
    """TRL PPOTrainer wrapper with Shaft naming."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _PPO_IMPORT_ERROR is not None:
            raise ImportError(
                "TRL PPO trainer is unavailable. Install RLHF deps: `uv pip install -e \".[rlhf]\"`."
            ) from _PPO_IMPORT_ERROR
        super().__init__(*args, **kwargs)


class ShaftGRPOTrainer(ShaftOptimizerMixin, _TRLGRPOTrainer):
    """TRL GRPOTrainer wrapper with Shaft naming."""

    def __init__(
        self,
        *args: Any,
        sample_plan: ShaftSamplePlan | None = None,
        grouped_sample_contract: ShaftGroupedSampleContract | None = None,
        online_eval_runner: ShaftOnlineEvalRunner | None = None,
        eval_config: EvalConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if _GRPO_IMPORT_ERROR is not None:
            raise ImportError(
                "TRL GRPO trainer is unavailable. Install RLHF deps: `uv pip install -e \".[rlhf]\"`."
            ) from _GRPO_IMPORT_ERROR
        self.sample_plan = sample_plan
        self.grouped_sample_contract = grouped_sample_contract
        if (sample_plan is None) != (grouped_sample_contract is None):
            raise ValueError(
                "GRPO sample_plan and grouped_sample_contract must be provided together."
            )
        super().__init__(*args, **kwargs)
        self.online_eval_runner = online_eval_runner
        self.eval_config = eval_config

    def _get_train_sampler(self, dataset=None):
        if self.sample_plan is None:
            return super()._get_train_sampler(dataset)
        assert self.grouped_sample_contract is not None
        return ShaftGroupedSampleSampler(
            self.sample_plan,
            contract=self.grouped_sample_contract,
        )

    def prepare_online_eval_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return HFTrainer._prepare_inputs(self, inputs)

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ):
        if self.online_eval_runner is None:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        barrier_if_distributed()
        _ = ignore_keys
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        self._memory_tracker.start()
        metrics: dict[str, float] = {}
        if eval_dataset is not None:
            metrics.update(
                self.online_eval_runner.evaluate(
                    self,
                    eval_dataset=eval_dataset,
                    metric_key_prefix=metric_key_prefix,
                )
            )
        report_metrics = {key: float(value) for key, value in metrics.items()}
        self.log(report_metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, report_metrics)
        self._memory_tracker.stop_and_update_metrics(report_metrics)
        barrier_if_distributed()
        return metrics
