from __future__ import annotations

from typing import Any

import torch

from shaft.training.distribution_loss import DistributionObjectivePlan, TeacherDistribution
from shaft.training.sft_trainer import ShaftSFTTrainer


class ShaftOfflineKDTrainer(ShaftSFTTrainer):
    """Teacher-forced CE plus a fixed offline teacher distribution objective."""

    def __init__(
        self,
        *args: Any,
        objective_plan: DistributionObjectivePlan,
        ce_weight: float,
        kd_weight: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.offline_kd_objective = objective_plan.build()
        self.offline_kd_ce_weight = float(ce_weight)
        self.offline_kd_weight = float(kd_weight)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        model_inputs = dict(inputs)
        completion_mask = model_inputs.pop("_shaft_offline_kd_completion_mask", None)
        teacher_distribution = model_inputs.pop(
            "_shaft_offline_kd_teacher_distribution", None
        )
        if not isinstance(completion_mask, torch.Tensor) or completion_mask.ndim != 2:
            raise TypeError("Offline KD batch is missing its 2-D completion mask.")
        if not isinstance(teacher_distribution, TeacherDistribution):
            raise TypeError("Offline KD batch is missing its teacher distribution.")
        if self.offline_kd_ce_weight > 0:
            ce_loss, outputs = super().compute_loss(
                model,
                model_inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            model_inputs.pop("labels", None)
            model_inputs.pop("loss_scale", None)
            if self.model_adapter is not None:
                model_inputs = self.model_adapter.prepare_sft_forward_inputs(
                    model=model,
                    inputs=model_inputs,
                )
            outputs = model(**model_inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            ce_loss = logits.new_zeros(())
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        if self.offline_kd_weight > 0:
            shifted_mask = completion_mask[:, 1:].to(device=logits.device, dtype=torch.bool)
            if not bool(shifted_mask.any().item()):
                raise ValueError("Offline KD batch has no causal completion positions.")
            student_logits = logits[:, :-1, :][shifted_mask]
            components = self.offline_kd_objective.compute(
                student_logits,
                teacher_distribution,
            )
            denominator = components.denominator
            if num_items_in_batch is not None:
                denominator = torch.as_tensor(
                    num_items_in_batch,
                    dtype=torch.float32,
                    device=components.numerator.device,
                )
            kd_loss = components.numerator / denominator.clamp_min(1.0)
            if num_items_in_batch is not None and self.args.average_tokens_across_devices:
                data_parallel_scale = self.accelerator.num_processes
                parallelism_config = getattr(self.accelerator, "parallelism_config", None)
                if parallelism_config is not None:
                    data_parallel_scale //= parallelism_config.tp_size
                kd_loss = kd_loss * (
                    data_parallel_scale if self.args.n_gpu <= 1 else self.args.n_gpu
                )
        else:
            kd_loss = logits.new_zeros(())
        loss = self.offline_kd_ce_weight * ce_loss + self.offline_kd_weight * kd_loss
        if return_outputs:
            return loss, outputs
        return loss
