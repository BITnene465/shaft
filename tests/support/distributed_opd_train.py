from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from shaft.config import load_config
from shaft.opd import ShaftOPDTrainer
from shaft.opd.remote_teacher import encode_teacher_distribution
from shaft.pipeline import run_opd


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        normalized = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(normalized.dtype).encode("ascii"))
        digest.update(str(tuple(normalized.shape)).encode("ascii"))
        digest.update(normalized.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: distributed_opd_train.py CONFIG PROBE_PATH")
    config = load_config(sys.argv[1])
    probe_path = Path(sys.argv[2])
    original_train = ShaftOPDTrainer.train

    def _train_with_probe(self, *args, **kwargs):
        teacher_provider = self.execution_runtime.teacher_provider
        teacher_model = getattr(teacher_provider, "model", None)
        rollout_backend = self.execution_runtime.rollout_backend
        rollout_trace = []
        teacher_trace = []
        original_generate = rollout_backend.generate
        original_score = teacher_provider.score

        def traced_generate(request):
            result = original_generate(request)
            rollout_trace.append(
                {
                    "model_version": int(request.model_version),
                    "request_ids": list(request.request_ids),
                    "digest": _tensor_digest(
                        result.sequences,
                        result.attention_mask,
                        result.completion_mask,
                    ),
                }
            )
            return result

        def traced_score(request):
            distribution = original_score(request)
            teacher_trace.append(
                {
                    "request_ids": list(request.request_ids),
                    "digest": hashlib.sha256(
                        encode_teacher_distribution(distribution)
                    ).hexdigest(),
                }
            )
            return distribution

        rollout_backend.generate = traced_generate
        teacher_provider.score = traced_score
        student_before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if not student_before:
            raise AssertionError("OPD integration probe found no trainable student parameters.")
        teacher_before = (
            {
                name: (id(parameter), int(parameter._version))
                for name, parameter in teacher_model.named_parameters()
            }
            if teacher_model is not None
            else None
        )
        try:
            result = original_train(self, *args, **kwargs)
        finally:
            rollout_backend.generate = original_generate
            teacher_provider.score = original_score
        student_after = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if student_after.keys() != student_before.keys():
            raise AssertionError("OPD trainable student parameter set changed during training.")
        student_changed = any(
            not torch.equal(student_before[name], parameter.detach())
            for name, parameter in student_after.items()
        )
        teacher_after = (
            {
                name: (id(parameter), int(parameter._version))
                for name, parameter in teacher_model.named_parameters()
            }
            if teacher_model is not None
            else None
        )
        teacher_unchanged = teacher_before == teacher_after
        student_max_abs_delta = max(
            float((student_before[name] - parameter.detach()).abs().max().item())
            for name, parameter in student_after.items()
        )
        flags = torch.tensor(
            [int(student_changed), int(teacher_unchanged)],
            device=self.args.device,
            dtype=torch.int64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        minimum_delta = torch.tensor(
            student_max_abs_delta,
            device=self.args.device,
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(minimum_delta, op=dist.ReduceOp.MIN)
            traces = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(
                traces,
                {"rollout": rollout_trace, "teacher": teacher_trace},
            )
        else:
            traces = [{"rollout": rollout_trace, "teacher": teacher_trace}]
        if not dist.is_initialized() or dist.get_rank() == 0:
            probe_path.write_text(
                json.dumps(
                    {
                        "all_ranks_student_changed": bool(flags[0].item()),
                        "all_ranks_teacher_unchanged": bool(flags[1].item()),
                        "teacher_model_loaded": teacher_model is not None,
                        "teacher_provider": str(teacher_provider.name),
                        "minimum_rank_student_max_abs_delta": float(
                            minimum_delta.item()
                        ),
                        "rank_traces": traces,
                        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return result

    ShaftOPDTrainer.train = _train_with_probe
    run_opd(config)


if __name__ == "__main__":
    main()
