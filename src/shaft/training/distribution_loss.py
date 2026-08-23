from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from shaft.plugins import Registry


DISTRIBUTION_DIVERGENCES = frozenset({"forward_kl", "reverse_kl", "jsd"})
_DIVERGENCES = DISTRIBUTION_DIVERGENCES
_DISTRIBUTION_MODE_ALIASES = {"dense_logits": "full_vocab"}


def _probability_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    if tensor.dtype.is_floating_point:
        return tensor.dtype
    raise TypeError("Distribution logits must use a floating-point dtype.")


@dataclass(frozen=True, slots=True)
class DistributionLossComponents:
    numerator: torch.Tensor
    denominator: torch.Tensor


@dataclass(frozen=True, slots=True)
class TeacherDistribution:
    """Canonical teacher distribution over flattened causal completion positions."""

    kind: str
    vocab_size: int
    dense_logits: torch.Tensor | None = None
    topk_token_ids: torch.Tensor | None = None
    topk_log_probs: torch.Tensor | None = None
    tail_log_probs: torch.Tensor | None = None
    temperature: float | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        object.__setattr__(self, "kind", kind)
        if kind not in {"dense_logits", "topk_tail"}:
            raise ValueError(f"Unsupported teacher distribution kind={kind!r}.")
        if type(self.vocab_size) is not int or self.vocab_size <= 0:
            raise ValueError("Teacher distribution vocab_size must be > 0.")
        if kind == "dense_logits":
            self._validate_dense()
        else:
            self._validate_topk()

    @property
    def num_positions(self) -> int:
        tensor = self.dense_logits if self.kind == "dense_logits" else self.topk_token_ids
        assert tensor is not None
        return int(tensor.shape[0])

    @property
    def top_k(self) -> int | None:
        if self.topk_token_ids is None:
            return None
        return int(self.topk_token_ids.shape[1])

    def _validate_dense(self) -> None:
        if self.dense_logits is None:
            raise ValueError("dense_logits teacher distribution requires dense_logits.")
        if self.dense_logits.ndim != 2:
            raise ValueError("Dense teacher logits must have shape [positions, vocabulary].")
        if int(self.dense_logits.shape[1]) != self.vocab_size:
            raise ValueError("Dense teacher logits disagree with vocab_size.")
        if not self.dense_logits.dtype.is_floating_point:
            raise TypeError("Dense teacher logits must be floating point.")
        if any(
            value is not None
            for value in (self.topk_token_ids, self.topk_log_probs, self.tail_log_probs)
        ):
            raise ValueError("Dense teacher distribution cannot carry top-k fields.")
        if self.temperature is not None:
            raise ValueError("Dense teacher distribution must not pre-bind temperature.")

    def _validate_topk(self) -> None:
        if self.dense_logits is not None:
            raise ValueError("topk_tail teacher distribution cannot carry dense_logits.")
        if self.topk_token_ids is None or self.topk_log_probs is None:
            raise ValueError("topk_tail teacher distribution requires token IDs and log probabilities.")
        if self.topk_token_ids.ndim != 2 or self.topk_log_probs.ndim != 2:
            raise ValueError("Top-k fields must have shape [positions, k].")
        if tuple(self.topk_token_ids.shape) != tuple(self.topk_log_probs.shape):
            raise ValueError("Top-k token IDs and log probabilities must have identical shapes.")
        if self.topk_token_ids.dtype != torch.long:
            raise TypeError("Top-k token IDs must use torch.long.")
        if not self.topk_log_probs.dtype.is_floating_point:
            raise TypeError("Top-k log probabilities must be floating point.")
        top_k = int(self.topk_token_ids.shape[1])
        if top_k <= 0 or top_k > self.vocab_size:
            raise ValueError("Teacher top-k width must be in [1, vocab_size].")
        if self.topk_token_ids.numel() and (
            int(self.topk_token_ids.min().item()) < 0
            or int(self.topk_token_ids.max().item()) >= self.vocab_size
        ):
            raise ValueError("Top-k teacher token ID is outside vocabulary.")
        if top_k > 1:
            sorted_ids = self.topk_token_ids.sort(dim=-1).values
            if bool(sorted_ids[:, 1:].eq(sorted_ids[:, :-1]).any().item()):
                raise ValueError("Top-k teacher token IDs must be unique within each row.")
        temperature = float(self.temperature or 0.0)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Top-k teacher distribution requires a finite temperature > 0.")
        if top_k == self.vocab_size:
            if self.tail_log_probs is not None:
                raise ValueError("Full-vocabulary top-k distribution must not carry a tail bucket.")
        else:
            if self.tail_log_probs is None:
                raise ValueError("Truncated top-k distribution requires tail_log_probs.")
            if tuple(self.tail_log_probs.shape) != (int(self.topk_token_ids.shape[0]),):
                raise ValueError("Teacher tail_log_probs must have shape [positions].")
            if not self.tail_log_probs.dtype.is_floating_point:
                raise TypeError("Teacher tail_log_probs must be floating point.")
        parts = [self.topk_log_probs]
        if self.tail_log_probs is not None:
            parts.append(self.tail_log_probs.unsqueeze(-1))
        log_probs = torch.cat(parts, dim=-1)
        if not bool(torch.isfinite(log_probs).all().item()):
            raise ValueError("Top-k teacher distribution contains non-finite log probabilities.")
        total = log_probs.exp().sum(dim=-1)
        tolerance = 5e-5 if log_probs.dtype == torch.float32 else 1e-10
        if not torch.allclose(total, torch.ones_like(total), atol=tolerance, rtol=tolerance):
            raise ValueError("Top-k teacher distribution is not normalized.")

    @classmethod
    def from_dense_logits(cls, logits: torch.Tensor) -> "TeacherDistribution":
        if logits.ndim != 2:
            raise ValueError("Flattened dense teacher logits must be 2-D.")
        return cls(
            kind="dense_logits",
            vocab_size=int(logits.shape[-1]),
            dense_logits=logits.detach(),
        )

    @classmethod
    def from_topk_logits(
        cls,
        logits: torch.Tensor,
        *,
        top_k: int,
        temperature: float,
    ) -> "TeacherDistribution":
        if logits.ndim != 2:
            raise ValueError("Flattened teacher logits must be 2-D for top-k projection.")
        top_k = int(top_k)
        vocab_size = int(logits.shape[-1])
        if top_k <= 0 or top_k > vocab_size:
            raise ValueError("Teacher top_k must be in [1, vocab_size].")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Teacher top-k temperature must be finite and > 0.")
        dtype = _probability_dtype(logits)
        scaled = logits.detach().to(dtype=dtype) / temperature
        log_normalizer = torch.logsumexp(scaled, dim=-1, keepdim=True)
        top_values, top_ids = torch.topk(scaled, k=top_k, dim=-1, sorted=True)
        top_log_probs = top_values - log_normalizer
        tail_log_probs = None
        if top_k < vocab_size:
            top_mass = top_log_probs.exp().sum(dim=-1)
            epsilon = torch.finfo(top_mass.dtype).eps
            tail_log_probs = torch.log1p(-top_mass.clamp(max=1.0 - epsilon))
        return cls(
            kind="topk_tail",
            vocab_size=vocab_size,
            topk_token_ids=top_ids.to(dtype=torch.long),
            topk_log_probs=top_log_probs,
            tail_log_probs=tail_log_probs,
            temperature=temperature,
        )


@dataclass(frozen=True, slots=True)
class DistributionObjectivePlan:
    mode: str
    backend_type: type["DistributionObjectiveBackend"]
    divergence: str
    temperature: float
    top_k: int | None
    token_chunk_size: int | None

    def build(self) -> "DistributionObjectiveBackend":
        return self.backend_type(self)

    def build_teacher_distribution(
        self,
        teacher_logits: torch.Tensor,
    ) -> TeacherDistribution:
        return self.backend_type.build_teacher_distribution(teacher_logits, plan=self)

    def validate_teacher_distribution(
        self,
        distribution: TeacherDistribution,
    ) -> None:
        self.backend_type.validate_teacher_distribution(distribution, plan=self)


class DistributionObjectiveBackend(ABC):
    name: str

    def __init__(self, plan: DistributionObjectivePlan) -> None:
        if str(self.name).strip().lower() != plan.mode:
            raise ValueError("Distribution objective implementation name does not match its plan.")
        self.plan = plan

    @abstractmethod
    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_distribution: TeacherDistribution,
    ) -> DistributionLossComponents:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_teacher_distribution(
        cls,
        teacher_logits: torch.Tensor,
        *,
        plan: DistributionObjectivePlan,
    ) -> TeacherDistribution:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def validate_teacher_distribution(
        cls,
        distribution: TeacherDistribution,
        *,
        plan: DistributionObjectivePlan,
    ) -> None:
        raise NotImplementedError

    def _position_slices(self, count: int):
        chunk_size = self.plan.token_chunk_size or count
        for start in range(0, count, max(int(chunk_size), 1)):
            yield slice(start, min(start + int(chunk_size), count))


def _validate_flat_student_logits(
    student_logits: torch.Tensor,
    teacher_distribution: TeacherDistribution,
) -> None:
    if student_logits.ndim != 2:
        raise ValueError("Flattened student logits must have shape [positions, vocabulary].")
    if int(student_logits.shape[0]) != teacher_distribution.num_positions:
        raise ValueError("Student and teacher position counts differ.")
    if int(student_logits.shape[1]) != teacher_distribution.vocab_size:
        raise ValueError("Student and teacher vocabulary sizes differ.")
    if int(student_logits.shape[0]) <= 0:
        raise ValueError("Distribution objective requires at least one completion position.")


def _divergence_from_log_probs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    divergence: str,
) -> torch.Tensor:
    if tuple(student_log_probs.shape) != tuple(teacher_log_probs.shape):
        raise ValueError("Student/teacher log-probability shapes differ.")
    if divergence == "reverse_kl":
        student_probs = student_log_probs.exp()
        return (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    if divergence == "forward_kl":
        teacher_probs = teacher_log_probs.exp()
        return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    if divergence == "jsd":
        midpoint_log_probs = torch.logaddexp(student_log_probs, teacher_log_probs) - math.log(2.0)
        student_term = (
            student_log_probs.exp() * (student_log_probs - midpoint_log_probs)
        ).sum(dim=-1)
        teacher_term = (
            teacher_log_probs.exp() * (teacher_log_probs - midpoint_log_probs)
        ).sum(dim=-1)
        return 0.5 * (student_term + teacher_term)
    raise ValueError(f"Unsupported distribution divergence={divergence!r}.")


class FullVocabularyDistributionObjective(DistributionObjectiveBackend):
    name = "full_vocab"

    @classmethod
    def build_teacher_distribution(
        cls,
        teacher_logits: torch.Tensor,
        *,
        plan: DistributionObjectivePlan,
    ) -> TeacherDistribution:
        _ = plan
        return TeacherDistribution.from_dense_logits(teacher_logits)

    @classmethod
    def validate_teacher_distribution(
        cls,
        distribution: TeacherDistribution,
        *,
        plan: DistributionObjectivePlan,
    ) -> None:
        _ = cls, plan
        if distribution.kind != "dense_logits":
            raise ValueError("full_vocab objective requires dense teacher logits.")

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_distribution: TeacherDistribution,
    ) -> DistributionLossComponents:
        _validate_flat_student_logits(student_logits, teacher_distribution)
        self.validate_teacher_distribution(teacher_distribution, plan=self.plan)
        assert teacher_distribution.dense_logits is not None
        numerators: list[torch.Tensor] = []
        for position_slice in self._position_slices(int(student_logits.shape[0])):
            student_chunk = student_logits[position_slice]
            teacher_chunk = teacher_distribution.dense_logits[position_slice]
            dtype = _probability_dtype(student_chunk)
            student_log_probs = F.log_softmax(
                student_chunk.to(dtype=dtype) / self.plan.temperature,
                dim=-1,
            )
            teacher_log_probs = F.log_softmax(
                teacher_chunk.detach().to(device=student_chunk.device, dtype=dtype)
                / self.plan.temperature,
                dim=-1,
            )
            values = _divergence_from_log_probs(
                student_log_probs,
                teacher_log_probs,
                divergence=self.plan.divergence,
            )
            numerators.append(values.sum() * (self.plan.temperature**2))
        return DistributionLossComponents(
            numerator=torch.stack(numerators).sum(),
            denominator=torch.tensor(
                int(student_logits.shape[0]),
                dtype=torch.float32,
                device=student_logits.device,
            ),
        )


class TopKTailDistributionObjective(DistributionObjectiveBackend):
    name = "topk_tail"

    @classmethod
    def build_teacher_distribution(
        cls,
        teacher_logits: torch.Tensor,
        *,
        plan: DistributionObjectivePlan,
    ) -> TeacherDistribution:
        if plan.top_k is None:
            raise ValueError("topk_tail objective plan has no top_k.")
        return TeacherDistribution.from_topk_logits(
            teacher_logits,
            top_k=plan.top_k,
            temperature=plan.temperature,
        )

    @classmethod
    def validate_teacher_distribution(
        cls,
        distribution: TeacherDistribution,
        *,
        plan: DistributionObjectivePlan,
    ) -> None:
        _ = cls
        if distribution.kind != "topk_tail":
            raise ValueError("topk_tail objective requires a topk_tail teacher distribution.")
        if distribution.top_k != plan.top_k:
            raise ValueError("Teacher top-k width differs from the objective plan.")
        if not math.isclose(
            float(distribution.temperature or 0.0),
            plan.temperature,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Teacher top-k temperature differs from the objective plan.")

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_distribution: TeacherDistribution,
    ) -> DistributionLossComponents:
        _validate_flat_student_logits(student_logits, teacher_distribution)
        self.validate_teacher_distribution(teacher_distribution, plan=self.plan)
        assert teacher_distribution.topk_token_ids is not None
        assert teacher_distribution.topk_log_probs is not None
        numerators: list[torch.Tensor] = []
        for position_slice in self._position_slices(int(student_logits.shape[0])):
            student_chunk = student_logits[position_slice]
            dtype = _probability_dtype(student_chunk)
            student_log_probs = F.log_softmax(
                student_chunk.to(dtype=dtype) / self.plan.temperature,
                dim=-1,
            )
            token_ids = teacher_distribution.topk_token_ids[position_slice].to(
                device=student_chunk.device
            )
            selected_student = student_log_probs.gather(dim=-1, index=token_ids)
            selected_teacher = teacher_distribution.topk_log_probs[position_slice].to(
                device=student_chunk.device,
                dtype=dtype,
            )
            student_parts = [selected_student]
            teacher_parts = [selected_teacher]
            if teacher_distribution.tail_log_probs is not None:
                selected_mass = selected_student.exp().sum(dim=-1)
                epsilon = torch.finfo(selected_mass.dtype).eps
                student_tail = torch.log1p(-selected_mass.clamp(max=1.0 - epsilon))
                teacher_tail = teacher_distribution.tail_log_probs[position_slice].to(
                    device=student_chunk.device,
                    dtype=dtype,
                )
                student_parts.append(student_tail.unsqueeze(-1))
                teacher_parts.append(teacher_tail.unsqueeze(-1))
            values = _divergence_from_log_probs(
                torch.cat(student_parts, dim=-1),
                torch.cat(teacher_parts, dim=-1),
                divergence=self.plan.divergence,
            )
            numerators.append(values.sum() * (self.plan.temperature**2))
        return DistributionLossComponents(
            numerator=torch.stack(numerators).sum(),
            denominator=torch.tensor(
                int(student_logits.shape[0]),
                dtype=torch.float32,
                device=student_logits.device,
            ),
        )


class DistributionObjectiveRegistry:
    def __init__(self) -> None:
        self._implementations: Registry[type[DistributionObjectiveBackend]] = Registry(
            "distribution_objective"
        )

    def register(
        self,
        name: str,
        implementation: type[DistributionObjectiveBackend],
    ) -> type[DistributionObjectiveBackend]:
        if not issubclass(implementation, DistributionObjectiveBackend):
            raise TypeError(
                "Distribution objective implementations must subclass "
                "DistributionObjectiveBackend."
            )
        normalized = str(name).strip().lower()
        if normalized != str(implementation.name).strip().lower():
            raise ValueError(
                "Distribution objective registration name differs from implementation name."
            )
        return self._implementations.register(normalized, implementation)

    def resolve(self, config: Any) -> DistributionObjectivePlan:
        configured_mode = str(config.mode).strip().lower()
        mode = _DISTRIBUTION_MODE_ALIASES.get(configured_mode, configured_mode)
        try:
            implementation = self._implementations.get(mode)
        except KeyError as exc:
            raise ValueError(
                "Unknown distribution objective mode "
                f"{mode!r}; registered={self._implementations.keys()}."
            ) from exc
        return DistributionObjectivePlan(
            mode=mode,
            backend_type=implementation,
            divergence=str(config.divergence).strip().lower(),
            temperature=float(config.temperature),
            top_k=None if config.top_k is None else int(config.top_k),
            token_chunk_size=(
                None if config.token_chunk_size is None else int(config.token_chunk_size)
            ),
        )

    def keys(self) -> list[str]:
        return self._implementations.keys()


DISTRIBUTION_OBJECTIVE_REGISTRY = DistributionObjectiveRegistry()
DISTRIBUTION_OBJECTIVE_REGISTRY.register(
    "full_vocab", FullVocabularyDistributionObjective
)
DISTRIBUTION_OBJECTIVE_REGISTRY.register("topk_tail", TopKTailDistributionObjective)


def resolve_distribution_objective_plan(config: Any) -> DistributionObjectivePlan:
    return DISTRIBUTION_OBJECTIVE_REGISTRY.resolve(config)


def distribution_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_mask: torch.Tensor,
    divergence: str,
    temperature: float = 1.0,
    token_chunk_size: int | None = None,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, DistributionLossComponents]:
    """Dense convenience entry using the canonical full-vocabulary objective backend."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Student/teacher logits must have identical shapes, got "
            f"student={tuple(student_logits.shape)}, teacher={tuple(teacher_logits.shape)}."
        )
    if student_logits.ndim != 3:
        raise ValueError("Distribution logits must have shape [batch, sequence, vocabulary].")
    if completion_mask.ndim != 2 or tuple(completion_mask.shape) != tuple(student_logits.shape[:2]):
        raise ValueError("completion_mask must match the logits batch/sequence axes.")
    if student_logits.shape[1] < 2:
        raise ValueError("Distribution loss requires at least two positions for causal shift.")
    divergence = str(divergence).strip().lower()
    if divergence not in _DIVERGENCES:
        raise ValueError(f"Unsupported distribution divergence={divergence!r}.")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Distribution temperature must be finite and > 0.")
    shifted_mask = completion_mask[:, 1:].to(device=student_logits.device, dtype=torch.bool)
    if not bool(shifted_mask.any().item()):
        raise ValueError("Batch has no valid completion tokens after causal shift.")
    flattened_student = student_logits[:, :-1, :][shifted_mask]
    flattened_teacher = teacher_logits[:, :-1, :].to(device=student_logits.device)[shifted_mask]
    plan = DistributionObjectivePlan(
        mode="full_vocab",
        backend_type=FullVocabularyDistributionObjective,
        divergence=divergence,
        temperature=temperature,
        top_k=None,
        token_chunk_size=token_chunk_size,
    )
    components = plan.build().compute(
        flattened_student,
        TeacherDistribution.from_dense_logits(flattened_teacher),
    )
    loss = components.numerator / components.denominator
    if not return_components:
        return loss
    return loss, components
