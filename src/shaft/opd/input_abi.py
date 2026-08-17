from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

from shaft.model.input_identity import artifact_semantic_value, stable_artifact_value
from shaft.model.types import ModelArtifacts, ProcessorPolicy


_INPUT_ABI_VERSION = "shaft-opd-input-abi-v1"
_DIGEST_CHARACTERS = frozenset("0123456789abcdef")
_KNOWN_TOKEN_ID_NAMES = (
    "bos_token_id",
    "eos_token_id",
    "unk_token_id",
    "sep_token_id",
    "pad_token_id",
    "cls_token_id",
    "mask_token_id",
    "additional_special_tokens_ids",
    "all_special_ids",
    "image_token_id",
    "video_token_id",
    "audio_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)
_KNOWN_PROCESSOR_TOKEN_NAMES = (
    "image_token",
    "image_token_id",
    "video_token",
    "video_token_id",
    "audio_token",
    "audio_token_id",
    "vision_start_token",
    "vision_start_token_id",
    "vision_end_token",
    "vision_end_token_id",
)


def _sha256(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_digest(value: str, *, role: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in _DIGEST_CHARACTERS for character in normalized):
        raise ValueError(f"{role} must be a SHA-256 digest.")
    return normalized


def _normalize_names(values: tuple[str, ...], *, role: str) -> tuple[str, ...]:
    normalized = tuple(sorted(str(value).strip() for value in values))
    if any(not value for value in normalized):
        raise ValueError(f"{role} must not contain empty field names.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{role} must contain unique field names.")
    return normalized


def _normalize_token_role(name: str) -> str:
    normalized = str(name).strip()
    if normalized.endswith("_ids"):
        return normalized.removesuffix("_ids")
    if normalized.endswith("_id"):
        return normalized.removesuffix("_id")
    return normalized


def _token_ids(value: Any, *, vocabulary: Mapping[str, int], role: str) -> tuple[int, ...]:
    raw_values = value if isinstance(value, (list, tuple)) else (value,)
    resolved: list[int] = []
    for raw_value in raw_values:
        if type(raw_value) is int:
            token_id = raw_value
        else:
            token = str(raw_value)
            if token not in vocabulary:
                raise ValueError(
                    f"OPD input ABI cannot resolve {role} token {token!r} in tokenizer.get_vocab()."
                )
            token_id = vocabulary[token]
        if token_id < 0:
            raise ValueError(f"OPD input ABI {role} contains a negative token ID.")
        resolved.append(token_id)
    return tuple(resolved)


def _record_token_role(
    output: dict[str, tuple[int, ...]],
    *,
    name: str,
    value: Any,
    vocabulary: Mapping[str, int],
) -> None:
    if value is None:
        return
    role = _normalize_token_role(name)
    token_ids = _token_ids(value, vocabulary=vocabulary, role=role)
    previous = output.setdefault(role, token_ids)
    if previous != token_ids:
        raise ValueError(
            f"OPD input ABI observes inconsistent {role} IDs: {previous!r} != {token_ids!r}."
        )


def _tokenizer_contract(
    tokenizer: Any,
) -> tuple[str, dict[str, int], tuple[tuple[str, tuple[int, ...]], ...]]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError(
            "OPD input ABI requires tokenizer.get_vocab() to validate the complete token-to-ID mapping."
        )
    vocabulary = get_vocab()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise ValueError("OPD input ABI requires tokenizer.get_vocab() to return a non-empty mapping.")
    normalized_vocab: dict[str, int] = {}
    for token, token_id in vocabulary.items():
        if type(token) is not str or type(token_id) is not int or token_id < 0:
            raise TypeError(
                "OPD input ABI tokenizer.get_vocab() entries must be string-to-nonnegative-int."
            )
        normalized_vocab[token] = token_id
    token_to_id_fingerprint = _sha256(sorted(normalized_vocab.items()))

    special_token_ids: dict[str, tuple[int, ...]] = {}
    special_tokens_map = getattr(tokenizer, "special_tokens_map", {})
    if special_tokens_map is not None:
        if not isinstance(special_tokens_map, Mapping):
            raise TypeError("OPD input ABI tokenizer.special_tokens_map must be a mapping.")
        for name, value in special_tokens_map.items():
            _record_token_role(
                special_token_ids,
                name=str(name),
                value=value,
                vocabulary=normalized_vocab,
            )

    dynamic_names = {
        name
        for name in vars(tokenizer)
        if name.endswith(("_token_id", "_token_ids"))
    }
    for name in (*_KNOWN_TOKEN_ID_NAMES, *sorted(dynamic_names)):
        if hasattr(tokenizer, name):
            _record_token_role(
                special_token_ids,
                name=name,
                value=getattr(tokenizer, name),
                vocabulary=normalized_vocab,
            )
    if not special_token_ids:
        raise ValueError("OPD input ABI requires the tokenizer to publish special token IDs.")
    return (
        token_to_id_fingerprint,
        normalized_vocab,
        tuple(sorted(special_token_ids.items())),
    )


def _processor_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_name, item in value.items():
            name = str(raw_name)
            normalized = name.strip().lower()
            if (
                "chat_template" in normalized
                or normalized == "processor_class"
                or normalized.endswith("_processor_type")
                or normalized.endswith(("_token", "_token_id", "_token_ids"))
            ):
                continue
            output[name] = _processor_config_value(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_processor_config_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("OPD processor ABI config cannot contain non-finite floats.")
        return value
    if isinstance(value, Path):
        return str(value)
    return stable_artifact_value(value)


def _component_config(component: Any) -> Any:
    to_dict = getattr(component, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        payload = to_dict()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"OPD processor ABI cannot serialize {type(component).__name__}.to_dict()."
        ) from exc
    if not isinstance(payload, Mapping):
        raise TypeError("OPD processor ABI component.to_dict() must return a mapping.")
    return _processor_config_value(artifact_semantic_value(payload))


def _processor_token_roles(
    processor: Any,
    *,
    vocabulary: Mapping[str, int],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    names = set(_KNOWN_PROCESSOR_TOKEN_NAMES)
    names.update(
        name
        for name in vars(processor)
        if name.endswith(("_token", "_token_id", "_token_ids"))
    )
    to_dict = getattr(processor, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except (AttributeError, TypeError, ValueError):
            payload = None
        if isinstance(payload, Mapping):
            names.update(
                str(name)
                for name in payload
                if str(name).endswith(("_token", "_token_id", "_token_ids"))
            )
    roles: dict[str, tuple[int, ...]] = {}
    for name in sorted(names):
        if hasattr(processor, name):
            _record_token_role(
                roles,
                name=name,
                value=getattr(processor, name),
                vocabulary=vocabulary,
            )
    return tuple(sorted(roles.items()))


def _policy_payload(policy: ProcessorPolicy) -> dict[str, Any]:
    return {
        "sample_aligned": list(policy.sample_aligned_model_input_names),
        "whole_batch": list(policy.whole_batch_model_input_names),
        "static": list(policy.static_model_input_names),
        "sequence": [
            {
                "name": field.name,
                "sequence_axis": field.sequence_axis,
                "padding_value": field.padding_value,
                "continuation_value": field.continuation_value,
            }
            for field in policy.processor_sequence_fields
        ],
        "rollout_tail_logits_input_name": policy.rollout_tail_logits_input_name,
    }


def _processor_contract(
    artifacts: ModelArtifacts,
    *,
    vocabulary: Mapping[str, int],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    processor = artifacts.processor
    policy = artifacts.model_adapter.processor_policy
    required_names = {"input_ids", "attention_mask", "use_cache"}
    if policy.rollout_tail_logits_input_name is not None:
        required_names.add(policy.rollout_tail_logits_input_name)
    optional_names = {
        *policy.sample_aligned_model_input_names,
        *policy.whole_batch_model_input_names,
        *policy.static_model_input_names,
        *(field.name for field in policy.processor_sequence_fields),
    } - required_names
    components: dict[str, Any] = {
        "processor_config": _component_config(processor),
        "processor_token_roles": _processor_token_roles(
            processor,
            vocabulary=vocabulary,
        ),
        "policy": _policy_payload(policy),
    }
    for name in ("image_processor", "video_processor"):
        component = getattr(processor, name, None)
        if component is None:
            continue
        components[name] = {
            "config": _component_config(component),
        }
    return (
        _sha256({"version": "shaft-opd-processor-abi-v1", **components}),
        tuple(sorted(required_names)),
        tuple(sorted(optional_names)),
    )


def _logits_vocab_size(model: Any) -> int:
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    output = get_output_embeddings() if callable(get_output_embeddings) else None
    if output is None:
        output = getattr(model, "lm_head", None)
    if output is None:
        raise ValueError(
            "OPD input ABI cannot resolve the model output head for logits vocabulary validation."
        )
    candidates: set[int] = set()
    weight = getattr(output, "weight", None)
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 1:
        candidates.add(int(weight.shape[0]))
    for name in ("out_features", "num_embeddings"):
        value = getattr(output, name, None)
        if value is not None:
            candidates.add(int(value))
    if len(candidates) != 1 or next(iter(candidates), 0) <= 0:
        raise ValueError(
            "OPD input ABI cannot prove one unambiguous logits vocabulary dimension; "
            f"observed={sorted(candidates)}."
        )
    return next(iter(candidates))


def _forward_contract(model: Any) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    forward = getattr(model, "forward", None)
    if not callable(forward):
        raise TypeError("OPD input ABI requires a callable model.forward.")
    try:
        parameters = inspect.signature(forward).parameters.values()
    except (TypeError, ValueError) as exc:
        raise ValueError("OPD input ABI cannot inspect model.forward input fields.") from exc
    accepted: set[str] = set()
    required: set[str] = set()
    accepts_kwargs = False
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
            continue
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            continue
        if parameter.name == "self":
            continue
        accepted.add(parameter.name)
        if parameter.default is inspect.Parameter.empty:
            required.add(parameter.name)
    return tuple(sorted(accepted)), tuple(sorted(required)), accepts_kwargs


@dataclass(frozen=True, slots=True)
class ShaftOPDInputABI:
    """Versioned contract for tensors shared by one OPD student and teacher."""

    token_to_id_fingerprint: str
    token_count: int
    special_token_ids: tuple[tuple[str, tuple[int, ...]], ...]
    logits_vocab_size: int
    processor_abi_fingerprint: str
    required_model_input_names: tuple[str, ...]
    optional_model_input_names: tuple[str, ...]
    forward_accepted_input_names: tuple[str, ...]
    forward_required_input_names: tuple[str, ...]
    forward_accepts_kwargs: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "token_to_id_fingerprint",
            _validate_digest(
                self.token_to_id_fingerprint,
                role="ShaftOPDInputABI.token_to_id_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "processor_abi_fingerprint",
            _validate_digest(
                self.processor_abi_fingerprint,
                role="ShaftOPDInputABI.processor_abi_fingerprint",
            ),
        )
        for name in ("token_count", "logits_vocab_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"ShaftOPDInputABI.{name} must be > 0.")
        special = tuple(sorted(self.special_token_ids))
        if not special or len({name for name, _ in special}) != len(special):
            raise ValueError("ShaftOPDInputABI.special_token_ids must be non-empty and unique.")
        for name, values in special:
            if not str(name).strip() or not values or any(type(value) is not int or value < 0 for value in values):
                raise ValueError("ShaftOPDInputABI.special_token_ids entries are invalid.")
        object.__setattr__(self, "special_token_ids", special)
        for name in (
            "required_model_input_names",
            "optional_model_input_names",
            "forward_accepted_input_names",
            "forward_required_input_names",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_names(getattr(self, name), role=f"ShaftOPDInputABI.{name}"),
            )
        if set(self.required_model_input_names) & set(self.optional_model_input_names):
            raise ValueError("OPD required and optional model input names must not overlap.")
        if not isinstance(self.forward_accepts_kwargs, bool):
            raise TypeError("ShaftOPDInputABI.forward_accepts_kwargs must be a boolean.")
        if not set(self.forward_required_input_names).issubset(
            self.forward_accepted_input_names
        ):
            raise ValueError("OPD required forward fields must also be explicitly accepted.")

    @property
    def fingerprint(self) -> str:
        return _sha256(
            {
                "version": _INPUT_ABI_VERSION,
                "token_to_id_fingerprint": self.token_to_id_fingerprint,
                "token_count": self.token_count,
                "special_token_ids": dict(self.special_token_ids),
                "logits_vocab_size": self.logits_vocab_size,
                "processor_abi_fingerprint": self.processor_abi_fingerprint,
                "required_model_input_names": self.required_model_input_names,
                "optional_model_input_names": self.optional_model_input_names,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _INPUT_ABI_VERSION,
            "token_to_id_fingerprint": self.token_to_id_fingerprint,
            "token_count": self.token_count,
            "special_token_ids": {name: list(values) for name, values in self.special_token_ids},
            "logits_vocab_size": self.logits_vocab_size,
            "processor_abi_fingerprint": self.processor_abi_fingerprint,
            "required_model_input_names": list(self.required_model_input_names),
            "optional_model_input_names": list(self.optional_model_input_names),
            "forward_accepted_input_names": list(self.forward_accepted_input_names),
            "forward_required_input_names": list(self.forward_required_input_names),
            "forward_accepts_kwargs": self.forward_accepts_kwargs,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ShaftOPDInputABI:
        expected = {
            "version",
            "token_to_id_fingerprint",
            "token_count",
            "special_token_ids",
            "logits_vocab_size",
            "processor_abi_fingerprint",
            "required_model_input_names",
            "optional_model_input_names",
            "forward_accepted_input_names",
            "forward_required_input_names",
            "forward_accepts_kwargs",
        }
        if set(payload) != expected:
            raise ValueError(
                "OPD input ABI fields differ from the protocol; "
                f"missing={sorted(expected - set(payload))} extra={sorted(set(payload) - expected)}."
            )
        if payload["version"] != _INPUT_ABI_VERSION:
            raise ValueError(f"Unsupported OPD input ABI version: {payload['version']!r}.")
        for name in ("token_to_id_fingerprint", "processor_abi_fingerprint"):
            if type(payload[name]) is not str:
                raise TypeError(f"OPD input ABI {name} must be a string.")
        for name in ("token_count", "logits_vocab_size"):
            if type(payload[name]) is not int:
                raise TypeError(f"OPD input ABI {name} must be an integer.")
        if type(payload["forward_accepts_kwargs"]) is not bool:
            raise TypeError("OPD input ABI forward_accepts_kwargs must be a boolean.")
        special = payload["special_token_ids"]
        if not isinstance(special, Mapping):
            raise TypeError("OPD input ABI special_token_ids must be an object.")
        if not all(
            type(name) is str
            and isinstance(values, list)
            and values
            and all(type(value) is int for value in values)
            for name, values in special.items()
        ):
            raise TypeError("OPD input ABI special_token_ids entries must be integer lists.")

        def _name_list(name: str) -> tuple[str, ...]:
            value = payload[name]
            if not isinstance(value, list) or not all(type(item) is str for item in value):
                raise TypeError(f"OPD input ABI {name} must be a string list.")
            return tuple(value)

        return cls(
            token_to_id_fingerprint=str(payload["token_to_id_fingerprint"]),
            token_count=int(payload["token_count"]),
            special_token_ids=tuple(
                (str(name), tuple(values))
                for name, values in special.items()
            ),
            logits_vocab_size=int(payload["logits_vocab_size"]),
            processor_abi_fingerprint=str(payload["processor_abi_fingerprint"]),
            required_model_input_names=_name_list("required_model_input_names"),
            optional_model_input_names=_name_list("optional_model_input_names"),
            forward_accepted_input_names=_name_list("forward_accepted_input_names"),
            forward_required_input_names=_name_list("forward_required_input_names"),
            forward_accepts_kwargs=payload["forward_accepts_kwargs"],
        )


def build_opd_input_abi(artifacts: ModelArtifacts) -> ShaftOPDInputABI:
    token_fingerprint, vocabulary, special_token_ids = _tokenizer_contract(
        artifacts.tokenizer
    )
    processor_fingerprint, required_names, optional_names = _processor_contract(
        artifacts,
        vocabulary=vocabulary,
    )
    accepted_names, forward_required_names, accepts_kwargs = _forward_contract(artifacts.model)
    logits_vocab_size = _logits_vocab_size(artifacts.model)
    max_token_id = max(vocabulary.values())
    if max_token_id >= logits_vocab_size:
        raise ValueError(
            "OPD input ABI tokenizer token IDs exceed the logits vocabulary dimension; "
            f"max_token_id={max_token_id} logits_vocab_size={logits_vocab_size}."
        )
    return ShaftOPDInputABI(
        token_to_id_fingerprint=token_fingerprint,
        token_count=len(vocabulary),
        special_token_ids=special_token_ids,
        logits_vocab_size=logits_vocab_size,
        processor_abi_fingerprint=processor_fingerprint,
        required_model_input_names=required_names,
        optional_model_input_names=optional_names,
        forward_accepted_input_names=accepted_names,
        forward_required_input_names=forward_required_names,
        forward_accepts_kwargs=accepts_kwargs,
    )


def _validate_forward_inputs(
    *,
    producer: ShaftOPDInputABI,
    consumer: ShaftOPDInputABI,
    role: str,
) -> None:
    produced = set(producer.required_model_input_names) | set(producer.optional_model_input_names)
    unsupported = (
        set() if consumer.forward_accepts_kwargs else produced - set(consumer.forward_accepted_input_names)
    )
    missing_required = set(consumer.forward_required_input_names) - set(
        producer.required_model_input_names
    )
    if unsupported or missing_required:
        raise ValueError(
            f"OPD {role} forward input fields are incompatible with student-produced tensors; "
            f"unsupported={sorted(unsupported)} missing_required={sorted(missing_required)}."
        )


def validate_opd_input_abi_compatibility(
    *,
    student: ShaftOPDInputABI,
    teacher: ShaftOPDInputABI,
) -> str:
    if student.special_token_ids != teacher.special_token_ids:
        raise ValueError(
            "OPD teacher/student special token IDs differ; "
            f"student={dict(student.special_token_ids)} teacher={dict(teacher.special_token_ids)}."
        )
    if (
        student.token_count != teacher.token_count
        or student.token_to_id_fingerprint != teacher.token_to_id_fingerprint
    ):
        raise ValueError(
            "OPD teacher/student complete token-to-ID mappings differ; "
            f"student_count={student.token_count} teacher_count={teacher.token_count}."
        )
    if student.logits_vocab_size != teacher.logits_vocab_size:
        raise ValueError(
            "OPD teacher/student logits vocabulary dimensions differ; "
            f"student={student.logits_vocab_size} teacher={teacher.logits_vocab_size}."
        )
    if (
        student.processor_abi_fingerprint != teacher.processor_abi_fingerprint
        or student.required_model_input_names != teacher.required_model_input_names
        or student.optional_model_input_names != teacher.optional_model_input_names
    ):
        raise ValueError(
            "OPD teacher/student multimodal processor/input ABI differs; "
            f"student={student.processor_abi_fingerprint} "
            f"teacher={teacher.processor_abi_fingerprint}."
        )
    _validate_forward_inputs(producer=student, consumer=student, role="student")
    _validate_forward_inputs(producer=student, consumer=teacher, role="teacher")
    return student.fingerprint
