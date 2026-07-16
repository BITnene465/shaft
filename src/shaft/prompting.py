from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import yaml


PROMPT_RENDERER_VERSION = "shaft-prompt-renderer-v1"
PROMPT_JSON_VERSION = "shaft-prompt-json-v1"
_ARGUMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXPRESSION = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\|\s*(json)\s*)?$"
)
_ARGUMENT_TYPES = {
    "string",
    "enum",
    "integer",
    "float",
    "boolean",
    "json",
    "bbox_2d_0_999",
}


def canonical_json(value: Any) -> str:
    """Serialize a JSON value with the one stable representation used by prompts/audit."""

    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Prompt argument is not strict JSON data.") from exc


def validate_prompt_text(value: str, *, source: str = "prompt text") -> str:
    """Reject text that cannot be represented as UTF-8 before hashing or execution."""

    if not isinstance(value, str):
        raise ValueError(f"Prompt text must be a string ({source}).")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"Prompt strings must contain valid UTF-8 Unicode scalar values ({source})."
        ) from exc
    return value


def _validate_json_value(value: Any) -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        _validate_utf8(value)
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError("Prompt JSON numbers must be finite.")
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Prompt JSON object keys must be strings.")
        for key, item in value.items():
            _validate_utf8(key)
            _validate_json_value(item)
        return
    raise ValueError("Prompt argument is not strict JSON data.")


def _validate_utf8(value: str) -> None:
    validate_prompt_text(value)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ShaftPromptArgument:
    name: str
    type: str
    enum_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.enum_values, (str, bytes)):
            raise ValueError(
                f"Prompt argument {self.name!r} enum_values must be a sequence of strings."
            )
        try:
            enum_values = tuple(self.enum_values)
        except TypeError as exc:
            raise ValueError(
                f"Prompt argument {self.name!r} enum_values must be a sequence."
            ) from exc
        object.__setattr__(self, "enum_values", enum_values)
        if not isinstance(self.name, str) or not _ARGUMENT_NAME.fullmatch(self.name):
            raise ValueError(f"Invalid prompt argument name {self.name!r}.")
        if not isinstance(self.type, str) or self.type not in _ARGUMENT_TYPES:
            raise ValueError(f"Unsupported prompt argument type {self.type!r}.")
        if self.type == "enum":
            for value in self.enum_values:
                if not isinstance(value, str):
                    raise ValueError(f"Enum prompt argument {self.name!r} values must be strings.")
                _validate_utf8(value)
            if not self.enum_values or len(self.enum_values) != len(set(self.enum_values)):
                raise ValueError(f"Enum prompt argument {self.name!r} requires unique values.")
        elif self.enum_values:
            raise ValueError(f"Only enum prompt arguments may define enum_values: {self.name!r}.")

    def validate(self, value: Any, *, source: str) -> Any:
        kind = self.type
        if kind == "string":
            valid = isinstance(value, str)
        elif kind == "enum":
            valid = isinstance(value, str) and value in self.enum_values
        elif kind == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif kind == "float":
            valid = (
                (isinstance(value, int) and not isinstance(value, bool))
                or (isinstance(value, float) and math.isfinite(value))
            )
        elif kind == "boolean":
            valid = isinstance(value, bool)
        elif kind == "json":
            canonical_json(value)
            valid = True
        elif kind == "bbox_2d_0_999":
            valid = _valid_bbox(value)
        else:  # pragma: no cover - schema construction prevents this
            valid = False
        if not valid:
            raise ValueError(
                f"Prompt argument {self.name!r} must have type {kind!r} ({source})."
            )
        return value


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        return False
    x1, y1, x2, y2 = value
    return bool(
        all(0 <= item <= 999 for item in value)
        and x1 <= x2
        and y1 <= y2
    )


@dataclass(frozen=True, slots=True)
class ShaftPromptSchema:
    arguments: tuple[ShaftPromptArgument, ...] = ()

    def __post_init__(self) -> None:
        try:
            arguments = tuple(self.arguments)
        except TypeError as exc:
            raise ValueError("Prompt schema arguments must be a sequence.") from exc
        object.__setattr__(self, "arguments", arguments)
        if any(not isinstance(argument, ShaftPromptArgument) for argument in arguments):
            raise ValueError("Prompt schema arguments must be ShaftPromptArgument values.")
        names = [argument.name for argument in self.arguments]
        if len(names) != len(set(names)):
            raise ValueError("Prompt schema contains duplicate argument names.")

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "arguments": [
                {
                    "name": argument.name,
                    "type": argument.type,
                    **(
                        {"values": sorted(argument.enum_values)}
                        if argument.type == "enum"
                        else {}
                    ),
                }
                for argument in sorted(self.arguments, key=lambda item: item.name)
            ]
        }

    @property
    def fingerprint(self) -> str:
        return _sha256(canonical_json(self.canonical_payload))

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any] | None,
        *,
        source: str,
    ) -> "ShaftPromptSchema":
        raw_arguments = {} if payload is None else payload
        if not isinstance(raw_arguments, dict):
            raise ValueError(f"Prompt arguments must be a mapping ({source}).")
        arguments: list[ShaftPromptArgument] = []
        for name, raw_spec in raw_arguments.items():
            if not isinstance(name, str) or not _ARGUMENT_NAME.fullmatch(name):
                raise ValueError(f"Invalid prompt argument name {name!r} ({source}).")
            if not isinstance(raw_spec, dict):
                raise ValueError(f"Prompt argument {name!r} schema must be a mapping ({source}).")
            unknown = sorted(set(raw_spec) - {"type", "values", "required"})
            if unknown:
                raise ValueError(
                    f"Unknown schema keys for prompt argument {name!r}: {unknown} ({source})."
                )
            if raw_spec.get("required", True) is not True:
                raise ValueError(
                    f"Prompt argument {name!r} only supports required: true ({source})."
                )
            kind = str(raw_spec.get("type", "")).strip()
            if kind not in _ARGUMENT_TYPES:
                raise ValueError(
                    f"Prompt argument {name!r} has unsupported type {kind!r} ({source})."
                )
            values: tuple[str, ...] = ()
            if kind == "enum":
                raw_values = raw_spec.get("values")
                if not isinstance(raw_values, list) or not raw_values:
                    raise ValueError(
                        f"Enum prompt argument {name!r} requires non-empty values ({source})."
                    )
                if any(not isinstance(value, str) for value in raw_values):
                    raise ValueError(
                        f"Enum prompt argument {name!r} values must be strings ({source})."
                    )
                if len(raw_values) != len(set(raw_values)):
                    raise ValueError(
                        f"Enum prompt argument {name!r} contains duplicate values ({source})."
                    )
                values = tuple(raw_values)
            elif "values" in raw_spec:
                raise ValueError(
                    f"Prompt argument {name!r} only supports values for enum type ({source})."
                )
            arguments.append(ShaftPromptArgument(name=name, type=kind, enum_values=values))
        return cls(tuple(arguments))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(argument.name for argument in self.arguments)

    def validate(self, values: dict[str, Any] | None, *, source: str) -> dict[str, Any]:
        resolved = {} if values is None else values
        if not isinstance(resolved, dict):
            raise ValueError(f"prompt_args must be a JSON object ({source}).")
        expected = set(self.names)
        actual = set(resolved)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            raise ValueError(f"Missing prompt arguments {missing} ({source}).")
        if extra:
            raise ValueError(f"Unexpected prompt arguments {extra} ({source}).")
        validated: dict[str, Any] = {}
        for argument in self.arguments:
            validated[argument.name] = argument.validate(
                resolved[argument.name],
                source=source,
            )
        canonical_json(validated)
        return validated


@dataclass(frozen=True, slots=True)
class _PromptPart:
    literal: str = ""
    argument: str | None = None
    json_filter: bool = False


@dataclass(frozen=True, slots=True)
class ShaftPromptProgram:
    template: str
    schema: ShaftPromptSchema
    source: str
    parts: tuple[_PromptPart, ...]
    referenced_arguments: tuple[str, ...]
    template_sha256: str
    schema_sha256: str
    program_sha256: str
    renderer_version: str = PROMPT_RENDERER_VERSION
    json_version: str = PROMPT_JSON_VERSION

    def render(self, values: dict[str, Any] | None = None, *, context: str | None = None) -> str:
        source = context or self.source
        validated = self.schema.validate(values, source=source)
        rendered = self._render(validated)
        self._validate_rendered(rendered, source=source)
        return rendered

    def render_with_audit(
        self,
        values: dict[str, Any] | None = None,
        *,
        context: str | None = None,
    ) -> tuple[str, dict[str, str]]:
        source = context or self.source
        validated = self.schema.validate(values, source=source)
        rendered = self._render(validated)
        self._validate_rendered(rendered, source=source)
        return rendered, self._audit(validated, rendered)

    def _render(self, validated: dict[str, Any]) -> str:
        chunks: list[str] = []
        for part in self.parts:
            if part.argument is None:
                chunks.append(part.literal)
                continue
            value = validated[part.argument]
            chunks.append(canonical_json(value) if part.json_filter else str(value))
        return "".join(chunks)

    @staticmethod
    def _validate_rendered(rendered: str, *, source: str) -> None:
        if not rendered.strip():
            raise ValueError(f"Rendered prompt must not be empty ({source}).")

    def audit(self, values: dict[str, Any] | None, rendered: str) -> dict[str, str]:
        validated = self.schema.validate(values, source=self.source)
        return self._audit(validated, rendered)

    def _audit(self, validated: dict[str, Any], rendered: str) -> dict[str, str]:
        return {
            "renderer_version": self.renderer_version,
            "json_version": self.json_version,
            "template_sha256": self.template_sha256,
            "schema_sha256": self.schema_sha256,
            "program_sha256": self.program_sha256,
            "args_sha256": _sha256(canonical_json(validated)),
            "rendered_sha256": _sha256(rendered),
        }


def compile_prompt(
    template: str,
    *,
    arguments: dict[str, Any] | ShaftPromptSchema | None = None,
    source: str = "prompt",
) -> ShaftPromptProgram:
    """Compile the deliberately small Shaft prompt language."""

    if not isinstance(template, str):
        raise ValueError(f"Prompt template must be a string ({source}).")
    validate_prompt_text(template, source=source)
    schema = (
        arguments
        if isinstance(arguments, ShaftPromptSchema)
        else ShaftPromptSchema.from_mapping(arguments, source=source)
    )
    schema_by_name = {argument.name: argument for argument in schema.arguments}
    parts: list[_PromptPart] = []
    referenced: list[str] = []
    position = 0
    while position < len(template):
        open_at = template.find("{{", position)
        if open_at < 0:
            parts.append(_PromptPart(literal=template[position:]))
            break
        if open_at > position:
            parts.append(_PromptPart(literal=template[position:open_at]))
        expression_end = template.find("}}", open_at + 2)
        if expression_end < 0:
            raise ValueError(f"Unclosed prompt expression at offset {open_at} ({source}).")
        expression = template[open_at + 2 : expression_end]
        match = _EXPRESSION.fullmatch(expression)
        if match is None or "{{" in expression:
            raise ValueError(f"Unsupported prompt expression {expression!r} ({source}).")
        name, filter_name = match.groups()
        argument = schema_by_name.get(name)
        if argument is None:
            raise ValueError(f"Prompt argument {name!r} is not declared ({source}).")
        uses_json = filter_name == "json"
        if not uses_json and argument.type not in {"string", "enum"}:
            raise ValueError(
                f"Prompt argument {name!r} with type {argument.type!r} must use the json filter "
                f"({source})."
            )
        parts.append(_PromptPart(argument=name, json_filter=uses_json))
        referenced.append(name)
        position = expression_end + 2
    if not parts:
        parts.append(_PromptPart(literal=""))
    template_sha256 = _sha256(template)
    schema_sha256 = schema.fingerprint
    program_sha256 = _sha256(
        canonical_json(
            {
                "renderer_version": PROMPT_RENDERER_VERSION,
                "json_version": PROMPT_JSON_VERSION,
                "template": template,
                "schema": schema.canonical_payload,
            }
        )
    )
    return ShaftPromptProgram(
        template=template,
        schema=schema,
        source=source,
        parts=tuple(parts),
        referenced_arguments=tuple(dict.fromkeys(referenced)),
        template_sha256=template_sha256,
        schema_sha256=schema_sha256,
        program_sha256=program_sha256,
    )


@dataclass(frozen=True)
class ShaftPromptTemplate:
    prompt_id: str
    system_prompt: str
    static_user_prompt: str | None
    metadata: dict[str, Any]
    source_path: str
    program: ShaftPromptProgram = field(repr=False)
    variant_id: str | None = None
    version: str | None = None
    sampling_weight: float = 1.0
    user_prompt_template: str | None = None

    @property
    def user_prompt(self) -> str:
        if self.static_user_prompt is None:
            raise ValueError(
                f"Prompt {self.prompt_id!r} is parameterized; call render(prompt_args) instead."
            )
        return self.static_user_prompt

    def render(self, prompt_args: dict[str, Any] | None = None, *, context: str | None = None) -> str:
        return self.program.render(prompt_args, context=context)

    def render_with_audit(
        self,
        prompt_args: dict[str, Any] | None = None,
        *,
        context: str | None = None,
    ) -> tuple[str, dict[str, str]]:
        return self.program.render_with_audit(prompt_args, context=context)


def load_prompt_template(path: str | Path, *, variant_id: str = "main") -> ShaftPromptTemplate:
    """Load a prompt from a legacy single prompt YAML or a versioned prompt pool YAML."""

    prompt_path = Path(path)
    payload = _load_yaml_mapping(prompt_path)
    metadata = _load_metadata(payload, prompt_path)
    prompts = payload.get("prompts")
    if isinstance(prompts, list):
        variants = _load_pool_prompts(prompt_path, payload=payload, metadata=metadata)
        for variant in variants:
            if variant.variant_id == variant_id:
                return variant
        raise ValueError(f"Prompt pool variant {variant_id!r} not found in {prompt_path}.")

    prompt = payload.get("prompt") or {}
    if not isinstance(prompt, dict):
        raise ValueError(f"prompt file must contain prompt mapping: {prompt_path}")
    prompt_id = str(metadata.get("id") or payload.get("prompt_id") or prompt_path.stem).strip()
    raw_system_prompt = prompt.get("system_prompt", payload.get("system_prompt", ""))
    raw_user_prompt = prompt.get("user_prompt", payload.get("user_prompt", ""))
    if not isinstance(raw_system_prompt, str) or not isinstance(raw_user_prompt, str):
        raise ValueError(f"Legacy prompt system_prompt/user_prompt must be strings: {prompt_path}")
    system_prompt = raw_system_prompt.strip()
    user_prompt = raw_user_prompt.strip()
    validate_prompt_text(system_prompt, source=f"{prompt_path}:system_prompt")
    if not user_prompt:
        raise ValueError(f"Missing user_prompt in {prompt_path}.")
    _reject_template_markers(system_prompt, source=f"{prompt_path}:system_prompt")
    _reject_template_markers(user_prompt, source=f"{prompt_path}:user_prompt")
    program = compile_prompt(user_prompt, source=str(prompt_path))
    return ShaftPromptTemplate(
        prompt_id=prompt_id,
        system_prompt=system_prompt,
        static_user_prompt=user_prompt,
        metadata=dict(metadata),
        source_path=str(prompt_path),
        program=program,
    )


def load_prompt_pool(path: str | Path) -> list[ShaftPromptTemplate]:
    """Load and compile every prompt variant from a versioned prompt pool YAML."""

    prompt_path = Path(path)
    payload = _load_yaml_mapping(prompt_path)
    metadata = _load_metadata(payload, prompt_path)
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Prompt pool must contain a non-empty prompts list: {prompt_path}")
    return _load_pool_prompts(prompt_path, payload=payload, metadata=metadata)


def _load_yaml_mapping(prompt_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(prompt_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"prompt file must contain a mapping: {prompt_path}")
    return payload


def _load_metadata(payload: dict[str, Any], prompt_path: Path) -> dict[str, Any]:
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"prompt metadata must be a mapping: {prompt_path}")
    return metadata


def _load_pool_prompts(
    prompt_path: Path,
    *,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> list[ShaftPromptTemplate]:
    pool_id = str(metadata.get("id") or "").strip()
    if not pool_id:
        raise ValueError(f"Missing prompt pool id in {prompt_path}.")
    version = str(metadata.get("version") or "").strip()
    if not version:
        raise ValueError(f"Missing prompt pool version in {prompt_path}.")
    raw_arguments = payload["arguments"] if "arguments" in payload else None
    schema = ShaftPromptSchema.from_mapping(raw_arguments, source=str(prompt_path))
    prompts = payload["prompts"]
    variants: list[ShaftPromptTemplate] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(prompts):
        if not isinstance(item, dict):
            raise TypeError(f"Prompt pool item must be a mapping: {prompt_path}:prompts[{index}]")
        variant_id = str(item.get("id") or "").strip()
        if not variant_id:
            raise ValueError(f"Prompt pool item is missing id: {prompt_path}:prompts[{index}]")
        if variant_id in seen_ids:
            raise ValueError(f"Duplicate prompt variant id {variant_id!r} in {prompt_path}")
        seen_ids.add(variant_id)
        variants.append(
            _load_pool_prompt_item(
                prompt_path,
                item=item,
                metadata=metadata,
                pool_id=pool_id,
                version=version,
                variant_id=variant_id,
                schema=schema,
            )
        )
    if not any(prompt.sampling_weight > 0 for prompt in variants):
        raise ValueError(f"Prompt pool must have at least one positive sampling_weight: {prompt_path}")
    return variants


def _load_pool_prompt_item(
    prompt_path: Path,
    *,
    item: dict[str, Any],
    metadata: dict[str, Any],
    pool_id: str,
    version: str,
    variant_id: str,
    schema: ShaftPromptSchema,
) -> ShaftPromptTemplate:
    source = f"{prompt_path}#{variant_id}"
    try:
        sampling_weight = float(item.get("sampling_weight", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Prompt pool variant {variant_id!r} has invalid sampling_weight in {prompt_path}."
        ) from exc
    if not math.isfinite(sampling_weight) or sampling_weight < 0:
        raise ValueError(
            f"Prompt pool variant {variant_id!r} sampling_weight must be finite and >= 0 "
            f"in {prompt_path}."
        )
    raw_system_prompt = item.get("system_prompt", "")
    if not isinstance(raw_system_prompt, str):
        raise ValueError(
            f"Prompt pool variant {variant_id!r} system_prompt must be a string in {prompt_path}."
        )
    system_prompt = raw_system_prompt.strip()
    validate_prompt_text(system_prompt, source=f"{source}:system_prompt")
    _reject_template_markers(system_prompt, source=f"{source}:system_prompt")
    has_static = "user_prompt" in item and item.get("user_prompt") is not None
    has_dynamic = "user_prompt_template" in item and item.get("user_prompt_template") is not None
    if has_static == has_dynamic:
        raise ValueError(
            f"Prompt pool variant {variant_id!r} must define exactly one of user_prompt or "
            f"user_prompt_template in {prompt_path}."
        )
    if has_static:
        raw_user_prompt = item["user_prompt"]
        if not isinstance(raw_user_prompt, str):
            raise ValueError(
                f"Prompt pool variant {variant_id!r} user_prompt must be a string in {prompt_path}."
            )
        user_prompt = raw_user_prompt.strip()
        if not user_prompt:
            raise ValueError(f"Prompt pool variant {variant_id!r} has empty user_prompt in {prompt_path}.")
        _reject_template_markers(user_prompt, source=f"{source}:user_prompt")
        user_prompt_template = None
        program = compile_prompt(user_prompt, arguments=schema, source=source)
    else:
        raw_template = item["user_prompt_template"]
        if not isinstance(raw_template, str):
            raise ValueError(
                f"Prompt pool variant {variant_id!r} user_prompt_template must be a string "
                f"in {prompt_path}."
            )
        user_prompt_template = raw_template.strip()
        if not user_prompt_template:
            raise ValueError(
                f"Prompt pool variant {variant_id!r} has empty user_prompt_template in {prompt_path}."
            )
        program = compile_prompt(user_prompt_template, arguments=schema, source=source)
        if not program.referenced_arguments:
            raise ValueError(
                f"Dynamic prompt variant {variant_id!r} must reference at least one argument "
                f"in {prompt_path}."
            )
        user_prompt = None
    prompt_metadata = dict(metadata)
    prompt_metadata["prompt_pool_id"] = pool_id
    prompt_metadata["prompt_version"] = version
    prompt_metadata["prompt_variant_id"] = variant_id
    return ShaftPromptTemplate(
        prompt_id=f"{pool_id}.{variant_id}",
        system_prompt=system_prompt,
        static_user_prompt=user_prompt,
        user_prompt_template=user_prompt_template,
        metadata=prompt_metadata,
        source_path=source,
        variant_id=variant_id,
        version=version,
        sampling_weight=sampling_weight,
        program=program,
    )


def _reject_template_markers(text: str, *, source: str) -> None:
    if "{{" in text:
        raise ValueError(f"Template expressions are not allowed in static prompt text ({source}).")
