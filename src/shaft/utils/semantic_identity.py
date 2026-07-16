from __future__ import annotations

import ast
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import dis
from enum import Enum
import functools
import hashlib
import inspect
import json
import logging
import math
from pathlib import Path
import textwrap
from types import CodeType, ModuleType
from typing import Any


_UNENCODABLE_RUNTIME_VALUE = object()
_RUNTIME_CONSTANT_NODE_BUDGET = 250_000
_RUNTIME_CONSTANT_DEPTH_BUDGET = 64


class _SemanticContext:
    """Track the active object graph without leaking process-local ids.

    Runtime implementations regularly point back to their defining class or
    module through closures and callable instance state.  Those references are
    semantic, but traversing them recursively without a graph guard makes the
    fingerprint builder recurse forever.  The marker emitted for a back edge is
    derived only from the stable kind/type of the value; ``id`` is used solely
    for traversal bookkeeping and never enters the persisted payload.
    """

    def __init__(self) -> None:
        self.active: set[tuple[str, int]] = set()
        # Process-local ids are cache keys only and never enter persisted
        # payloads. Reusing one large registry across several methods must not
        # re-hash or re-materialize it for every reference.
        self.runtime_constant_cache: dict[tuple[str, int], Any] = {}
        self.callable_digest_cache: dict[tuple[int, bool], str] = {}
        self.runtime_constant_nodes = 0
        self.runtime_constant_depth = 0

    def enter(self, kind: str, value: Any) -> bool:
        key = (kind, id(value))
        if key in self.active:
            return False
        self.active.add(key)
        return True

    def leave(self, kind: str, value: Any) -> None:
        self.active.remove((kind, id(value)))

    def enter_runtime_constant(self) -> None:
        self.runtime_constant_nodes += 1
        if self.runtime_constant_nodes > _RUNTIME_CONSTANT_NODE_BUDGET:
            raise ValueError(
                "Semantic runtime-constant identity exceeded its node budget "
                f"({_RUNTIME_CONSTANT_NODE_BUDGET})."
            )
        next_depth = self.runtime_constant_depth + 1
        if next_depth > _RUNTIME_CONSTANT_DEPTH_BUDGET:
            raise ValueError(
                "Semantic runtime-constant identity exceeded its depth budget "
                f"({_RUNTIME_CONSTANT_DEPTH_BUDGET})."
            )
        self.runtime_constant_depth = next_depth

    def leave_runtime_constant(self) -> None:
        self.runtime_constant_depth -= 1


def _cycle_payload(kind: str, value: Any) -> dict[str, Any]:
    return {
        "cycle": {
            "kind": kind,
            "type": _qualified_name(value),
        }
    }


def _qualified_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _type_expression_payload(value: Any) -> dict[str, str] | None:
    """Encode stdlib typing aliases without walking their delegated class state."""

    qualified_type = _qualified_name(value)
    if qualified_type == "types.GenericAlias" or qualified_type in {
        "collections.abc._CallableGenericAlias",
        "typing._CallableGenericAlias",
        "typing._GenericAlias",
        "typing._SpecialGenericAlias",
        "typing._UnionGenericAlias",
        "typing._LiteralGenericAlias",
        "typing._AnnotatedAlias",
        "typing._SpecialForm",
        "typing._AnyMeta",
    }:
        # These objects are immutable type expressions. Their stdlib repr is a
        # canonical spelling of origin/arguments, not a process-address repr.
        return {
            "type_expression": qualified_type,
            "value": repr(value),
        }
    return None


def _torch_enum_payload(value: Any) -> dict[str, Any] | None:
    qualified_type = _qualified_name(value)
    if qualified_type != "torch._C._distributed_c10d.ReduceOp.RedOpType":
        return None
    name = getattr(value, "name", None)
    enum_value = getattr(value, "value", None)
    if type(name) is not str or type(enum_value) is not int:
        return None
    return {
        "torch_enum": qualified_type,
        "name": name,
        "value": enum_value,
    }


def _semantic_state_provider(value: Any) -> Any | None:
    """Return an explicitly declared semantic-state provider.

    Dynamic proxies such as ``unittest.mock.MagicMock`` manufacture arbitrary
    attributes from ``__getattr__``. Querying ``shaft_semantic_state`` with a
    normal ``getattr`` would therefore create an endless chain of child mocks
    and turn identity construction into an OOM. Static lookup proves that the
    protocol is actually declared before normal descriptor binding is allowed.
    """

    try:
        inspect.getattr_static(value, "shaft_semantic_state")
    except AttributeError:
        return None
    provider = getattr(value, "shaft_semantic_state")
    return provider if callable(provider) else None


def _declares_dynamic_attribute_protocol(value: Any) -> bool:
    cls = value if isinstance(value, type) else type(value)
    for base in cls.__mro__:
        namespace = vars(base)
        if "__getattr__" in namespace:
            return True
        descriptor = namespace.get("__getattribute__")
        if base is not object and isinstance(
            getattr(inspect.unwrap(descriptor), "__code__", None),
            CodeType,
        ):
            return True
    return False


def _known_semantic_projection(
    value: Any,
    *,
    context: _SemanticContext,
) -> Any | None:
    """Project vetted upstream lazy containers without materializing them."""

    if _qualified_name(value) != "transformers.models.auto.auto_factory._LazyAutoMapping":
        return None
    projection: dict[str, Any] = {}
    for name in (
        "_config_mapping",
        "_reverse_config_mapping",
        "_model_mapping",
        "_extra_content",
    ):
        try:
            item = inspect.getattr_static(value, name)
        except AttributeError:
            return None
        if type(item) not in {dict, OrderedDict}:
            return None
        projection[name] = item
    cls = type(value)
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return None
    live_methods: dict[str, str] = {}
    for name in (
        "__init__",
        "__len__",
        "__getitem__",
        "_load_attr_from_module",
        "keys",
        "get",
        "__bool__",
        "values",
        "items",
        "__iter__",
        "__contains__",
        "register",
    ):
        try:
            target = inspect.getattr_static(cls, name)
        except AttributeError:
            return None
        if isinstance(target, (classmethod, staticmethod)):
            target = target.__func__
        if not callable(target):
            return None
        live_methods[name] = _callable_digest(
            target,
            include_dependencies=False,
            context=context,
        )
    # ``_reverse_config_mapping`` is constructor-derived but is read directly
    # by __getitem__/__contains__/register and can be mutated independently, so
    # it is a live behavior source. ``_modules`` remains excluded because it is
    # only an access-history import cache.
    return {
        "type": _qualified_name(value),
        "implementation": {
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "live_methods": live_methods,
        },
        "state": projection,
    }


def _code_payload(code: CodeType) -> dict[str, Any]:
    constants: list[Any] = []
    for item in code.co_consts:
        if isinstance(item, CodeType):
            constants.append(_code_payload(item))
        elif item is None or isinstance(item, (bool, int, float, str)):
            constants.append(item)
        elif isinstance(item, bytes):
            constants.append({"bytes": item.hex()})
        else:
            constants.append({"type": _qualified_name(item)})
    return {
        "bytecode": code.co_code.hex(),
        "constants": constants,
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
    }


def _canonical(
    value: Any,
    *,
    context: _SemanticContext,
    include_dependencies: bool,
) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if type(value) is object:
        # Third-party APIs commonly use a private ``object()`` instance as an
        # omitted-argument sentinel.  Its process-local identity is not
        # reproducible, while the presence and exact built-in type of the
        # sentinel are.  Never fall through to repr/id based encoding.
        return {"opaque_sentinel": "builtins.object"}
    qualified_type = _qualified_name(value)
    torch_enum = _torch_enum_payload(value)
    if torch_enum is not None:
        return torch_enum
    type_expression = _type_expression_payload(value)
    if type_expression is not None:
        return type_expression
    if qualified_type in {
        "torch.device",
        "torch.dtype",
        "torch.layout",
        "torch.memory_format",
    }:
        # PyTorch exposes these as immutable scalar configuration values with
        # no ``__dict__``. Their textual form is the documented stable value
        # (for example ``cuda:1`` or ``torch.bfloat16``), not an object repr.
        return {
            "typed_scalar": qualified_type,
            "value": str(value),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return {
            "enum": _qualified_name(value),
            "value": _canonical(
                value.value,
                context=context,
                include_dependencies=include_dependencies,
            ),
        }
    if isinstance(value, type):
        return {"class": _qualified_name(value)}
    if isinstance(value, logging.Logger):
        return {"logger": value.name}
    if qualified_type == "unittest.mock._SentinelObject":
        # ``unittest.mock.DEFAULT`` is an immutable named sentinel used in
        # real stdlib signatures. It is not a dynamic Mock proxy: the declared
        # name is its complete process-stable value identity.
        name = inspect.getattr_static(value, "name", None)
        if type(name) is not str:
            raise TypeError("Semantic identity cannot encode an unnamed mock sentinel.")
        return {"named_sentinel": qualified_type, "name": name}
    if type(value).__module__ == "unittest.mock":
        # Mock objects are dynamic callable proxies: hashing their fabricated
        # call surface would accept test/runtime state that has no stable
        # implementation identity and can recursively manufacture child mocks.
        raise TypeError("Semantic identity cannot encode a dynamic unittest.mock proxy.")
    if isinstance(value, ModuleType):
        raise TypeError(
            "Semantic identity cannot encode a module captured as an opaque value; "
            "reference deterministic module attributes directly or provide a bounded "
            "shaft_semantic_state() wrapper."
        )
    if isinstance(value, Mapping) and type(value) not in {dict, OrderedDict}:
        semantic_state = _semantic_state_provider(value)
        if semantic_state is not None:
            return {
                "custom_mapping": _qualified_name(value),
                "implementation": _component_payload(
                    value,
                    include_dependencies=include_dependencies,
                    context=context,
                ),
                "semantic_state": _canonical(
                    semantic_state(),
                    context=context,
                    include_dependencies=include_dependencies,
                ),
            }
        known_projection = _known_semantic_projection(value, context=context)
        if known_projection is not None:
            return _canonical(
                known_projection,
                context=context,
                include_dependencies=include_dependencies,
            )
        raise TypeError(
            "Semantic identity cannot encode custom Mapping "
            f"{_qualified_name(value)}; implement shaft_semantic_state() with "
            "bounded, deterministic state."
        )
    if callable(value):
        return {
            "callable_sha256": _callable_digest(
                value,
                include_dependencies=include_dependencies,
                context=context,
            )
        }
    semantic_state = _semantic_state_provider(value)
    if semantic_state is not None:
        return {
            "object": _qualified_name(value),
            "implementation": _component_payload(
                value,
                include_dependencies=include_dependencies,
                context=context,
            ),
            "semantic_state": _canonical(
                semantic_state(),
                context=context,
                include_dependencies=include_dependencies,
            ),
        }
    if is_dataclass(value) and not isinstance(value, type):
        if not context.enter("dataclass", value):
            return _cycle_payload("dataclass", value)
        try:
            return {
                field.name: _canonical(
                    getattr(value, field.name),
                    context=context,
                    include_dependencies=include_dependencies,
                )
                for field in fields(value)
            }
        finally:
            context.leave("dataclass", value)
    known_projection = _known_semantic_projection(value, context=context)
    if known_projection is not None:
        return _canonical(
            known_projection,
            context=context,
            include_dependencies=include_dependencies,
        )
    if type(value) in {dict, OrderedDict}:
        if not context.enter("mapping", value):
            return _cycle_payload("mapping", value)
        try:
            entries = [
                [
                    _canonical(
                        key,
                        context=context,
                        include_dependencies=include_dependencies,
                    ),
                    _canonical(
                        item,
                        context=context,
                        include_dependencies=include_dependencies,
                    ),
                ]
                for key, item in value.items()
            ]
            if type(value) is dict:
                entries.sort(
                    key=lambda item: json.dumps(
                        item[0],
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            return {
                "ordered_mapping" if type(value) is OrderedDict else "mapping": entries
            }
        finally:
            context.leave("mapping", value)
    if isinstance(value, Mapping):
        raise AssertionError("custom Mapping must be handled before dataclass/callable values")
    if type(value) in {list, tuple}:
        if not context.enter("sequence", value):
            return _cycle_payload("sequence", value)
        try:
            return [
                _canonical(
                    item,
                    context=context,
                    include_dependencies=include_dependencies,
                )
                for item in value
            ]
        finally:
            context.leave("sequence", value)
    if type(value) in {set, frozenset}:
        if not context.enter("set", value):
            return _cycle_payload("set", value)
        try:
            normalized = [
                _canonical(
                    item,
                    context=context,
                    include_dependencies=include_dependencies,
                )
                for item in value
            ]
            return sorted(
                normalized,
                key=lambda item: json.dumps(item, sort_keys=True),
            )
        finally:
            context.leave("set", value)
    if _declares_dynamic_attribute_protocol(value):
        raise TypeError(
            "Semantic identity cannot encode a dynamic attribute proxy without an "
            "explicit shaft_semantic_state() projection."
        )
    try:
        state = vars(value)
    except TypeError:
        state = None
    if isinstance(state, Mapping):
        if not context.enter("object", value):
            return _cycle_payload("object", value)
        try:
            return {
                "object": _qualified_name(value),
                "implementation": _component_payload(
                    value,
                    include_dependencies=include_dependencies,
                    context=context,
                ),
                "state": _canonical(
                    state,
                    context=context,
                    include_dependencies=include_dependencies,
                ),
            }
        finally:
            context.leave("object", value)
    raise TypeError(
        f"Semantic identity cannot encode {_qualified_name(value)}; use a dataclass "
        "or implement shaft_semantic_state()."
    )


def _runtime_constant_payload(
    value: Any,
    *,
    context: _SemanticContext,
) -> Any:
    context.enter_runtime_constant()
    try:
        return _runtime_constant_payload_impl(value, context=context)
    finally:
        context.leave_runtime_constant()


def _runtime_constant_payload_impl(
    value: Any,
    *,
    context: _SemanticContext,
) -> Any:
    """Encode live constants without serializing arbitrary process state.

    JSON-like immutable values, enums, paths, dataclasses, and exact containers
    are value-bound. Classes are identity-bound. Opaque instances are never
    repr-serialized; they must expose deterministic state or fail closed.
    """

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
        return value
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if type(value) is object:
        return {"opaque_sentinel": "builtins.object"}
    qualified_type = _qualified_name(value)
    torch_enum = _torch_enum_payload(value)
    if torch_enum is not None:
        return torch_enum
    if qualified_type in {
        "torch.device",
        "torch.dtype",
        "torch.layout",
        "torch.memory_format",
    }:
        return {
            "typed_scalar": qualified_type,
            "value": str(value),
        }
    type_expression = _type_expression_payload(value)
    if type_expression is not None:
        return type_expression
    if isinstance(value, Enum):
        encoded = _runtime_constant_payload(value.value, context=context)
        if encoded is _UNENCODABLE_RUNTIME_VALUE:
            return _UNENCODABLE_RUNTIME_VALUE
        return {"enum": _qualified_name(value), "value": encoded}
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, ModuleType):
        # A module name/version does not bind live monkeypatches. Callables may
        # encode the exact attributes they reference through
        # ``_module_reference_payload`` below; an otherwise opaque module value
        # is not a safe exact-resume constant.
        return _UNENCODABLE_RUNTIME_VALUE
    if isinstance(value, logging.Logger):
        # A module logger is an observability sink, not trajectory state. Its
        # handler/manager graph is mutable, cyclic and process-local; binding
        # the stable logger name proves which sink the implementation targets
        # without making log configuration part of exact resume.
        return {"logger": value.name}
    if isinstance(value, type):
        return {"class": _qualified_name(value)}
    if qualified_type == "unittest.mock._SentinelObject":
        name = inspect.getattr_static(value, "name", None)
        if type(name) is not str:
            return _UNENCODABLE_RUNTIME_VALUE
        return {
            "named_sentinel": "unittest.mock._SentinelObject",
            "name": name,
        }
    if type(value).__module__ == "unittest.mock":
        return _UNENCODABLE_RUNTIME_VALUE
    if isinstance(value, Mapping) and type(value) not in {dict, OrderedDict}:
        semantic_state = _semantic_state_provider(value)
        if semantic_state is not None:
            encoded = _runtime_constant_payload(semantic_state(), context=context)
            if encoded is _UNENCODABLE_RUNTIME_VALUE:
                return _UNENCODABLE_RUNTIME_VALUE
            return {
                "custom_mapping": _qualified_name(value),
                "implementation": _component_payload(
                    value,
                    include_dependencies=False,
                    context=context,
                ),
                "semantic_state": encoded,
            }
        known_projection = _known_semantic_projection(value, context=context)
        if known_projection is not None:
            encoded = _runtime_constant_payload(known_projection, context=context)
            if encoded is _UNENCODABLE_RUNTIME_VALUE:
                return _UNENCODABLE_RUNTIME_VALUE
            return {"semantic_projection": encoded}
        return _UNENCODABLE_RUNTIME_VALUE
    if callable(value):
        return {
            "callable_sha256": _callable_digest(
                value,
                include_dependencies=False,
                context=context,
            )
        }
    semantic_state = _semantic_state_provider(value)
    if semantic_state is not None:
        encoded = _runtime_constant_payload(semantic_state(), context=context)
        if encoded is _UNENCODABLE_RUNTIME_VALUE:
            return _UNENCODABLE_RUNTIME_VALUE
        return {
            "object": _qualified_name(value),
            "implementation": _component_payload(
                value,
                include_dependencies=False,
                context=context,
            ),
            "semantic_state": encoded,
        }
    if is_dataclass(value) and not isinstance(value, type):
        cache_key = ("runtime_dataclass", id(value))
        if cache_key in context.runtime_constant_cache:
            return context.runtime_constant_cache[cache_key]
        if not context.enter("runtime_dataclass", value):
            return _cycle_payload("runtime_dataclass", value)
        try:
            payload: dict[str, Any] = {}
            for field in fields(value):
                encoded = _runtime_constant_payload(
                    getattr(value, field.name),
                    context=context,
                )
                if encoded is _UNENCODABLE_RUNTIME_VALUE:
                    return _UNENCODABLE_RUNTIME_VALUE
                payload[field.name] = encoded
            result = {"dataclass": _qualified_name(value), "fields": payload}
            context.runtime_constant_cache[cache_key] = result
            return result
        finally:
            context.leave("runtime_dataclass", value)
    known_projection = _known_semantic_projection(value, context=context)
    if known_projection is not None:
        encoded = _runtime_constant_payload(known_projection, context=context)
        if encoded is _UNENCODABLE_RUNTIME_VALUE:
            return _UNENCODABLE_RUNTIME_VALUE
        return {"semantic_state": encoded}
    if type(value) in {dict, OrderedDict}:
        cache_key = ("runtime_mapping", id(value))
        if cache_key in context.runtime_constant_cache:
            return context.runtime_constant_cache[cache_key]
        if not context.enter("runtime_mapping", value):
            return _cycle_payload("runtime_mapping", value)
        try:
            entry_digests: list[bytes] = []
            for key, item in value.items():
                encoded_key = _runtime_constant_payload(key, context=context)
                encoded_item = _runtime_constant_payload(item, context=context)
                if (
                    encoded_key is _UNENCODABLE_RUNTIME_VALUE
                    or encoded_item is _UNENCODABLE_RUNTIME_VALUE
                ):
                    return _UNENCODABLE_RUNTIME_VALUE
                entry_digests.append(
                    hashlib.sha256(
                        _canonical_payload_bytes(encoded_key)
                        + b"\x00"
                        + _canonical_payload_bytes(encoded_item)
                    ).digest()
                )
            ordered = type(value) is OrderedDict
            if not ordered:
                entry_digests.sort()
            kind = "ordered_mapping" if ordered else "mapping"
            digest = hashlib.sha256(f"{kind}\x00".encode("ascii"))
            for entry_digest in entry_digests:
                digest.update(entry_digest)
            result = {
                "container": kind,
                "length": len(entry_digests),
                "sha256": digest.hexdigest(),
            }
            context.runtime_constant_cache[cache_key] = result
            return result
        finally:
            context.leave("runtime_mapping", value)
    if isinstance(value, Mapping):
        raise AssertionError("custom Mapping must be handled before dataclass/callable values")
    if type(value) in {list, tuple}:
        cache_key = ("runtime_sequence", id(value))
        if cache_key in context.runtime_constant_cache:
            return context.runtime_constant_cache[cache_key]
        if not context.enter("runtime_sequence", value):
            return _cycle_payload("runtime_sequence", value)
        try:
            kind = "tuple" if type(value) is tuple else "list"
            digest = hashlib.sha256(f"sequence:{kind}\x00".encode("ascii"))
            length = 0
            for item in value:
                encoded = _runtime_constant_payload(item, context=context)
                if encoded is _UNENCODABLE_RUNTIME_VALUE:
                    return _UNENCODABLE_RUNTIME_VALUE
                digest.update(hashlib.sha256(_canonical_payload_bytes(encoded)).digest())
                length += 1
            result = {
                "container": kind,
                "length": length,
                "sha256": digest.hexdigest(),
            }
            context.runtime_constant_cache[cache_key] = result
            return result
        finally:
            context.leave("runtime_sequence", value)
    if type(value) in {set, frozenset}:
        cache_key = ("runtime_set", id(value))
        if cache_key in context.runtime_constant_cache:
            return context.runtime_constant_cache[cache_key]
        if not context.enter("runtime_set", value):
            return _cycle_payload("runtime_set", value)
        try:
            kind = "frozenset" if type(value) is frozenset else "set"
            item_digests: list[bytes] = []
            for item in value:
                encoded = _runtime_constant_payload(item, context=context)
                if encoded is _UNENCODABLE_RUNTIME_VALUE:
                    return _UNENCODABLE_RUNTIME_VALUE
                item_digests.append(
                    hashlib.sha256(_canonical_payload_bytes(encoded)).digest()
                )
            item_digests.sort()
            digest = hashlib.sha256(f"set:{kind}\x00".encode("ascii"))
            for item_digest in item_digests:
                digest.update(item_digest)
            result = {
                "container": kind,
                "length": len(item_digests),
                "sha256": digest.hexdigest(),
            }
            context.runtime_constant_cache[cache_key] = result
            return result
        finally:
            context.leave("runtime_set", value)
    if _declares_dynamic_attribute_protocol(value):
        return _UNENCODABLE_RUNTIME_VALUE
    try:
        state = vars(value)
    except TypeError:
        state = None
    if type(state) is dict:
        if not context.enter("runtime_object", value):
            return _cycle_payload("runtime_object", value)
        try:
            encoded = _runtime_constant_payload(state, context=context)
            if encoded is _UNENCODABLE_RUNTIME_VALUE:
                return _UNENCODABLE_RUNTIME_VALUE
            return {
                "object": _qualified_name(value),
                "implementation": _component_payload(
                    value,
                    include_dependencies=False,
                    context=context,
                ),
                "state": encoded,
            }
        finally:
            context.leave("runtime_object", value)
    return _UNENCODABLE_RUNTIME_VALUE


def _canonical_payload_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _referenced_module_attribute_chains(code: CodeType) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Return statically visible ``module.attr`` chains rooted at globals.

    ``co_names`` alone cannot distinguish a module object from the attributes
    consumed through it. Walking bytecode lets identity bind only the declared
    live surface instead of serializing an entire framework module/registry.
    Every prefix is retained so intermediate module replacements are covered.
    """

    chains: dict[str, set[tuple[str, ...]]] = {}
    root: str | None = None
    attributes: list[str] = []
    for instruction in dis.get_instructions(code):
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"} and type(
            instruction.argval
        ) is str:
            root = instruction.argval
            attributes = []
            continue
        if (
            root is not None
            and instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
            and type(instruction.argval) is str
        ):
            attributes.append(instruction.argval)
            chains.setdefault(root, set()).add(tuple(attributes))
            continue
        if instruction.opname in {"CACHE", "EXTENDED_ARG", "NOP"}:
            continue
        root = None
        attributes = []
    return {
        name: tuple(sorted(value))
        for name, value in sorted(chains.items())
    }


def _module_reference_payload(
    module: ModuleType,
    *,
    attribute_chains: tuple[tuple[str, ...], ...],
    context: _SemanticContext,
) -> Any:
    """Bind only live module attributes directly consumed by one callable."""

    if not attribute_chains:
        return _UNENCODABLE_RUNTIME_VALUE
    version = getattr(module, "__version__", None)
    if type(version) not in {int, float, str} or (
        type(version) is float and not math.isfinite(version)
    ):
        version = None
    attributes: dict[str, Any] = {}
    for chain in attribute_chains:
        current: Any = module
        resolved_path: list[str] = []
        missing_path: str | None = None
        for name in chain:
            try:
                current = inspect.getattr_static(current, name)
            except AttributeError:
                missing_path = ".".join((*resolved_path, name))
                break
            resolved_path.append(name)
        if missing_path is not None:
            # Optional backends regularly leave guarded module attributes
            # undefined. Missing is itself a stable live state; if the backend
            # later installs that attribute, the payload changes.
            attributes[".".join(chain)] = {"missing_attribute": missing_path}
            continue
        if isinstance(current, (classmethod, staticmethod)):
            current = current.__func__
        if isinstance(current, ModuleType):
            encoded = {
                "module": str(getattr(current, "__name__", "")),
                "version": (
                    getattr(current, "__version__", None)
                    if type(getattr(current, "__version__", None)) in {int, float, str}
                    else None
                ),
                "source_sha256": _module_source_sha256(current),
            }
        elif callable(current):
            encoded = {
                "callable_sha256": _callable_digest(
                    current,
                    include_dependencies=False,
                    context=context,
                )
            }
        else:
            encoded = _runtime_constant_payload(current, context=context)
        if encoded is _UNENCODABLE_RUNTIME_VALUE:
            return _UNENCODABLE_RUNTIME_VALUE
        attributes[".".join(chain)] = encoded
    return {
        "module": str(getattr(module, "__name__", "")),
        "version": version,
        "source_sha256": _module_source_sha256(module),
        "referenced_attributes": attributes,
    }


def _callable_digest(
    value: Any,
    *,
    include_dependencies: bool,
    context: _SemanticContext,
) -> str:
    target = inspect.unwrap(value)
    cache_key = (id(target), include_dependencies)
    is_active = ("callable", id(target)) in context.active
    if not is_active and cache_key in context.callable_digest_cache:
        return context.callable_digest_cache[cache_key]
    payload = _callable_payload(
        target,
        include_dependencies=include_dependencies,
        context=context,
    )
    digest = hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()
    if not is_active:
        context.callable_digest_cache[cache_key] = digest
    return digest


def _callable_payload(
    value: Any,
    *,
    include_dependencies: bool = True,
    context: _SemanticContext,
) -> dict[str, Any]:
    target = inspect.unwrap(value)
    if not context.enter("callable", target):
        return _cycle_payload("callable", target)
    code = getattr(target, "__code__", None)
    try:
        payload: dict[str, Any] = {
            "qualified_name": (
                f"{getattr(target, '__module__', type(target).__module__)}."
                f"{getattr(target, '__qualname__', type(target).__qualname__)}"
            )
        }
        if isinstance(code, CodeType):
            payload["code"] = _code_payload(code)
            payload["defaults"] = _canonical(
                getattr(target, "__defaults__", None),
                context=context,
                include_dependencies=False,
            )
            payload["kwdefaults"] = _canonical(
                getattr(target, "__kwdefaults__", None),
                context=context,
                include_dependencies=False,
            )
            closure: list[Any] = []
            for cell in getattr(target, "__closure__", ()) or ():
                try:
                    closure.append(
                        _canonical(
                            cell.cell_contents,
                            context=context,
                            include_dependencies=False,
                        )
                    )
                except ValueError:
                    closure.append({"empty_closure_cell": True})
            payload["closure"] = closure
            module = inspect.getmodule(target)
            if module is not None:
                try:
                    module_source = inspect.getsource(module)
                except (OSError, TypeError):
                    module_source = ""
                payload["owning_module_source_sha256"] = (
                    hashlib.sha256(module_source.encode("utf-8")).hexdigest()
                    if module_source
                    else None
                )
                if include_dependencies:
                    # Bind the globals a declared implementation actually
                    # consumes, including extension code outside shaft.*. The
                    # runtime-constant encoder below has bounded depth/nodes
                    # and rejects custom lazy mappings, so this stays
                    # fail-closed without traversing arbitrary registries.
                    dependencies: dict[str, Any] = {}
                    runtime_constants: dict[str, Any] = {}
                    module_attribute_chains = _referenced_module_attribute_chains(code)
                    for name in sorted(set(code.co_names)):
                        dependency = vars(module).get(name)
                        if dependency is target:
                            continue
                        if isinstance(dependency, ModuleType):
                            encoded = _module_reference_payload(
                                dependency,
                                attribute_chains=module_attribute_chains.get(name, ()),
                                context=context,
                            )
                            if encoded is _UNENCODABLE_RUNTIME_VALUE:
                                raise TypeError(
                                    "Exact callable semantic identity cannot encode module "
                                    f"global {module.__name__}.{name} without a bounded, "
                                    "statically referenced attribute surface."
                                )
                            runtime_constants[name] = encoded
                            continue
                        if callable(dependency):
                            dependencies[name] = {
                                "sha256": _callable_digest(
                                    dependency,
                                    include_dependencies=False,
                                    context=context,
                                )
                            }
                            continue
                        encoded = _runtime_constant_payload(
                            dependency,
                            context=context,
                        )
                        if encoded is _UNENCODABLE_RUNTIME_VALUE:
                            raise TypeError(
                                "Exact callable semantic identity cannot encode runtime "
                                f"global {module.__name__}.{name} of type "
                                f"{_qualified_name(dependency)}. Implement "
                                "shaft_semantic_state() with bounded, deterministic state."
                            )
                        runtime_constants[name] = encoded
                    payload["shaft_dependencies"] = dependencies
                    payload["runtime_constants"] = runtime_constants
        else:
            payload["component"] = _component_payload(
                target,
                include_dependencies=include_dependencies,
                context=context,
            )
        if not isinstance(target, type):
            state = getattr(target, "__dict__", None)
            if isinstance(state, Mapping) and state:
                payload["state"] = _canonical(
                    state,
                    context=context,
                    include_dependencies=False,
                )
        return payload
    finally:
        context.leave("callable", target)


def _component_payload(
    value: Any,
    *,
    include_dependencies: bool = True,
    context: _SemanticContext,
) -> dict[str, Any]:
    cls = value if isinstance(value, type) else type(value)
    if not context.enter("component", value):
        return _cycle_payload("component", value)
    mro: list[dict[str, Any]] = []
    try:
        for base in cls.__mro__:
            if base is object:
                continue
            try:
                source = inspect.getsource(base)
            except (OSError, TypeError):
                source = ""
            declared_runtime_names = _declared_runtime_names(source)
            source_visible = bool(source)
            runtime_methods: dict[str, Any] = {}
            runtime_constants: dict[str, Any] = {}
            for name, descriptor in sorted(vars(base).items()):
                if name == "_abc_impl" and _qualified_name(descriptor) == "_abc._abc_data":
                    # CPython attaches the mutable ABC membership cache at
                    # runtime. It is derived from registrations/import order,
                    # not declared behavior, and class source/MRO already bind
                    # the abstract protocol implementation.
                    continue
                if inspect.ismemberdescriptor(descriptor) or inspect.isgetsetdescriptor(
                    descriptor
                ):
                    # ``slots=True`` and C-extension fields expose generated
                    # storage descriptors under source-declared names. The
                    # declaration/source binds the field; the process-local
                    # descriptor object is not an independent runtime policy.
                    continue
                target = descriptor
                binding = "instance"
                if isinstance(descriptor, (classmethod, staticmethod)):
                    target = descriptor.__func__
                    binding = "class" if isinstance(descriptor, classmethod) else "static"
                elif isinstance(descriptor, property):
                    target = descriptor.fget
                    binding = "property"
                elif isinstance(descriptor, functools.cached_property):
                    target = descriptor.func
                    binding = "cached_property"
                if callable(target) and isinstance(
                    getattr(inspect.unwrap(target), "__code__", None),
                    CodeType,
                ):
                    if (
                        source_visible
                        and name not in declared_runtime_names
                    ) or (not source_visible and name.startswith("__")):
                        # Dataclass/upstream constructors attach generated or
                        # short-lived helpers (for example ``__repr__`` and
                        # Transformers' ``smart_apply``) to a class. They are
                        # derived implementation detail, not an independently
                        # declared policy surface, and their closures may retain
                        # enormous runtime registries. The class source binds
                        # the declaration that generates them; live replacements
                        # of source-declared methods remain covered here.
                        continue
                    runtime_methods[name] = {
                        "binding": binding,
                        "sha256": _callable_digest(
                            target,
                            # Shaft-owned and source-unavailable extension
                            # classes must bind their live globals. For a
                            # source-visible upstream class, the full owning
                            # module source is already recorded below; walking
                            # every method's mutable framework registries makes
                            # unrelated HF caches part of the contract and can
                            # explode startup work.
                            include_dependencies=(
                                include_dependencies
                                and (
                                    str(getattr(base, "__module__", "")).startswith(
                                        "shaft."
                                    )
                                    or not source_visible
                                )
                            ),
                            context=context,
                        )
                    }
                    continue
                if name.startswith("__") or isinstance(descriptor, property):
                    continue
                if source_visible and name not in declared_runtime_names:
                    # Runtime-injected class state is not part of the declared
                    # policy surface. Binding every injected registry/cache can
                    # explode identity work and make construction history part
                    # of the contract. Declared constants remain live-bound.
                    continue
                encoded = _runtime_constant_payload(
                    descriptor,
                    context=context,
                )
                if encoded is _UNENCODABLE_RUNTIME_VALUE:
                    raise TypeError(
                        "Exact component semantic identity cannot encode declared runtime "
                        f"attribute {_qualified_name(base)}.{name} of type "
                        f"{_qualified_name(descriptor)}. Implement shaft_semantic_state() "
                        "with bounded, deterministic state."
                    )
                runtime_constants[name] = encoded
            mro.append(
                {
                    "class": _qualified_name(base),
                    "source_sha256": (
                        hashlib.sha256(source.encode("utf-8")).hexdigest() if source else None
                    ),
                    "runtime_methods": runtime_methods,
                    "runtime_constants": runtime_constants,
                    "owning_module_source_sha256": (
                        None
                        if inspect.getmodule(base) is None
                        else _module_source_sha256(inspect.getmodule(base))
                    ),
                }
            )
        return {
            "class": _qualified_name(cls),
            "implementation_mro": mro,
        }
    finally:
        context.leave("component", value)


def _module_source_sha256(module: Any) -> str | None:
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _declared_runtime_names(source: str) -> frozenset[str]:
    if not source:
        return frozenset()
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return frozenset()
    class_node = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef)),
        None,
    )
    if class_node is None:
        return frozenset()
    names: set[str] = set()
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return frozenset(names)


def callable_semantic_fingerprint(
    value: Any,
    *,
    role: str,
    include_dependencies: bool = True,
) -> str:
    if not callable(value):
        raise TypeError(f"{role} semantic identity requires a callable value.")
    if type(include_dependencies) is not bool:
        raise TypeError("include_dependencies must be a boolean.")
    payload = {
        "version": "shaft-shared-callable-semantic-v8",
        "role": str(role),
        "include_dependencies": bool(include_dependencies),
        "callable": _callable_payload(
            value,
            include_dependencies=include_dependencies,
            context=_SemanticContext(),
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def component_semantic_fingerprint(
    value: Any,
    *,
    role: str,
    include_state: bool = True,
) -> str:
    context = _SemanticContext()
    payload = {
        "version": "shaft-shared-component-semantic-v10",
        "role": str(role),
        "component": _component_payload(value, context=context),
        "state": (
            None
            if isinstance(value, type) or not include_state
            else _canonical(
                value,
                context=context,
                include_dependencies=False,
            )
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
