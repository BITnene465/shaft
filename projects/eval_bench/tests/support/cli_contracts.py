from __future__ import annotations

from eval_bench.cli import CLI_JSON_OUTPUT_SCHEMAS, _build_parser


def parser_command_names() -> set[str]:
    subparsers_action = next(
        action for action in _build_parser()._actions if action.dest == "command"
    )
    return set(subparsers_action.choices)


def assert_cli_json_payload(command_name: str, payload: object) -> None:
    schema = CLI_JSON_OUTPUT_SCHEMAS[command_name]
    assert_schema_node(schema, payload, command_name)


def assert_schema_node(schema: object, value: object, path: str) -> None:
    if isinstance(schema, str):
        assert_schema_type(schema, value, path)
        return
    assert isinstance(schema, dict), f"{path}: schema must be a string or object"
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        assert_schema_type(schema_type, value, path)
    if schema.get("required"):
        assert isinstance(value, dict), f"{path}: required fields need object payload"
        for key in schema["required"]:
            assert key in value, f"{path}: missing required field {key}"
    properties = schema.get("properties")
    if isinstance(properties, dict) and isinstance(value, dict):
        for key, child_schema in properties.items():
            if key in value:
                assert_schema_node(child_schema, value[key], f"{path}.{key}")
    item_shape = schema.get("item_shape")
    if item_shape is None:
        return
    if schema_type == "array":
        assert isinstance(value, list), f"{path}: expected array payload"
        if value:
            assert_schema_node(
                {"type": "object", "properties": item_shape},
                value[0],
                f"{path}[0]",
            )
    elif schema_type in {"object", "object|null"} and value is not None:
        assert isinstance(value, dict), f"{path}: expected object payload"
        assert_schema_node({"type": "object", "properties": item_shape}, value, path)


def assert_schema_type(schema_type: str, value: object, path: str) -> None:
    if schema_type.endswith("|null") and value is None:
        return
    if schema_type.startswith("list["):
        assert isinstance(value, list), f"{path}: expected {schema_type}"
        return
    if schema_type == "array":
        assert isinstance(value, list), f"{path}: expected array"
        return
    if schema_type in {"object", "dict"}:
        assert isinstance(value, dict), f"{path}: expected object"
        return
    if schema_type == "object|null":
        assert value is None or isinstance(value, dict), f"{path}: expected object|null"
        return
    if schema_type == "str":
        assert isinstance(value, str), f"{path}: expected str"
        return
    if schema_type == "str|null":
        assert value is None or isinstance(value, str), f"{path}: expected str|null"
        return
    if schema_type == "int":
        assert isinstance(value, int) and not isinstance(value, bool), f"{path}: expected int"
        return
    if schema_type == "float":
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{path}: expected float"
        )
        return
    if schema_type == "float|null":
        assert value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)), (
            f"{path}: expected float|null"
        )
        return
    if schema_type == "bool":
        assert isinstance(value, bool), f"{path}: expected bool"
