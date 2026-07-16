from __future__ import annotations

from pathlib import Path

import pytest

from shaft.prompting import (
    ShaftPromptArgument,
    ShaftPromptSchema,
    compile_prompt,
    load_prompt_pool,
    load_prompt_template,
)


def test_load_prompt_template_reads_pool_main_variant(tmp_path: Path) -> None:
    prompt_path = tmp_path / "pool.yaml"
    prompt_path.write_text(
        "\n".join(
            [
                "metadata:",
                "  id: prompt.pool",
                "  version: test-version",
                "  target_labels: [arrow]",
                "prompts:",
                "  - id: main",
                "    system_prompt: system",
                "    user_prompt: user",
                "  - id: alternate",
                "    system_prompt: alternate system",
                "    user_prompt: alternate user",
                "",
            ]
        ),
        encoding="utf-8",
    )

    prompt = load_prompt_template(prompt_path)

    assert prompt.prompt_id == "prompt.pool.main"
    assert prompt.system_prompt == "system"
    assert prompt.user_prompt == "user"
    assert prompt.version == "test-version"
    assert prompt.variant_id == "main"
    assert prompt.metadata["prompt_pool_id"] == "prompt.pool"
    assert prompt.metadata["prompt_variant_id"] == "main"


def test_load_prompt_pool_reads_all_variants(tmp_path: Path) -> None:
    prompt_path = tmp_path / "pool.yaml"
    prompt_path.write_text(
        "\n".join(
            [
                "metadata:",
                "  id: prompt.pool",
                "  version: test-version",
                "prompts:",
                "  - id: main",
                "    system_prompt: system",
                "    user_prompt: user",
                "  - id: alternate",
                "    system_prompt: alternate system",
                "    user_prompt: alternate user",
                "",
            ]
        ),
        encoding="utf-8",
    )

    prompts = load_prompt_pool(prompt_path)

    assert [prompt.prompt_id for prompt in prompts] == [
        "prompt.pool.main",
        "prompt.pool.alternate",
    ]
    assert [prompt.variant_id for prompt in prompts] == ["main", "alternate"]
    assert {prompt.version for prompt in prompts} == {"test-version"}
    assert [prompt.sampling_weight for prompt in prompts] == [1.0, 1.0]


def test_load_prompt_pool_reads_sampling_weights(tmp_path: Path) -> None:
    prompt_path = tmp_path / "weighted.yaml"
    prompt_path.write_text(
        """
metadata:
  id: prompt.weighted
  version: test-version
prompts:
  - id: disabled
    sampling_weight: 0
    user_prompt: disabled
  - id: active
    sampling_weight: 3.5
    user_prompt: active
""".strip()
        + "\n",
        encoding="utf-8",
    )

    prompts = load_prompt_pool(prompt_path)

    assert [prompt.sampling_weight for prompt in prompts] == [0.0, 3.5]


def test_load_prompt_template_keeps_legacy_single_prompt_compatible(tmp_path: Path) -> None:
    prompt_path = tmp_path / "single.yaml"
    prompt_path.write_text(
        "\n".join(
            [
                "metadata:",
                "  id: prompt.single",
                "prompt:",
                "  system_prompt: system",
                "  user_prompt: user",
                "",
            ]
        ),
        encoding="utf-8",
    )

    prompt = load_prompt_template(prompt_path)

    assert prompt.prompt_id == "prompt.single"
    assert prompt.system_prompt == "system"
    assert prompt.user_prompt == "user"
    assert prompt.variant_id is None
    assert prompt.version is None


def test_compiled_prompt_renders_restricted_variables_and_canonical_json() -> None:
    prompt = compile_prompt(
        'Draw {{ label }} at {{ bbox | json }} with {{ payload | json }}.',
        arguments={
            "label": {"type": "enum", "values": ["shape", "line"]},
            "bbox": {"type": "bbox_2d_0_999"},
            "payload": {"type": "json"},
        },
        source="unit-test",
    )

    rendered = prompt.render(
        {
            "label": "shape",
            "bbox": [10, 20, 300, 400],
            "payload": {"z": "中文", "a": True},
        }
    )

    assert rendered == 'Draw shape at [10,20,300,400] with {"a":true,"z":"中文"}.'


@pytest.mark.parametrize(
    ("template", "match"),
    [
        ("{{ item.upper }}", "Unsupported prompt expression"),
        ("{{ item | upper }}", "Unsupported prompt expression"),
        ("{{ item", "Unclosed prompt expression"),
    ],
)
def test_compiled_prompt_rejects_unsupported_template_syntax(
    template: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        compile_prompt(
            template,
            arguments={"item": {"type": "string"}},
            source="bad-template",
        )


def test_compiled_prompt_rejects_missing_extra_and_invalid_bbox_values() -> None:
    prompt = compile_prompt(
        "bbox={{ bbox | json }}",
        arguments={"bbox": {"type": "bbox_2d_0_999"}},
        source="bbox-test",
    )

    with pytest.raises(ValueError, match="Missing prompt arguments.*bbox"):
        prompt.render({})
    with pytest.raises(ValueError, match="Unexpected prompt arguments.*other"):
        prompt.render({"bbox": [0, 0, 1, 1], "other": 1})
    with pytest.raises(ValueError, match="bbox_2d_0_999"):
        prompt.render({"bbox": [0, 20, 10, 10]})


def test_compiled_prompt_rejects_plain_interpolation_for_non_text_argument() -> None:
    with pytest.raises(ValueError, match="must use the json filter"):
        compile_prompt(
            "bbox={{ bbox }}",
            arguments={"bbox": {"type": "bbox_2d_0_999"}},
            source="bbox-test",
        )


def test_load_prompt_pool_supports_shared_arguments_and_mixed_variants(tmp_path: Path) -> None:
    prompt_path = tmp_path / "dynamic.yaml"
    prompt_path.write_text(
        """
metadata:
  id: prompt.dynamic
  version: v1
arguments:
  label:
    type: enum
    values: [shape, line]
prompts:
  - id: detailed
    user_prompt_template: "Reconstruct {{ label }}."
  - id: static
    user_prompt: "Inspect the target."
""".strip()
        + "\n",
        encoding="utf-8",
    )

    prompts = load_prompt_pool(prompt_path)

    assert prompts[0].render({"label": "shape"}) == "Reconstruct shape."
    assert prompts[1].render({"label": "shape"}) == "Inspect the target."


@pytest.mark.parametrize(
    "variant",
    [
        'user_prompt: static\n    user_prompt_template: "{{ label }}"',
        "system_prompt: '{{ label }}'\n    user_prompt: static",
        "user_prompt_template: no-placeholder",
    ],
)
def test_load_prompt_pool_rejects_ambiguous_or_invalid_dynamic_variants(
    tmp_path: Path,
    variant: str,
) -> None:
    prompt_path = tmp_path / "bad-pool.yaml"
    prompt_path.write_text(
        f"""
metadata:
  id: prompt.bad
  version: v1
arguments:
  label:
    type: string
prompts:
  - id: main
    {variant}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_prompt_pool(prompt_path)


def test_load_prompt_pool_accepts_nested_json_in_static_prompt(tmp_path: Path) -> None:
    prompt_path = tmp_path / "static-json-pool.yaml"
    prompt_path.write_text(
        """
metadata:
  id: prompt.static-json
  version: v1
prompts:
  - id: main
    system_prompt: Return JSON only.
    user_prompt: '{"type":"shape","parameters":{"shape_type":"rectangle"}}'
""".strip()
        + "\n",
        encoding="utf-8",
    )

    prompts = load_prompt_pool(prompt_path)

    assert prompts[0].user_prompt.endswith('"rectangle"}}')


def test_compiled_prompt_treats_closing_braces_without_an_expression_as_literal() -> None:
    prompt = compile_prompt('Return {"parameters":{"shape_type":"rectangle"}}')

    assert prompt.render({}) == 'Return {"parameters":{"shape_type":"rectangle"}}'


@pytest.mark.parametrize("value", [(1, 2), {1: "bad"}, float("nan")])
def test_json_filter_rejects_non_json_python_values(value: object) -> None:
    prompt = compile_prompt(
        "{{ value | json }}",
        arguments={"value": {"type": "json"}},
    )

    with pytest.raises(ValueError, match="Prompt"):
        prompt.render({"value": value})


def test_rendered_prompt_must_not_be_empty_and_audit_is_deterministic() -> None:
    prompt = compile_prompt(
        "{{ value }}",
        arguments={"value": {"type": "string"}},
    )

    with pytest.raises(ValueError, match="must not be empty"):
        prompt.render({"value": ""})
    first = prompt.render_with_audit({"value": "stable"})
    second = prompt.render_with_audit({"value": "stable"})
    assert first == second


def test_prompt_program_fingerprint_is_canonical_and_schema_sensitive() -> None:
    first = compile_prompt(
        "{{ value }}",
        arguments={
            "unused": {"type": "integer"},
            "value": {"type": "string"},
        },
    )
    reordered = compile_prompt(
        "{{ value }}",
        arguments={
            "value": {"type": "string"},
            "unused": {"type": "integer"},
        },
    )
    enum = compile_prompt(
        "{{ value }}",
        arguments={"value": {"type": "enum", "values": ["x"]}},
    )

    assert first.program_sha256 == reordered.program_sha256
    assert first.program_sha256 != enum.program_sha256


def test_compiled_enum_is_deeply_immutable_and_string_only() -> None:
    values = ["a", "b"]
    arguments = {"value": {"type": "enum", "values": values}}
    prompt = compile_prompt("{{ value }}", arguments=arguments)
    values.append("c")

    with pytest.raises(ValueError, match="type 'enum'"):
        prompt.render({"value": "c"})
    with pytest.raises(ValueError, match="values must be strings"):
        compile_prompt(
            "{{ value | json }}",
            arguments={"value": {"type": "enum", "values": [{"x": 1}]}},
        )


def test_public_prompt_schema_constructors_snapshot_mutable_inputs() -> None:
    enum_values = ["a"]
    argument = ShaftPromptArgument(
        name="value",
        type="enum",
        enum_values=enum_values,  # type: ignore[arg-type]
    )
    schema_arguments = [argument]
    schema = ShaftPromptSchema(arguments=schema_arguments)  # type: ignore[arg-type]
    prompt = compile_prompt("{{ value }}", arguments=schema)

    enum_values.append("b")
    schema_arguments.append(ShaftPromptArgument(name="other", type="string"))

    assert prompt.render({"value": "a"}) == "a"
    assert prompt.schema.names == ("value",)
    with pytest.raises(ValueError, match="type 'enum'"):
        prompt.render({"value": "b"})


@pytest.mark.parametrize("value", ["\ud800", {"\ud800": "x"}])
def test_prompt_json_rejects_invalid_unicode_scalars(value: object) -> None:
    prompt = compile_prompt(
        "{{ value | json }}",
        arguments={"value": {"type": "json"}},
    )

    with pytest.raises(ValueError, match="UTF-8"):
        prompt.render({"value": value})


def test_prompt_template_and_pool_system_reject_invalid_unicode_scalars(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        compile_prompt("\ud800")

    prompt_path = tmp_path / "invalid-unicode.yaml"
    prompt_path.write_text(
        'metadata: {id: prompt.invalid-unicode, version: v1}\n'
        'prompts:\n'
        '  - id: main\n'
        '    system_prompt: "\\uD800"\n'
        '    user_prompt: ok\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="UTF-8"):
        load_prompt_pool(prompt_path)


def test_prompt_pool_rejects_non_string_static_text_and_dynamic_user_access(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        """
metadata: {id: prompt.invalid, version: v1}
prompts:
  - id: main
    user_prompt: [not, text]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="user_prompt must be a string"):
        load_prompt_pool(invalid)

    dynamic = tmp_path / "dynamic-only.yaml"
    dynamic.write_text(
        """
metadata: {id: prompt.dynamic-only, version: v1}
arguments:
  value: {type: string}
prompts:
  - id: main
    user_prompt_template: "{{ value }}"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    variant = load_prompt_template(dynamic)
    with pytest.raises(ValueError, match="call render"):
        _ = variant.user_prompt
