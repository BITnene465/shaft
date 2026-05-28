from __future__ import annotations

from pathlib import Path

from shaft.prompting import load_prompt_pool, load_prompt_template


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
