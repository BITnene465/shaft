from __future__ import annotations

from dataclasses import dataclass, field
import re

from shaft.prompting import ShaftPromptProgram, compile_prompt, validate_prompt_text


@dataclass
class InferGenerationConfig:
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0


@dataclass
class InferEngineConfig:
    model_type: str = "qwen3vl"
    model_name_or_path: str = "models/Qwen3-VL-4B-Instruct"
    template: str | None = None
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    torch_dtype: str = "bfloat16"
    load_mode: str = "full"
    backend: str = "hf_local"
    endpoint: str | None = None
    api_key: str | None = None
    served_model_name: str | None = None
    request_timeout_seconds: float = 60.0
    device: str | None = None
    device_map: str | dict[str, str] | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None
    generation: InferGenerationConfig = field(default_factory=InferGenerationConfig)


@dataclass
class InferStageConfig:
    name: str
    engine: str
    user_prompt_template: str
    arguments: dict[str, object] = field(default_factory=dict)
    output_key: str | None = None
    system_prompt: str = ""
    generation: InferGenerationConfig | None = None
    codec: str = "text"
    min_pixels: int | None = None
    max_pixels: int | None = None
    backend_options: dict[str, object] = field(default_factory=dict)
    max_retries: int = 0
    retry_backoff_seconds: float = 0.0
    fail_fast: bool = True
    timeout_seconds: float | None = None


@dataclass
class InferPipelineConfig:
    engines: dict[str, InferEngineConfig] = field(default_factory=dict)
    stages: list[InferStageConfig] = field(default_factory=list)


def compile_stage_prompt(stage: InferStageConfig, *, source: str) -> ShaftPromptProgram:
    if not isinstance(stage.user_prompt_template, str):
        raise ValueError(f"{source}.user_prompt_template must be a string.")
    validate_prompt_text(stage.system_prompt, source=f"{source}.system_prompt")
    legacy = re.search(
        r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*):\{[A-Za-z_][A-Za-z0-9_]*\}\}(?!\})",
        stage.user_prompt_template,
    ) or re.search(
        r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)(?:[.!:\[][^{\n}]*)?\}(?!\})",
        stage.user_prompt_template,
    )
    if legacy is not None:
        name = legacy.group(1)
        raise ValueError(
            f"{source}.user_prompt_template uses legacy placeholder {{{name}}}; "
            f"declare arguments.{name} and use double braces: "
            f"{{{{ {name} }}}} or {{{{ {name} | json }}}}."
        )
    if "{{" in stage.system_prompt:
        raise ValueError(f"{source}.system_prompt must be static.")
    program = compile_prompt(
        stage.user_prompt_template,
        arguments=stage.arguments,
        source=f"{source}.user_prompt_template",
    )
    declared = set(program.schema.names)
    referenced = set(program.referenced_arguments)
    unused = sorted(declared - referenced)
    if unused:
        raise ValueError(f"{source}.arguments contains unused fields: {unused}.")
    return program
