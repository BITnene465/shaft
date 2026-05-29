from __future__ import annotations

from dataclasses import dataclass, field


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
