from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InferGenerationConfig:
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


@dataclass
class InferModelConfig:
    model_type: str = "qwen3vl"
    model_name_or_path: str = "models/Qwen3-VL-4B-Instruct"
    template: str | None = None
    trust_remote_code: bool = True
    attn_implementation: str | None = "flash_attention_2"
    torch_dtype: str = "bfloat16"
    finetune_mode: str = "full"
    device: str | None = None
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


@dataclass
class InferPipelineConfig:
    engines: dict[str, InferModelConfig] = field(default_factory=dict)
    stages: list[InferStageConfig] = field(default_factory=list)
