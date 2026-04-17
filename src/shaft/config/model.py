from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FreezeConfig:
    groups: list[str] = field(default_factory=list)
    prefixes: list[str] = field(default_factory=list)
    regex: str | None = None
    trainable_prefixes: list[str] = field(default_factory=list)
    trainable_regex: str | None = None


@dataclass
class FinetuneConfig:
    mode: str = "full"  # full | lora | dora | qlora
    freeze: FreezeConfig = field(default_factory=FreezeConfig)
    target_modules: list[str] = field(default_factory=lambda: ["auto"])
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    use_rslora: bool = False
    qlora_load_in_4bit: bool = True
    qlora_use_double_quant: bool = True
    qlora_quant_type: str = "nf4"
    qlora_compute_dtype: str = "bfloat16"


@dataclass
class ModelConfig:
    model_type: str = "qwen3vl"
    model_name_or_path: str = "models/Qwen3-VL-4B-Instruct"
    template: str | None = None
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    torch_dtype: str = "bfloat16"
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
