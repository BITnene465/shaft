from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from shaft.cli.common import apply_common_overrides
from shaft.config import RuntimeConfig, load_config_from_text
from shaft.webui.types import ShaftSFTWebUIOverrides


class ShaftWebUIConfigService:
    def read_config_text(self, config_path: str | Path) -> str:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config path not found: {path}")
        return path.read_text(encoding="utf-8")

    def resolve_sft_config(
        self,
        *,
        config_path: str | Path,
        yaml_text: str,
        overrides: ShaftSFTWebUIOverrides | None = None,
    ) -> tuple[RuntimeConfig, str]:
        config = load_config_from_text(yaml_text, config_path=config_path)
        if str(config.algorithm.name).strip().lower() != "sft":
            raise ValueError("Web UI 当前只支持 SFT 训练配置。")
        if overrides is not None:
            config = apply_common_overrides(config, self._build_override_namespace(overrides))
        rendered = yaml.safe_dump(
            asdict(config),
            sort_keys=False,
            allow_unicode=True,
        )
        return config, rendered

    @staticmethod
    def _build_override_namespace(overrides: ShaftSFTWebUIOverrides) -> argparse.Namespace:
        return argparse.Namespace(
            run_id=overrides.run_id,
            seed=overrides.seed,
            epochs=overrides.epochs,
            max_steps=overrides.max_steps,
            learning_rate=overrides.learning_rate,
            train_batch_size=overrides.train_batch_size,
            eval_batch_size=overrides.eval_batch_size,
            mix_strategy=overrides.mix_strategy,
            optimizer_name=overrides.optimizer_name,
            scheduler_name=overrides.scheduler_name,
            scheduler_num_cycles=overrides.scheduler_num_cycles,
            scheduler_power=overrides.scheduler_power,
            loss_name=overrides.loss_name,
            loss_scale=overrides.loss_scale,
            finetune_mode=overrides.finetune_mode,
            lora_r=overrides.lora_r,
            lora_alpha=overrides.lora_alpha,
            lora_dropout=overrides.lora_dropout,
            qlora_load_in_4bit=overrides.qlora_load_in_4bit,
            use_cpu=overrides.use_cpu,
            init_from=overrides.init_from,
            resume_from=overrides.resume_from,
        )
