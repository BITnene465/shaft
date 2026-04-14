from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelArtifacts:
    model: torch.nn.Module
    tokenizer: object
    processor: object

