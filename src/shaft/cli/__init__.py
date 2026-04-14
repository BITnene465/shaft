from .registry import COMMAND_REGISTRY, CommandSpec, register_command
from .train import main as train_main

__all__ = ["COMMAND_REGISTRY", "CommandSpec", "register_command", "train_main"]
