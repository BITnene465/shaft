from __future__ import annotations

from .common import add_common_train_args, run_from_args
from .registry import register_command


@register_command("rlhf")
class RLHFCommand:
    help = "Run RLHF training (DPO/PPO)."

    @classmethod
    def configure_parser(cls, parser) -> None:
        add_common_train_args(parser)
        parser.add_argument("--algorithm", choices=["dpo", "ppo"], required=True)

    @classmethod
    def run(cls, args):
        return run_from_args(args, allowed_algorithms={"dpo", "ppo"})
