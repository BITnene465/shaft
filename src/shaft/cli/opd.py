from __future__ import annotations

from .common import add_common_train_args, run_from_args
from .registry import register_command


@register_command("opd")
class OPDCommand:
    help = "Run on-policy direct-loss distillation."

    @classmethod
    def configure_parser(cls, parser) -> None:
        add_common_train_args(parser)

    @classmethod
    def run(cls, args):
        return run_from_args(args, forced_algorithm="opd")
