from .registry import COMMAND_REGISTRY, CommandSpec, register_command


def train_main(argv=None):
    from .train import main

    return main(argv)


def export_main(argv=None):
    from .export import main

    return main(argv)


def infer_main(argv=None):
    from .infer import main

    return main(argv)


def web_main(argv=None):
    from .web import main

    return main(argv)

__all__ = [
    "COMMAND_REGISTRY",
    "CommandSpec",
    "register_command",
    "train_main",
    "export_main",
    "infer_main",
    "web_main",
]
