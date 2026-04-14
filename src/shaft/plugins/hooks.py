from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from transformers import TrainerCallback

from .registry import Registry

HookFn = Callable[[dict[str, Any]], None]
HOOK_EVENTS = {"before_batch", "after_batch", "before_step", "after_step", "on_save"}
HOOK_REGISTRY: Registry[Callable[[], "Hook"] | "Hook"] = Registry("hook")

class Hook(Protocol):
    def before_batch(self, state: dict[str, Any]) -> None: ...

    def after_batch(self, state: dict[str, Any]) -> None: ...

    def before_step(self, state: dict[str, Any]) -> None: ...

    def after_step(self, state: dict[str, Any]) -> None: ...

    def on_save(self, state: dict[str, Any]) -> None: ...


@dataclass
class FunctionHook:
    name: str
    before_batch_fn: HookFn | None = None
    after_batch_fn: HookFn | None = None
    before_step_fn: HookFn | None = None
    after_step_fn: HookFn | None = None
    on_save_fn: HookFn | None = None

    def before_batch(self, state: dict[str, Any]) -> None:
        if self.before_batch_fn is not None:
            self.before_batch_fn(state)

    def after_batch(self, state: dict[str, Any]) -> None:
        if self.after_batch_fn is not None:
            self.after_batch_fn(state)

    def before_step(self, state: dict[str, Any]) -> None:
        if self.before_step_fn is not None:
            self.before_step_fn(state)

    def after_step(self, state: dict[str, Any]) -> None:
        if self.after_step_fn is not None:
            self.after_step_fn(state)

    def on_save(self, state: dict[str, Any]) -> None:
        if self.on_save_fn is not None:
            self.on_save_fn(state)


def hook(event: str, *, name: str | None = None):
    normalized_event = str(event).strip().lower()
    if normalized_event not in HOOK_EVENTS:
        allowed = ", ".join(sorted(HOOK_EVENTS))
        raise ValueError(f"Unsupported hook event {event!r}. Allowed: {allowed}")

    def _decorator(fn: HookFn):
        hook_name = (name or fn.__name__).strip().lower()
        if not hook_name:
            raise ValueError("Hook name cannot be empty.")

        def _builder() -> Hook:
            kwargs = {f"{normalized_event}_fn": fn}
            return FunctionHook(name=hook_name, **kwargs)

        HOOK_REGISTRY.register(hook_name, _builder)
        return fn

    return _decorator


@dataclass
class HookManager:
    hooks: list[Hook]

    def _emit(self, fn_name: str, state: dict[str, Any]) -> None:
        for hook in self.hooks:
            fn = getattr(hook, fn_name, None)
            if callable(fn):
                fn(state)

    def before_batch(self, state: dict[str, Any]) -> None:
        self._emit("before_batch", state)

    def after_batch(self, state: dict[str, Any]) -> None:
        self._emit("after_batch", state)

    def before_step(self, state: dict[str, Any]) -> None:
        self._emit("before_step", state)

    def after_step(self, state: dict[str, Any]) -> None:
        self._emit("after_step", state)

    def on_save(self, state: dict[str, Any]) -> None:
        self._emit("on_save", state)


def build_hook_manager(enabled_hooks: list[str] | None) -> HookManager:
    if not enabled_hooks:
        return HookManager(hooks=[])
    hooks = [HOOK_REGISTRY.create(hook_name) for hook_name in enabled_hooks]
    return HookManager(hooks=hooks)


class TrainerHookCallback(TrainerCallback):
    def __init__(self, hook_manager: HookManager):
        self.hook_manager = hook_manager

    @staticmethod
    def _payload(args, state, control, **kwargs) -> dict[str, Any]:
        return {
            "args": args,
            "trainer_state": state,
            "control": control,
            **kwargs,
        }

    def on_step_begin(self, args, state, control, **kwargs):
        payload = self._payload(args, state, control, **kwargs)
        self.hook_manager.before_batch(payload)
        self.hook_manager.before_step(payload)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        payload = self._payload(args, state, control, **kwargs)
        self.hook_manager.after_step(payload)
        self.hook_manager.after_batch(payload)
        return control

    def on_save(self, args, state, control, **kwargs):
        payload = self._payload(args, state, control, **kwargs)
        self.hook_manager.on_save(payload)
        return control
