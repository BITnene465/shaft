from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from transformers import TrainerCallback

from .registry import Registry

HookFn = Callable[[dict[str, Any]], None]
HOOK_EVENTS = {"before_batch", "after_batch", "before_step", "after_step", "on_save"}
HOOK_REGISTRY: Registry[Callable[[], "Hook"] | "Hook"] = Registry("hook")

logger = logging.getLogger(__name__)

class Hook(Protocol):
    shaft_trajectory_neutral: bool

    def before_batch(self, state: dict[str, Any]) -> None: ...

    def after_batch(self, state: dict[str, Any]) -> None: ...

    def before_step(self, state: dict[str, Any]) -> None: ...

    def after_step(self, state: dict[str, Any]) -> None: ...

    def on_save(self, state: dict[str, Any]) -> None: ...


@dataclass
class FunctionHook:
    name: str
    shaft_trajectory_neutral: bool = False
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


def hook(
    event: str,
    *,
    name: str | None = None,
    trajectory_neutral: bool = False,
):
    if type(trajectory_neutral) is not bool:
        raise TypeError("hook trajectory_neutral must be a boolean.")
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
            return FunctionHook(
                name=hook_name,
                shaft_trajectory_neutral=trajectory_neutral,
                **kwargs,
            )

        HOOK_REGISTRY.register(hook_name, _builder)
        return fn

    return _decorator


@dataclass
class HookManager:
    hooks: list[Hook]
    _disabled_neutral_hook_ids: set[int] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    @staticmethod
    def _hook_label(hook: Hook) -> str:
        name = inspect.getattr_static(hook, "name", None)
        if type(name) is str and name.strip():
            return name.strip()
        return f"{type(hook).__module__}.{type(hook).__qualname__}"

    def _emit(self, fn_name: str, state: dict[str, Any]) -> None:
        for hook in self.hooks:
            hook_id = id(hook)
            if hook_id in self._disabled_neutral_hook_ids:
                continue
            try:
                fn = getattr(hook, fn_name, None)
                if callable(fn):
                    fn(state)
            except Exception as exc:
                # A trajectory-neutral observer may degrade independently on
                # each rank: it cannot alter training state by contract, and
                # synchronizing every step just to report telemetry failures
                # would add a collective to the hot path. Disable the failed
                # observer locally and warn exactly once. Trajectory-affecting
                # hooks retain fail-fast semantics. Use static lookup so a
                # dynamic property cannot forge neutrality while handling an
                # observer failure.
                neutral = inspect.getattr_static(
                    hook,
                    "shaft_trajectory_neutral",
                    None,
                )
                if neutral is not True:
                    raise
                self._disabled_neutral_hook_ids.add(hook_id)
                logger.warning(
                    "[plugin-disabled] trajectory-neutral hook %s failed during %s; "
                    "this rank will continue without that observer: %s",
                    self._hook_label(hook),
                    fn_name,
                    str(exc) or type(exc).__name__,
                )

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
