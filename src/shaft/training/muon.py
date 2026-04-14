from __future__ import annotations

from typing import Any

import torch


class Muon(torch.optim.Optimizer):
    """A lightweight Muon-style optimizer.

    This implementation keeps momentum-SGD behavior and applies row-wise
    normalization for matrix-shaped updates, which is the practical core
    used in Muon-style optimization for transformer linear layers.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid lr={lr}.")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum={momentum}.")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay={weight_decay}.")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps={eps}.")
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "weight_decay": float(weight_decay),
            "eps": float(eps),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            weight_decay = float(group["weight_decay"])
            eps = float(group["eps"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")

                state: dict[str, Any] = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = grad.detach().clone()
                else:
                    buf.mul_(momentum).add_(grad)

                update = grad.add(buf, alpha=momentum) if nesterov else buf

                # Muon-style normalization on matrix updates.
                if update.ndim >= 2:
                    flat = update.reshape(update.shape[0], -1)
                    row_norm = flat.norm(dim=1, keepdim=True).clamp_min(eps)
                    update = (flat / row_norm).reshape_as(update)

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                p.add_(update, alpha=-lr)

        return loss

