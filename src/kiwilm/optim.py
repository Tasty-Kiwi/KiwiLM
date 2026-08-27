"""Optimizer construction for controlled AdamW and Muon experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


def zeroth_power_via_newton_schulz(gradient: Tensor, steps: int = 5) -> Tensor:
    """Approximate the polar factor used by Muon in float32."""

    if gradient.ndim != 2:
        raise ValueError("Muon orthogonalization requires a matrix")
    matrix = gradient.float()
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.mT
    matrix = matrix / (matrix.norm() + 1e-7)
    # Quintic iteration used by the reference Muon implementation. It favors a
    # useful update spectrum after five steps over exact late convergence.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = matrix @ matrix.mT
        matrix = a * matrix + (b * gram + c * (gram @ gram)) @ matrix
    if transposed:
        matrix = matrix.mT
    return matrix.to(gradient.dtype)


class MuonWithAuxAdamW(Optimizer):
    """Muon for selected matrices and AdamW for all auxiliary parameters."""

    def __init__(
        self,
        muon_parameters: Iterable[nn.Parameter],
        adamw_parameters: Iterable[nn.Parameter],
        *,
        muon_lr: float,
        adamw_lr: float,
        weight_decay: float,
        beta2: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        if muon_lr <= 0 or adamw_lr <= 0:
            raise ValueError("optimizer learning rates must be positive")
        groups = [
            {
                "params": list(muon_parameters),
                "algorithm": "muon",
                "lr": muon_lr,
                "lr_multiplier": muon_lr / adamw_lr,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
                "weight_decay": weight_decay,
            },
            {
                "params": list(adamw_parameters),
                "algorithm": "adamw",
                "lr": adamw_lr,
                "lr_multiplier": 1.0,
                "betas": (0.9, beta2),
                "eps": 1e-8,
                "weight_decay": weight_decay,
            },
        ]
        if not groups[0]["params"]:
            raise ValueError("Muon requires at least one dense matrix parameter")
        super().__init__(groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[no-untyped-def]
        loss = None if closure is None else closure()
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    def _step_muon(self, group: dict[str, object]) -> None:
        lr = float(group["lr"])
        momentum = float(group["momentum"])
        weight_decay = float(group["weight_decay"])
        for parameter in group["params"]:  # type: ignore[assignment]
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if gradient.ndim != 2:
                raise RuntimeError("a non-matrix parameter reached the Muon group")
            state = self.state[parameter]
            buffer = state.get("momentum_buffer")
            if buffer is None:
                buffer = torch.zeros_like(gradient)
                state["momentum_buffer"] = buffer
            buffer.mul_(momentum).add_(gradient)
            update = gradient.add(buffer, alpha=momentum) if group["nesterov"] else buffer
            update = zeroth_power_via_newton_schulz(update, steps=int(group["ns_steps"]))
            scale = math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
            if weight_decay:
                parameter.mul_(1.0 - lr * weight_decay)
            parameter.add_(update, alpha=-lr * scale)

    def _step_adamw(self, group: dict[str, object]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]  # type: ignore[misc]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        for parameter in group["params"]:  # type: ignore[assignment]
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            state = self.state[parameter]
            state["step"] = int(state.get("step", 0)) + 1
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg is None or exp_avg_sq is None:
                exp_avg = torch.zeros_like(gradient)
                exp_avg_sq = torch.zeros_like(gradient)
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
            exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            step = state["step"]
            denominator = exp_avg_sq.sqrt().div_(math.sqrt(1.0 - beta2**step)).add_(eps)
            step_size = lr / (1.0 - beta1**step)
            if weight_decay:
                parameter.mul_(1.0 - lr * weight_decay)
            parameter.addcdiv_(exp_avg, denominator, value=-step_size)


def split_muon_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Select dense Linear weights while excluding tied embeddings and tables."""

    embedding_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    muon_ids: set[int] = set()
    for module in model.modules():
        if (
            isinstance(module, nn.Linear)
            and module.weight.requires_grad
            and id(module.weight) not in embedding_ids
        ):
            muon_ids.add(id(module.weight))
    muon: list[nn.Parameter] = []
    adamw: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        (muon if parameter_id in muon_ids else adamw).append(parameter)
    return muon, adamw


__all__ = [
    "MuonWithAuxAdamW",
    "split_muon_parameters",
    "zeroth_power_via_newton_schulz",
]
