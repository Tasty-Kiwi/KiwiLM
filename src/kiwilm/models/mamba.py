"""Portable Mamba-1-style selective state-space components."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kiwilm.models.components import CausalConv1d


class SelectiveStateSpace(nn.Module):
    """Input-dependent diagonal state-space recurrence."""

    def __init__(
        self,
        inner_dim: int,
        *,
        state_dim: int,
        dt_rank: int,
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.state_dim = state_dim
        self.dt_rank = dt_rank
        self.input_projection = nn.Linear(
            inner_dim,
            dt_rank + 2 * state_dim,
            bias=False,
        )
        self.delta_projection = nn.Linear(dt_rank, inner_dim, bias=True)
        initial_a = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.a_log = nn.Parameter(initial_a.log().repeat(inner_dim, 1))
        self.skip = nn.Parameter(torch.ones(inner_dim))

    def reset_parameters(self) -> None:
        """Initialize the continuous-time dynamics in a stable range."""

        nn.init.uniform_(
            self.delta_projection.weight,
            -self.dt_rank**-0.5,
            self.dt_rank**-0.5,
        )
        with torch.no_grad():
            minimum_delta = 0.001
            maximum_delta = 0.1
            log_delta = torch.rand(
                self.inner_dim,
                dtype=self.delta_projection.bias.dtype,
                device=self.delta_projection.bias.device,
            ).mul_(
                math.log(maximum_delta) - math.log(minimum_delta)
            )
            delta = log_delta.add_(math.log(minimum_delta)).exp_()
            inverse_softplus = delta + torch.log(-torch.expm1(-delta))
            self.delta_projection.bias.copy_(inverse_softplus)
            initial_a = torch.arange(
                1,
                self.state_dim + 1,
                dtype=self.a_log.dtype,
                device=self.a_log.device,
            )
            self.a_log.copy_(initial_a.log().repeat(self.inner_dim, 1))
            self.skip.fill_(1.0)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> Tensor:
        """Return an empty float32 recurrent state."""

        return torch.zeros(
            batch_size,
            self.inner_dim,
            self.state_dim,
            dtype=torch.float32,
            device=device,
        )

    def step(
        self,
        values: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Advance one token and return its output and next recurrent state."""

        projected = self.input_projection(values)
        delta_values, input_b, output_c = torch.split(
            projected,
            (self.dt_rank, self.state_dim, self.state_dim),
            dim=-1,
        )
        delta = F.softplus(self.delta_projection(delta_values)).float()
        values_float = values.float()
        input_b = input_b.float()
        output_c = output_c.float()
        continuous_a = -self.a_log.float().exp()
        transition = torch.exp(
            delta.unsqueeze(-1) * continuous_a.unsqueeze(0)
        )
        input_update = (
            delta.unsqueeze(-1)
            * input_b.unsqueeze(1)
            * values_float.unsqueeze(-1)
        )
        next_state = transition * state + input_update
        output = (
            (next_state * output_c.unsqueeze(1)).sum(dim=-1)
            + self.skip.float() * values_float
        )
        return output.to(dtype=values.dtype), next_state

    def forward(
        self,
        values: Tensor,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Scan a full sequence while preserving strict causality."""

        if state is None:
            state = self.initial_state(
                values.shape[0],
                device=values.device,
            )
        outputs = []
        for token_values in values.unbind(dim=1):
            output, state = self.step(token_values, state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state


class MambaBlock(nn.Module):
    """Pre-normalized portable Mamba block with a residual connection."""

    def __init__(
        self,
        d_model: int,
        *,
        inner_dim: int,
        state_dim: int,
        conv_kernel: int,
        dt_rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.input_projection = nn.Linear(d_model, 2 * inner_dim, bias=False)
        self.causal_conv = CausalConv1d(
            inner_dim,
            inner_dim,
            conv_kernel,
            dilation=1,
            groups=inner_dim,
        )
        self.selective_state_space = SelectiveStateSpace(
            inner_dim,
            state_dim=state_dim,
            dt_rank=dt_rank,
        )
        self.output_projection = nn.Linear(inner_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def reset_ssm_parameters(self) -> None:
        """Restore the dynamics initialization after generic model init."""

        self.selective_state_space.reset_parameters()

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        projected, gate = self.input_projection(self.norm(values)).chunk(2, dim=-1)
        convolved = self.causal_conv(projected.transpose(1, 2)).transpose(1, 2)
        scanned, _ = self.selective_state_space(F.silu(convolved))
        output = scanned * F.silu(gate)
        return residual + self.dropout(self.output_projection(output))
