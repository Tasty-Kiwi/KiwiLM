"""Reusable neural-network components shared by KiwiLM architectures."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """A length-preserving convolution padded strictly on the left."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            padding=0,
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class GatedCNNBlock(nn.Module):
    """Pre-normalized residual block with a dense causal convolution and GLU."""

    def __init__(
        self,
        d_model: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(
            d_model,
            2 * d_model,
            kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        convolved = self.conv(self.norm(values).transpose(1, 2))
        gated = F.glu(convolved, dim=1).transpose(1, 2)
        return residual + self.dropout(gated)

    def prefill(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Run a sequence and retain normalized history for incremental decoding."""

        normalized = self.norm(values)
        convolved = self.conv(normalized.transpose(1, 2))
        gated = F.glu(convolved, dim=1).transpose(1, 2)
        output = values + self.dropout(gated)
        history_length = self.conv.left_padding
        if history_length:
            history = normalized[:, -history_length:, :]
            if history.shape[1] < history_length:
                history = F.pad(
                    history.transpose(1, 2),
                    (history_length - history.shape[1], 0),
                ).transpose(1, 2)
        else:
            history = normalized[:, :0]
        return output, history

    def decode_step(self, values: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        """Process one position using a cache produced by :meth:`prefill`."""

        if values.ndim != 3 or values.shape[1] != 1:
            raise ValueError("incremental CNN input must have shape [batch, 1, width]")
        normalized = self.norm(values)
        required = self.conv.left_padding
        if history.shape != (values.shape[0], required, values.shape[2]):
            raise ValueError("incremental CNN history has an incompatible shape")
        window = torch.cat((history, normalized), dim=1)
        convolved = self.conv.conv(window.transpose(1, 2))
        gated = F.glu(convolved, dim=1).transpose(1, 2)
        output = values + self.dropout(gated)
        next_history = window[:, -required:, :] if required else window[:, :0]
        return output, next_history


def initialize_weights(module: nn.Module) -> None:
    """Apply the initialization shared by all KiwiLM toy models."""

    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def validate_input_ids(input_ids: Tensor, *, context_length: int) -> None:
    """Validate the common causal-LM input contract."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids sequence length must be positive")
    if input_ids.shape[1] > context_length:
        raise ValueError(
            f"sequence length {input_ids.shape[1]} exceeds context length {context_length}"
        )
