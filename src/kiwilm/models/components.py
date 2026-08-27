"""Reusable neural-network helpers for KiwiLM 2."""

from __future__ import annotations

from torch import Tensor, nn


def initialize_weights(module: nn.Module) -> None:
    """Apply the initialization shared by the KiwiLM 2 variants."""

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
