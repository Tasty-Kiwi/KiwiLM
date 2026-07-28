"""Shared RoPE causal self-attention primitive."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def apply_rotary_embedding(values: Tensor, *, position_offset: int = 0) -> Tensor:
    """Apply rotary position embeddings to ``[batch, heads, time, width]``."""

    sequence_length = values.shape[-2]
    head_dim = values.shape[-1]
    positions = torch.arange(
        position_offset,
        position_offset + sequence_length,
        device=values.device,
        dtype=torch.float32,
    )
    frequencies = torch.arange(
        0,
        head_dim,
        2,
        device=values.device,
        dtype=torch.float32,
    )
    inverse_frequencies = 1.0 / (10_000 ** (frequencies / head_dim))
    angles = torch.outer(positions, inverse_frequencies)
    cosines = angles.cos().to(dtype=values.dtype)[None, None, :, :]
    sines = angles.sin().to(dtype=values.dtype)[None, None, :, :]
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = torch.stack(
        (even * cosines - odd * sines, odd * cosines + even * sines),
        dim=-1,
    )
    return rotated.flatten(start_dim=-2)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using RoPE and PyTorch SDPA."""

    def __init__(self, d_model: int, *, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_probability = dropout
        self.qkv_projection = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        batch_size, sequence_length, d_model = values.shape
        query, key, value = self.qkv_projection(values).chunk(3, dim=-1)
        query = apply_rotary_embedding(
            self._split_heads(query, batch_size, sequence_length)
        )
        key = apply_rotary_embedding(
            self._split_heads(key, batch_size, sequence_length)
        )
        value = self._split_heads(value, batch_size, sequence_length)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout_probability if self.training else 0.0,
            is_causal=True,
        )
        return self._merge_heads(attended, d_model)

    def prefill(self, values: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Attend over a prompt and return its rotated key/value cache."""

        batch_size, sequence_length, d_model = values.shape
        query, key, value = self.qkv_projection(values).chunk(3, dim=-1)
        query = apply_rotary_embedding(
            self._split_heads(query, batch_size, sequence_length)
        )
        key = apply_rotary_embedding(
            self._split_heads(key, batch_size, sequence_length)
        )
        value = self._split_heads(value, batch_size, sequence_length)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
        )
        return self._merge_heads(attended, d_model), (key, value)

    def decode_step(
        self,
        values: Tensor,
        cache: tuple[Tensor, Tensor],
        *,
        position: int,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Attend one position over cached past keys and values."""

        batch_size, sequence_length, d_model = values.shape
        if sequence_length != 1:
            raise ValueError("incremental attention input must contain one position")
        query, key, value = self.qkv_projection(values).chunk(3, dim=-1)
        query = apply_rotary_embedding(
            self._split_heads(query, batch_size, 1),
            position_offset=position,
        )
        key = apply_rotary_embedding(
            self._split_heads(key, batch_size, 1),
            position_offset=position,
        )
        value = self._split_heads(value, batch_size, 1)
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise ValueError(
                "incremental attention cache has an incompatible structure"
            )
        cached_key, cached_value = cache
        expected_shape = (
            batch_size,
            self.num_heads,
            position,
            self.head_dim,
        )
        if cached_key.shape != expected_shape or cached_value.shape != expected_shape:
            raise ValueError("incremental attention cache has an incompatible shape")
        key = torch.cat((cached_key, key), dim=2)
        value = torch.cat((cached_value, value), dim=2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        return self._merge_heads(attended, d_model), (key, value)

    def _split_heads(
        self,
        values: Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> Tensor:
        return values.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _merge_heads(self, values: Tensor, d_model: int) -> Tensor:
        batch_size, _, sequence_length, _ = values.shape
        merged = (
            values.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, d_model)
        )
        return self.output_dropout(self.output_projection(merged))


__all__ = ["CausalSelfAttention", "apply_rotary_embedding"]
