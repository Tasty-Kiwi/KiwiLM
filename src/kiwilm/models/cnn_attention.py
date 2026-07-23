"""Causal CNN-attention-CNN language model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kiwilm.config import CNNAttentionConfig, ModelConfig
from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.components import (
    GatedCNNBlock,
    initialize_weights,
    validate_input_ids,
)
from kiwilm.models.registry import register_model


def apply_rotary_embedding(values: Tensor) -> Tensor:
    """Apply rotary position embeddings to ``[batch, heads, time, width]``."""

    sequence_length = values.shape[-2]
    head_dim = values.shape[-1]
    positions = torch.arange(
        sequence_length,
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

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        query = apply_rotary_embedding(split_heads(query))
        key = apply_rotary_embedding(split_heads(key))
        value = split_heads(value)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout_probability if self.training else 0.0,
            is_causal=True,
        )
        merged = (
            attended.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                d_model,
            )
        )
        return self.output_dropout(self.output_projection(merged))


class TransformerAttentionBlock(nn.Module):
    """Pre-normalized causal attention followed by a GELU feed-forward layer."""

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.attention(self.attention_norm(values))
        return values + self.feedforward(self.feedforward_norm(values))


class CNNAttentionLM(CausalLanguageModel):
    """Embedding, 3 gated CNNs, attention, 3 gated CNNs, and an LM head."""

    def __init__(self, config: CNNAttentionConfig | ModelConfig | None = None) -> None:
        super().__init__()
        if config is None:
            attention_config = CNNAttentionConfig()
        elif isinstance(config, CNNAttentionConfig):
            attention_config = config
        elif type(config) is ModelConfig and config.architecture == "cnn_attention":
            attention_config = CNNAttentionConfig.from_dict(config.to_dict())
        else:
            raise TypeError("CNNAttentionLM requires a CNNAttentionConfig")

        self.config = attention_config
        self.token_embedding = nn.Embedding(
            attention_config.vocab_size,
            attention_config.d_model,
        )
        self.pre_attention_blocks = nn.ModuleList(
            GatedCNNBlock(
                attention_config.d_model,
                kernel_size=attention_config.kernel_size,
                dilation=dilation,
                dropout=attention_config.dropout,
            )
            for dilation in attention_config.pre_attention_dilations
        )
        self.attention_block = TransformerAttentionBlock(
            attention_config.d_model,
            num_heads=attention_config.num_heads,
            feedforward_dim=attention_config.feedforward_dim,
            dropout=attention_config.dropout,
        )
        self.post_attention_blocks = nn.ModuleList(
            GatedCNNBlock(
                attention_config.d_model,
                kernel_size=attention_config.kernel_size,
                dilation=dilation,
                dropout=attention_config.dropout,
            )
            for dilation in attention_config.post_attention_dilations
        )
        self.final_norm = nn.LayerNorm(attention_config.d_model)
        self.lm_head = nn.Linear(
            attention_config.d_model,
            attention_config.vocab_size,
            bias=True,
        )
        self.apply(initialize_weights)
        if attention_config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        validate_input_ids(input_ids, context_length=self.config.context_length)

        values = self.token_embedding(input_ids)
        for block in self.pre_attention_blocks:
            values = block(values)
        values = self.attention_block(values)
        for block in self.post_attention_blocks:
            values = block(values)
        return self.lm_head(self.final_norm(values))


def _build_cnn_attention(config: ModelConfig) -> CausalLanguageModel:
    return CNNAttentionLM(config)


register_model("cnn_attention", _build_cnn_attention)
