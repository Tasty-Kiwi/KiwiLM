"""KiwiLM-SAN: a deep attention-only causal language model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kiwilm.config import KiwiLMSANConfig, ModelConfig
from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.components import initialize_weights, validate_input_ids
from kiwilm.models.registry import register_model


class ZeroCenteredRMSNorm(nn.Module):
    """RMS normalization with a learnable ``1 + gamma`` gain."""

    def __init__(self, width: int, *, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(width))

    def forward(self, values: Tensor) -> Tensor:
        inverse_rms = torch.rsqrt(values.float().square().mean(dim=-1, keepdim=True) + self.eps).to(
            dtype=values.dtype
        )
        gain = (1.0 + self.weight).to(dtype=values.dtype)
        return values * inverse_rms * gain


def _apply_rotary_embedding(
    values: Tensor,
    *,
    rope_base: float,
    position_offset: int = 0,
) -> Tensor:
    """Apply RoPE to a ``[batch, heads, time, width]`` tensor."""

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
    inverse_frequencies = 1.0 / (rope_base ** (frequencies / head_dim))
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


class GroupedQueryAttention(nn.Module):
    """Bias-free RoPE GQA with zero-centered per-head Q/K normalization."""

    def __init__(
        self,
        d_model: int,
        *,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float,
        rms_norm_eps: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.d_model = d_model
        self.dropout_probability = dropout
        self.rope_base = rope_base
        kv_width = num_kv_heads * self.head_dim
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, kv_width, bias=False)
        self.value_projection = nn.Linear(d_model, kv_width, bias=False)
        self.query_norm = ZeroCenteredRMSNorm(
            self.head_dim,
            eps=rms_norm_eps,
        )
        self.key_norm = ZeroCenteredRMSNorm(
            self.head_dim,
            eps=rms_norm_eps,
        )
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, values: Tensor) -> Tensor:
        query, key, value = self._project(values, position_offset=0)
        attended = self._attention(
            query,
            key,
            value,
            is_causal=True,
            dropout_p=self.dropout_probability if self.training else 0.0,
        )
        return self.output_projection(self._merge_query_heads(attended))

    def prefill(self, values: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Attend over a prompt and retain compact rotated KV heads."""

        query, key, value = self._project(values, position_offset=0)
        attended = self._attention(
            query,
            key,
            value,
            is_causal=True,
            dropout_p=0.0,
        )
        output = self.output_projection(self._merge_query_heads(attended))
        return output, (key, value)

    def decode_step(
        self,
        values: Tensor,
        cache: tuple[Tensor, Tensor],
        *,
        position: int,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Attend one query over compact cached KV heads."""

        if values.ndim != 3 or values.shape[1] != 1:
            raise ValueError("incremental attention input must contain one position")
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise ValueError("incremental attention cache has an incompatible structure")
        cached_key, cached_value = cache
        expected_shape = (
            values.shape[0],
            self.num_kv_heads,
            position,
            self.head_dim,
        )
        if (
            not isinstance(cached_key, Tensor)
            or not isinstance(cached_value, Tensor)
            or cached_key.shape != expected_shape
            or cached_value.shape != expected_shape
        ):
            raise ValueError("incremental attention cache has an incompatible shape")

        query, key, value = self._project(values, position_offset=position)
        key = torch.cat((cached_key, key), dim=2)
        value = torch.cat((cached_value, value), dim=2)
        attended = self._attention(
            query,
            key,
            value,
            is_causal=False,
            dropout_p=0.0,
        )
        output = self.output_projection(self._merge_query_heads(attended))
        return output, (key, value)

    def _project(
        self,
        values: Tensor,
        *,
        position_offset: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, sequence_length, _ = values.shape
        query = self._split_heads(
            self.query_projection(values),
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_heads=self.num_query_heads,
        )
        key = self._split_heads(
            self.key_projection(values),
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_heads=self.num_kv_heads,
        )
        value = self._split_heads(
            self.value_projection(values),
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_heads=self.num_kv_heads,
        )
        query = _apply_rotary_embedding(
            self.query_norm(query),
            rope_base=self.rope_base,
            position_offset=position_offset,
        )
        key = _apply_rotary_embedding(
            self.key_norm(key),
            rope_base=self.rope_base,
            position_offset=position_offset,
        )
        return query, key, value

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        is_causal: bool,
        dropout_p: float,
    ) -> Tensor:
        if self.num_query_heads == self.num_kv_heads:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        if query.device.type == "cuda":
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=is_causal,
                enable_gqa=True,
            )

        repeats = self.num_query_heads // self.num_kv_heads
        expanded_key = key.repeat_interleave(repeats, dim=1)
        expanded_value = value.repeat_interleave(repeats, dim=1)
        return F.scaled_dot_product_attention(
            query,
            expanded_key,
            expanded_value,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    def _split_heads(
        self,
        values: Tensor,
        *,
        batch_size: int,
        sequence_length: int,
        num_heads: int,
    ) -> Tensor:
        return values.view(
            batch_size,
            sequence_length,
            num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _merge_query_heads(self, values: Tensor) -> Tensor:
        batch_size, _, sequence_length, _ = values.shape
        return values.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)


class KiwiLMSANBlock(nn.Module):
    """Pre-norm GQA with sandwich normalization and a gated residual."""

    def __init__(
        self,
        d_model: int,
        *,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float,
        rms_norm_eps: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.pre_norm = ZeroCenteredRMSNorm(d_model, eps=rms_norm_eps)
        self.attention = GroupedQueryAttention(
            d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            rms_norm_eps=rms_norm_eps,
            rope_base=rope_base,
        )
        self.sandwich_norm = ZeroCenteredRMSNorm(d_model, eps=rms_norm_eps)
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        update = self.attention(self.pre_norm(values))
        update = self.dropout(self.sandwich_norm(update))
        return values + torch.sigmoid(self.residual_gate) * update

    def prefill(self, values: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        update, cache = self.attention.prefill(self.pre_norm(values))
        update = self.dropout(self.sandwich_norm(update))
        return values + torch.sigmoid(self.residual_gate) * update, cache

    def decode_step(
        self,
        values: Tensor,
        cache: tuple[Tensor, Tensor],
        *,
        position: int,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        update, cache = self.attention.decode_step(
            self.pre_norm(values),
            cache,
            position=position,
        )
        update = self.dropout(self.sandwich_norm(update))
        return values + torch.sigmoid(self.residual_gate) * update, cache


@dataclass(slots=True)
class KiwiLMSANCache:
    """Raw token window and one compact GQA KV cache per SAN block."""

    token_ids: Tensor
    attention: list[tuple[Tensor, Tensor]]


class KiwiLMSANLM(CausalLanguageModel):
    """Deep Simple Attention Network adapted to KiwiLM's model interface."""

    def __init__(self, config: KiwiLMSANConfig | None = None) -> None:
        super().__init__()
        model_config = config or KiwiLMSANConfig()
        if not isinstance(model_config, KiwiLMSANConfig):
            raise TypeError("KiwiLMSANLM requires a KiwiLMSANConfig")

        self.config = model_config
        self.embedding_scale = math.sqrt(model_config.d_model)
        self.token_embedding = nn.Embedding(
            model_config.vocab_size,
            model_config.d_model,
        )
        self.blocks = nn.ModuleList(
            KiwiLMSANBlock(
                model_config.d_model,
                num_query_heads=model_config.num_query_heads,
                num_kv_heads=model_config.num_kv_heads,
                dropout=model_config.dropout,
                rms_norm_eps=model_config.rms_norm_eps,
                rope_base=model_config.rope_base,
            )
            for _ in range(model_config.num_layers)
        )
        self.final_norm = ZeroCenteredRMSNorm(
            model_config.d_model,
            eps=model_config.rms_norm_eps,
        )
        self.lm_head = nn.Linear(
            model_config.d_model,
            model_config.vocab_size,
            bias=True,
        )
        self.apply(initialize_weights)
        output_std = 0.02 / math.sqrt(2 * model_config.num_layers)
        for block in self.blocks:
            nn.init.normal_(
                block.attention.output_projection.weight,
                mean=0.0,
                std=output_std,
            )
        if model_config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids) * self.embedding_scale
        for block in self.blocks:
            values = block(values)
        return self.lm_head(self.final_norm(values))

    def prefill(
        self,
        input_ids: Tensor,
    ) -> tuple[Tensor, KiwiLMSANCache]:
        input_ids = input_ids[:, -self.config.context_length :]
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids) * self.embedding_scale
        attention_caches: list[tuple[Tensor, Tensor]] = []
        for block in self.blocks:
            values, attention_cache = block.prefill(values)
            attention_caches.append(attention_cache)
        return self.lm_head(self.final_norm(values)), KiwiLMSANCache(
            token_ids=input_ids,
            attention=attention_caches,
        )

    def decode_step(
        self,
        input_ids: Tensor,
        cache: KiwiLMSANCache,
    ) -> tuple[Tensor, KiwiLMSANCache]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step input must have shape [batch, 1]")
        if not isinstance(cache, KiwiLMSANCache):
            raise ValueError("incremental cache has an incompatible structure")
        if (
            cache.token_ids.ndim != 2
            or cache.token_ids.shape[1] < 1
            or cache.token_ids.shape[1] > self.config.context_length
        ):
            raise ValueError("incremental cache has an incompatible token window")
        if cache.token_ids.shape[0] != input_ids.shape[0]:
            raise ValueError("decode_step batch size differs from the cache")
        if len(cache.attention) != len(self.blocks):
            raise ValueError("incremental cache has an incompatible structure")

        token_ids = torch.cat((cache.token_ids, input_ids), dim=1)
        if token_ids.shape[1] > self.config.context_length:
            return self.prefill(token_ids[:, -self.config.context_length :])

        values = self.token_embedding(input_ids) * self.embedding_scale
        position = cache.token_ids.shape[1]
        attention_caches: list[tuple[Tensor, Tensor]] = []
        for block, attention_cache in zip(
            self.blocks,
            cache.attention,
            strict=True,
        ):
            values, attention_cache = block.decode_step(
                values,
                attention_cache,
                position=position,
            )
            attention_caches.append(attention_cache)
        return self.lm_head(self.final_norm(values)), KiwiLMSANCache(
            token_ids=token_ids,
            attention=attention_caches,
        )


def _build_kiwilm_san(config: ModelConfig) -> CausalLanguageModel:
    if not isinstance(config, KiwiLMSANConfig):
        raise TypeError("kiwilm_san requires KiwiLMSANConfig")
    return KiwiLMSANLM(config)


register_model("kiwilm_san", _build_kiwilm_san)


__all__ = [
    "GroupedQueryAttention",
    "KiwiLMSANBlock",
    "KiwiLMSANCache",
    "KiwiLMSANLM",
    "ZeroCenteredRMSNorm",
]
