"""Causal CNN-attention-CNN language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kiwilm.config import (
    CNNAttentionConfig,
    CNNAttentionMambaConfig,
    CNNDeepInterleavedAttentionConfig,
    CNNDualAttentionConfig,
    CNNFFNAttentionConfig,
    CNNInterleavedAttentionConfig,
    ModelConfig,
)
from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.components import (
    GatedCNNBlock,
    initialize_weights,
    validate_input_ids,
)
from kiwilm.models.legacy.mamba import MambaBlock
from kiwilm.models.registry import register_model


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
            query, key, value, dropout_p=0.0, is_causal=True
        )
        return self._merge_heads(attended, d_model), (key, value)

    def decode_step(
        self,
        values: Tensor,
        cache: tuple[Tensor, Tensor],
        *,
        position: int,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Attend one query over cached past keys and values."""

        batch_size, sequence_length, d_model = values.shape
        if sequence_length != 1:
            raise ValueError("incremental attention input must contain one position")
        query, key, value = self.qkv_projection(values).chunk(3, dim=-1)
        query = apply_rotary_embedding(
            self._split_heads(query, batch_size, 1), position_offset=position
        )
        key = apply_rotary_embedding(
            self._split_heads(key, batch_size, 1), position_offset=position
        )
        value = self._split_heads(value, batch_size, 1)
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise ValueError("incremental attention cache has an incompatible structure")
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
            query, key, value, dropout_p=0.0, is_causal=False
        )
        return self._merge_heads(attended, d_model), (key, value)

    def _split_heads(
        self, values: Tensor, batch_size: int, sequence_length: int
    ) -> Tensor:
        return values.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _merge_heads(self, values: Tensor, d_model: int) -> Tensor:
        batch_size, _, sequence_length, _ = values.shape
        merged = (
            values.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, d_model)
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

    def prefill(self, values: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attended, cache = self.attention.prefill(self.attention_norm(values))
        values = values + attended
        return values + self.feedforward(self.feedforward_norm(values)), cache

    def decode_step(
        self,
        values: Tensor,
        cache: tuple[Tensor, Tensor],
        *,
        position: int,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attended, cache = self.attention.decode_step(
            self.attention_norm(values), cache, position=position
        )
        values = values + attended
        return values + self.feedforward(self.feedforward_norm(values)), cache


class ResidualFeedForwardBlock(nn.Module):
    """Pre-normalized GELU feed-forward layer with a residual connection."""

    def __init__(
        self,
        d_model: int,
        *,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return values + self.feedforward(self.norm(values))


@dataclass(slots=True)
class CNNAttentionCache:
    """Incremental state for Model B generation."""

    token_ids: Tensor
    pre_cnn: list[Tensor]
    attention: tuple[Tensor, Tensor]
    post_cnn: list[Tensor]


@dataclass(slots=True)
class CNNInterleavedAttentionCache:
    """Incremental state shared by interleaved-attention models."""

    token_ids: Tensor
    cnn_groups: list[list[Tensor]]
    attention: list[tuple[Tensor, Tensor]]


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

        _initialize_backbone(self, attention_config)
        self.apply(initialize_weights)
        if attention_config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        validate_input_ids(input_ids, context_length=self.config.context_length)

        values = _forward_backbone(self, input_ids)
        return self.lm_head(self.final_norm(values))

    def prefill(self, input_ids: Tensor) -> tuple[Tensor, CNNAttentionCache]:
        """Populate incremental caches from an input window."""

        input_ids = input_ids[:, -self.config.context_length :]
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids)
        pre_cache: list[Tensor] = []
        for block in self.pre_attention_blocks:
            values, history = block.prefill(values)
            pre_cache.append(history)
        values, attention_cache = self.attention_block.prefill(values)
        post_cache: list[Tensor] = []
        for block in self.post_attention_blocks:
            values, history = block.prefill(values)
            post_cache.append(history)
        logits = self.lm_head(self.final_norm(values))
        return logits, CNNAttentionCache(
            token_ids=input_ids,
            pre_cnn=pre_cache,
            attention=attention_cache,
            post_cnn=post_cache,
        )

    def decode_step(
        self, input_ids: Tensor, cache: CNNAttentionCache
    ) -> tuple[Tensor, CNNAttentionCache]:
        """Decode one new token, rebuilding at the context-window boundary."""

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step input must have shape [batch, 1]")
        if (
            cache.token_ids.ndim != 2
            or cache.token_ids.shape[1] < 1
            or cache.token_ids.shape[1] > self.config.context_length
        ):
            raise ValueError("incremental cache has an incompatible token window")
        if cache.token_ids.shape[0] != input_ids.shape[0]:
            raise ValueError("decode_step batch size differs from the cache")
        token_ids = torch.cat((cache.token_ids, input_ids), dim=1)
        if token_ids.shape[1] > self.config.context_length:
            return self.prefill(token_ids[:, -self.config.context_length :])

        values = self.token_embedding(input_ids)
        pre_cache: list[Tensor] = []
        for block, history in zip(
            self.pre_attention_blocks, cache.pre_cnn, strict=True
        ):
            values, history = block.decode_step(values, history)
            pre_cache.append(history)
        values, attention_cache = self.attention_block.decode_step(
            values, cache.attention, position=cache.token_ids.shape[1]
        )
        post_cache: list[Tensor] = []
        for block, history in zip(
            self.post_attention_blocks, cache.post_cnn, strict=True
        ):
            values, history = block.decode_step(values, history)
            post_cache.append(history)
        logits = self.lm_head(self.final_norm(values))
        return logits, CNNAttentionCache(
            token_ids=token_ids,
            pre_cnn=pre_cache,
            attention=attention_cache,
            post_cnn=post_cache,
        )


class CNNFFNAttentionLM(CausalLanguageModel):
    """Model G: Model B with a residual FFN after every gated CNN."""

    def __init__(self, config: CNNFFNAttentionConfig | None = None) -> None:
        super().__init__()
        ffn_config = config or CNNFFNAttentionConfig()
        if not isinstance(ffn_config, CNNFFNAttentionConfig):
            raise TypeError("CNNFFNAttentionLM requires a CNNFFNAttentionConfig")

        _initialize_backbone(self, ffn_config)
        self.pre_attention_ffn_blocks = nn.ModuleList(
            ResidualFeedForwardBlock(
                ffn_config.d_model,
                feedforward_dim=ffn_config.feedforward_dim,
                dropout=ffn_config.dropout,
            )
            for _ in ffn_config.pre_attention_dilations
        )
        self.post_attention_ffn_blocks = nn.ModuleList(
            ResidualFeedForwardBlock(
                ffn_config.d_model,
                feedforward_dim=ffn_config.feedforward_dim,
                dropout=ffn_config.dropout,
            )
            for _ in ffn_config.post_attention_dilations
        )
        self.apply(initialize_weights)
        if ffn_config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids)
        for cnn, feedforward in zip(
            self.pre_attention_blocks,
            self.pre_attention_ffn_blocks,
            strict=True,
        ):
            values = feedforward(cnn(values))
        values = self.attention_block(values)
        for cnn, feedforward in zip(
            self.post_attention_blocks,
            self.post_attention_ffn_blocks,
            strict=True,
        ):
            values = feedforward(cnn(values))
        return self.lm_head(self.final_norm(values))

    def prefill(self, input_ids: Tensor) -> tuple[Tensor, CNNAttentionCache]:
        """Populate Model B's incremental caches while applying every FFN."""

        input_ids = input_ids[:, -self.config.context_length :]
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids)
        pre_cache: list[Tensor] = []
        for cnn, feedforward in zip(
            self.pre_attention_blocks,
            self.pre_attention_ffn_blocks,
            strict=True,
        ):
            values, history = cnn.prefill(values)
            values = feedforward(values)
            pre_cache.append(history)
        values, attention_cache = self.attention_block.prefill(values)
        post_cache: list[Tensor] = []
        for cnn, feedforward in zip(
            self.post_attention_blocks,
            self.post_attention_ffn_blocks,
            strict=True,
        ):
            values, history = cnn.prefill(values)
            values = feedforward(values)
            post_cache.append(history)
        logits = self.lm_head(self.final_norm(values))
        return logits, CNNAttentionCache(
            token_ids=input_ids,
            pre_cnn=pre_cache,
            attention=attention_cache,
            post_cnn=post_cache,
        )

    def decode_step(
        self,
        input_ids: Tensor,
        cache: CNNAttentionCache,
    ) -> tuple[Tensor, CNNAttentionCache]:
        """Decode one token with CNN histories and the shared attention cache."""

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step input must have shape [batch, 1]")
        if not isinstance(cache, CNNAttentionCache):
            raise ValueError("incremental cache has an incompatible structure")
        if (
            cache.token_ids.ndim != 2
            or cache.token_ids.shape[1] < 1
            or cache.token_ids.shape[1] > self.config.context_length
        ):
            raise ValueError("incremental cache has an incompatible token window")
        if cache.token_ids.shape[0] != input_ids.shape[0]:
            raise ValueError("decode_step batch size differs from the cache")
        if (
            len(cache.pre_cnn) != len(self.pre_attention_blocks)
            or len(cache.post_cnn) != len(self.post_attention_blocks)
        ):
            raise ValueError("incremental cache has an incompatible structure")

        token_ids = torch.cat((cache.token_ids, input_ids), dim=1)
        if token_ids.shape[1] > self.config.context_length:
            return self.prefill(token_ids[:, -self.config.context_length :])

        values = self.token_embedding(input_ids)
        pre_cache: list[Tensor] = []
        for cnn, feedforward, history in zip(
            self.pre_attention_blocks,
            self.pre_attention_ffn_blocks,
            cache.pre_cnn,
            strict=True,
        ):
            values, history = cnn.decode_step(values, history)
            values = feedforward(values)
            pre_cache.append(history)
        values, attention_cache = self.attention_block.decode_step(
            values,
            cache.attention,
            position=cache.token_ids.shape[1],
        )
        post_cache: list[Tensor] = []
        for cnn, feedforward, history in zip(
            self.post_attention_blocks,
            self.post_attention_ffn_blocks,
            cache.post_cnn,
            strict=True,
        ):
            values, history = cnn.decode_step(values, history)
            values = feedforward(values)
            post_cache.append(history)
        logits = self.lm_head(self.final_norm(values))
        return logits, CNNAttentionCache(
            token_ids=token_ids,
            pre_cnn=pre_cache,
            attention=attention_cache,
            post_cnn=post_cache,
        )


class CNNDualAttentionLM(CausalLanguageModel):
    """Model C: Model B followed by a second full attention block."""

    def __init__(self, config: CNNDualAttentionConfig | None = None) -> None:
        super().__init__()
        dual_config = config or CNNDualAttentionConfig()
        if not isinstance(dual_config, CNNDualAttentionConfig):
            raise TypeError("CNNDualAttentionLM requires a CNNDualAttentionConfig")
        _initialize_backbone(self, dual_config)
        self.final_attention_block = TransformerAttentionBlock(
            dual_config.d_model,
            num_heads=dual_config.num_heads,
            feedforward_dim=dual_config.feedforward_dim,
            dropout=dual_config.dropout,
        )
        self.apply(initialize_weights)
        if dual_config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.final_attention_block(_forward_backbone(self, input_ids))
        return self.lm_head(self.final_norm(values))


class CNNAttentionMambaLM(CausalLanguageModel):
    """Model D: Model B followed by a portable Mamba block."""

    def __init__(self, config: CNNAttentionMambaConfig | None = None) -> None:
        super().__init__()
        mamba_config = config or CNNAttentionMambaConfig()
        if not isinstance(mamba_config, CNNAttentionMambaConfig):
            raise TypeError(
                "CNNAttentionMambaLM requires a CNNAttentionMambaConfig"
            )
        _initialize_backbone(self, mamba_config)
        self.mamba_block = MambaBlock(
            mamba_config.d_model,
            inner_dim=mamba_config.mamba_inner_dim,
            state_dim=mamba_config.mamba_state_dim,
            conv_kernel=mamba_config.mamba_conv_kernel,
            dt_rank=mamba_config.mamba_dt_rank,
            dropout=mamba_config.dropout,
        )
        self.apply(initialize_weights)
        self.mamba_block.reset_ssm_parameters()
        if mamba_config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.mamba_block(_forward_backbone(self, input_ids))
        return self.lm_head(self.final_norm(values))


class CNNInterleavedAttentionLM(CausalLanguageModel):
    """Model E: pairs of gated CNNs separated by two attention blocks."""

    def __init__(
        self,
        config: CNNInterleavedAttentionConfig | None = None,
    ) -> None:
        super().__init__()
        interleaved_config = config or CNNInterleavedAttentionConfig()
        if (
            not isinstance(
                interleaved_config,
                CNNInterleavedAttentionConfig,
            )
            or interleaved_config.architecture != "cnn_interleaved_attention"
        ):
            raise TypeError(
                "CNNInterleavedAttentionLM requires "
                "a CNNInterleavedAttentionConfig"
            )
        _initialize_interleaved_model(
            self,
            interleaved_config,
            dilation_groups=(
                interleaved_config.dilations[0:2],
                interleaved_config.dilations[2:4],
                interleaved_config.dilations[4:6],
            ),
            attention_count=2,
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids)
        for group_index, cnn_group in enumerate(self.cnn_groups):
            for block in cnn_group:
                values = block(values)
            if group_index < len(self.attention_blocks):
                values = self.attention_blocks[group_index](values)
        return self.lm_head(self.final_norm(values))

    def prefill(
        self, input_ids: Tensor
    ) -> tuple[Tensor, CNNInterleavedAttentionCache]:
        """Populate all CNN histories and both attention KV caches."""

        input_ids = input_ids[:, -self.config.context_length :]
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids)
        cnn_caches: list[list[Tensor]] = []
        attention_caches: list[tuple[Tensor, Tensor]] = []
        for group_index, cnn_group in enumerate(self.cnn_groups):
            group_cache: list[Tensor] = []
            for block in cnn_group:
                values, history = block.prefill(values)
                group_cache.append(history)
            cnn_caches.append(group_cache)
            if group_index < len(self.attention_blocks):
                values, attention_cache = self.attention_blocks[
                    group_index
                ].prefill(values)
                attention_caches.append(attention_cache)
        logits = self.lm_head(self.final_norm(values))
        return logits, CNNInterleavedAttentionCache(
            token_ids=input_ids,
            cnn_groups=cnn_caches,
            attention=attention_caches,
        )

    def decode_step(
        self,
        input_ids: Tensor,
        cache: CNNInterleavedAttentionCache,
    ) -> tuple[Tensor, CNNInterleavedAttentionCache]:
        """Decode one token with exact context-window rollover behavior."""

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step input must have shape [batch, 1]")
        if (
            cache.token_ids.ndim != 2
            or cache.token_ids.shape[1] < 1
            or cache.token_ids.shape[1] > self.config.context_length
        ):
            raise ValueError("incremental cache has an incompatible token window")
        if cache.token_ids.shape[0] != input_ids.shape[0]:
            raise ValueError("decode_step batch size differs from the cache")
        if (
            len(cache.cnn_groups) != len(self.cnn_groups)
            or len(cache.attention) != len(self.attention_blocks)
            or any(
                len(cached_group) != len(cnn_group)
                for cached_group, cnn_group in zip(
                    cache.cnn_groups,
                    self.cnn_groups,
                    strict=True,
                )
            )
        ):
            raise ValueError("incremental cache has an incompatible structure")
        token_ids = torch.cat((cache.token_ids, input_ids), dim=1)
        if token_ids.shape[1] > self.config.context_length:
            return self.prefill(token_ids[:, -self.config.context_length :])

        values = self.token_embedding(input_ids)
        cnn_caches: list[list[Tensor]] = []
        attention_caches: list[tuple[Tensor, Tensor]] = []
        for group_index, (cnn_group, cached_group) in enumerate(
            zip(self.cnn_groups, cache.cnn_groups, strict=True)
        ):
            group_cache: list[Tensor] = []
            for block, history in zip(cnn_group, cached_group, strict=True):
                values, history = block.decode_step(values, history)
                group_cache.append(history)
            cnn_caches.append(group_cache)
            if group_index < len(self.attention_blocks):
                values, attention_cache = self.attention_blocks[
                    group_index
                ].decode_step(
                    values,
                    cache.attention[group_index],
                    position=cache.token_ids.shape[1],
                )
                attention_caches.append(attention_cache)
        logits = self.lm_head(self.final_norm(values))
        return logits, CNNInterleavedAttentionCache(
            token_ids=token_ids,
            cnn_groups=cnn_caches,
            attention=attention_caches,
        )


class CNNDeepInterleavedAttentionLM(CNNInterleavedAttentionLM):
    """Model F: Model E plus local refinement and final global attention."""

    def __init__(
        self,
        config: CNNDeepInterleavedAttentionConfig | None = None,
    ) -> None:
        nn.Module.__init__(self)
        deep_config = config or CNNDeepInterleavedAttentionConfig()
        if not isinstance(deep_config, CNNDeepInterleavedAttentionConfig):
            raise TypeError(
                "CNNDeepInterleavedAttentionLM requires "
                "a CNNDeepInterleavedAttentionConfig"
            )
        _initialize_interleaved_model(
            self,
            deep_config,
            dilation_groups=(
                deep_config.dilations[0:2],
                deep_config.dilations[2:4],
                (
                    *deep_config.dilations[4:6],
                    *deep_config.refinement_dilations,
                ),
            ),
            attention_count=3,
        )


def _initialize_interleaved_model(
    model: CNNInterleavedAttentionLM,
    config: CNNInterleavedAttentionConfig,
    *,
    dilation_groups: tuple[tuple[int, ...], ...],
    attention_count: int,
) -> None:
    model.config = config
    model.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
    model.cnn_groups = nn.ModuleList(
        nn.ModuleList(
            GatedCNNBlock(
                config.d_model,
                kernel_size=config.kernel_size,
                dilation=dilation,
                dropout=config.dropout,
            )
            for dilation in dilations
        )
        for dilations in dilation_groups
    )
    model.attention_blocks = nn.ModuleList(
        TransformerAttentionBlock(
            config.d_model,
            num_heads=config.num_heads,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
        )
        for _ in range(attention_count)
    )
    model.final_norm = nn.LayerNorm(config.d_model)
    model.lm_head = nn.Linear(
        config.d_model,
        config.vocab_size,
        bias=True,
    )
    model.apply(initialize_weights)
    if config.tie_embeddings:
        model.lm_head.weight = model.token_embedding.weight


def _initialize_backbone(
    model: CausalLanguageModel,
    config: CNNAttentionConfig,
) -> None:
    model.config = config
    model.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
    model.pre_attention_blocks = nn.ModuleList(
        GatedCNNBlock(
            config.d_model,
            kernel_size=config.kernel_size,
            dilation=dilation,
            dropout=config.dropout,
        )
        for dilation in config.pre_attention_dilations
    )
    model.attention_block = TransformerAttentionBlock(
        config.d_model,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
    )
    model.post_attention_blocks = nn.ModuleList(
        GatedCNNBlock(
            config.d_model,
            kernel_size=config.kernel_size,
            dilation=dilation,
            dropout=config.dropout,
        )
        for dilation in config.post_attention_dilations
    )
    model.final_norm = nn.LayerNorm(config.d_model)
    model.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True)


def _forward_backbone(model: CausalLanguageModel, input_ids: Tensor) -> Tensor:
    values = model.token_embedding(input_ids)
    for block in model.pre_attention_blocks:
        values = block(values)
    values = model.attention_block(values)
    for block in model.post_attention_blocks:
        values = block(values)
    return values


def _build_cnn_attention(config: ModelConfig) -> CausalLanguageModel:
    return CNNAttentionLM(config)


register_model("cnn_attention", _build_cnn_attention)


def _build_cnn_attention_ffn(config: ModelConfig) -> CausalLanguageModel:
    if not isinstance(config, CNNFFNAttentionConfig):
        raise TypeError("cnn_attention_ffn requires CNNFFNAttentionConfig")
    return CNNFFNAttentionLM(config)


register_model("cnn_attention_ffn", _build_cnn_attention_ffn)


def _build_cnn_dual_attention(config: ModelConfig) -> CausalLanguageModel:
    if not isinstance(config, CNNDualAttentionConfig):
        raise TypeError("cnn_dual_attention requires CNNDualAttentionConfig")
    return CNNDualAttentionLM(config)


def _build_cnn_attention_mamba(config: ModelConfig) -> CausalLanguageModel:
    if not isinstance(config, CNNAttentionMambaConfig):
        raise TypeError("cnn_attention_mamba requires CNNAttentionMambaConfig")
    return CNNAttentionMambaLM(config)


def _build_cnn_interleaved_attention(
    config: ModelConfig,
) -> CausalLanguageModel:
    if not isinstance(config, CNNInterleavedAttentionConfig):
        raise TypeError(
            "cnn_interleaved_attention requires "
            "CNNInterleavedAttentionConfig"
        )
    return CNNInterleavedAttentionLM(config)


def _build_cnn_deep_interleaved_attention(
    config: ModelConfig,
) -> CausalLanguageModel:
    if not isinstance(config, CNNDeepInterleavedAttentionConfig):
        raise TypeError(
            "cnn_deep_interleaved_attention requires "
            "CNNDeepInterleavedAttentionConfig"
        )
    return CNNDeepInterleavedAttentionLM(config)


register_model("cnn_dual_attention", _build_cnn_dual_attention)
register_model("cnn_attention_mamba", _build_cnn_attention_mamba)
register_model(
    "cnn_interleaved_attention",
    _build_cnn_interleaved_attention,
)
register_model(
    "cnn_deep_interleaved_attention",
    _build_cnn_deep_interleaved_attention,
)
