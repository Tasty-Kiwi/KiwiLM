"""Model Y: a modern RMSNorm/SwiGLU decoder-only Transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kiwilm.config import ModelConfig, ModelYConfig
from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.components import initialize_weights, validate_input_ids
from kiwilm.models.model_x import ResidualSwiGLUBlock, RMSAttentionBlock
from kiwilm.models.registry import register_model


class ModelYBlock(nn.Module):
    """Pre-RMSNorm RoPE attention followed by a pre-RMSNorm SwiGLU."""

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        swiglu_dim: int,
        dropout: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention = RMSAttentionBlock(
            d_model,
            num_heads=num_heads,
            dropout=dropout,
            rms_norm_eps=rms_norm_eps,
        )
        self.feedforward = ResidualSwiGLUBlock(
            d_model,
            hidden_dim=swiglu_dim,
            dropout=dropout,
            rms_norm_eps=rms_norm_eps,
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.feedforward(self.attention(values))

    def prefill(self, values: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        values, cache = self.attention.prefill(values)
        return self.feedforward(values), cache

    def decode_step(
        self,
        values: Tensor,
        cache: tuple[Tensor, Tensor],
        *,
        position: int,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        values, cache = self.attention.decode_step(
            values,
            cache,
            position=position,
        )
        return self.feedforward(values), cache


@dataclass(slots=True)
class ModelYCache:
    """Raw token window plus one rotated attention KV cache per block."""

    token_ids: Tensor
    attention: list[tuple[Tensor, Tensor]]


class ModelYLM(CausalLanguageModel):
    """Four modern decoder blocks, final RMSNorm, and a tied LM head."""

    def __init__(self, config: ModelYConfig | None = None) -> None:
        super().__init__()
        model_config = config or ModelYConfig()
        if not isinstance(model_config, ModelYConfig):
            raise TypeError("ModelYLM requires a ModelYConfig")

        self.config = model_config
        self.token_embedding = nn.Embedding(
            model_config.vocab_size,
            model_config.d_model,
        )
        self.blocks = nn.ModuleList(
            ModelYBlock(
                model_config.d_model,
                num_heads=model_config.num_heads,
                swiglu_dim=model_config.swiglu_dim,
                dropout=model_config.dropout,
                rms_norm_eps=model_config.rms_norm_eps,
            )
            for _ in range(model_config.num_layers)
        )
        self.final_norm = nn.RMSNorm(
            model_config.d_model,
            eps=model_config.rms_norm_eps,
        )
        self.lm_head = nn.Linear(
            model_config.d_model,
            model_config.vocab_size,
            bias=True,
        )
        self.apply(initialize_weights)
        if model_config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids)
        for block in self.blocks:
            values = block(values)
        return self.lm_head(self.final_norm(values))

    def prefill(
        self,
        input_ids: Tensor,
    ) -> tuple[Tensor, ModelYCache]:
        input_ids = input_ids[:, -self.config.context_length :]
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids)
        attention_caches: list[tuple[Tensor, Tensor]] = []
        for block in self.blocks:
            values, attention_cache = block.prefill(values)
            attention_caches.append(attention_cache)
        return self.lm_head(self.final_norm(values)), ModelYCache(
            token_ids=input_ids,
            attention=attention_caches,
        )

    def decode_step(
        self,
        input_ids: Tensor,
        cache: ModelYCache,
    ) -> tuple[Tensor, ModelYCache]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step input must have shape [batch, 1]")
        if not isinstance(cache, ModelYCache):
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

        values = self.token_embedding(input_ids)
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
        return self.lm_head(self.final_norm(values)), ModelYCache(
            token_ids=token_ids,
            attention=attention_caches,
        )


def _build_model_y(config: ModelConfig) -> CausalLanguageModel:
    if not isinstance(config, ModelYConfig):
        raise TypeError("model_y requires ModelYConfig")
    return ModelYLM(config)


register_model("model_y", _build_model_y)


__all__ = [
    "ModelYBlock",
    "ModelYCache",
    "ModelYLM",
]
