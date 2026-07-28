"""Controlled GPT-style decoder-only Transformer language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kiwilm.config import ModelConfig, TransformerConfig
from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.cnn_attention import TransformerAttentionBlock
from kiwilm.models.components import initialize_weights, validate_input_ids
from kiwilm.models.registry import register_model


@dataclass(slots=True)
class TransformerCache:
    """Incremental state for the decoder-only Transformer."""

    token_ids: Tensor
    attention: list[tuple[Tensor, Tensor]]


class TransformerLM(CausalLanguageModel):
    """Embedding, four Transformer blocks, final norm, and tied LM head."""

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        transformer_config = config or TransformerConfig()
        if not isinstance(transformer_config, TransformerConfig):
            raise TypeError("TransformerLM requires a TransformerConfig")

        self.config = transformer_config
        self.token_embedding = nn.Embedding(
            transformer_config.vocab_size,
            transformer_config.d_model,
        )
        self.blocks = nn.ModuleList(
            TransformerAttentionBlock(
                transformer_config.d_model,
                num_heads=transformer_config.num_heads,
                feedforward_dim=transformer_config.feedforward_dim,
                dropout=transformer_config.dropout,
            )
            for _ in range(transformer_config.num_layers)
        )
        self.final_norm = nn.LayerNorm(transformer_config.d_model)
        self.lm_head = nn.Linear(
            transformer_config.d_model,
            transformer_config.vocab_size,
            bias=True,
        )
        self.apply(initialize_weights)
        if transformer_config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids)
        for block in self.blocks:
            values = block(values)
        return self.lm_head(self.final_norm(values))

    def prefill(self, input_ids: Tensor) -> tuple[Tensor, TransformerCache]:
        """Populate every attention KV cache from the current token window."""

        input_ids = input_ids[:, -self.config.context_length :]
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids)
        attention_caches: list[tuple[Tensor, Tensor]] = []
        for block in self.blocks:
            values, attention_cache = block.prefill(values)
            attention_caches.append(attention_cache)
        logits = self.lm_head(self.final_norm(values))
        return logits, TransformerCache(
            token_ids=input_ids,
            attention=attention_caches,
        )

    def decode_step(
        self,
        input_ids: Tensor,
        cache: TransformerCache,
    ) -> tuple[Tensor, TransformerCache]:
        """Decode one token and rebuild caches when the context window rolls."""

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step input must have shape [batch, 1]")
        if not isinstance(cache, TransformerCache):
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
        attention_caches: list[tuple[Tensor, Tensor]] = []
        position = cache.token_ids.shape[1]
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
        logits = self.lm_head(self.final_norm(values))
        return logits, TransformerCache(
            token_ids=token_ids,
            attention=attention_caches,
        )


def _build_transformer(config: ModelConfig) -> CausalLanguageModel:
    if not isinstance(config, TransformerConfig):
        raise TypeError("transformer requires TransformerConfig")
    return TransformerLM(config)


register_model("transformer", _build_transformer)


__all__ = ["TransformerCache", "TransformerLM"]
