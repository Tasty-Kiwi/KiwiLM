"""Model X alternating gated-convolution and attention language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kiwilm.config import ModelConfig, ModelXConfig
from kiwilm.models.attention import CausalSelfAttention
from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.components import CausalConv1d, initialize_weights, validate_input_ids
from kiwilm.models.registry import register_model


class RMSGatedCNNBlock(nn.Module):
    """Pre-RMSNorm residual dense causal convolution with a GLU gate."""

    def __init__(
        self,
        d_model: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=rms_norm_eps)
        self.conv = CausalConv1d(
            d_model,
            2 * d_model,
            kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.mix(values)

    def mix(self, values: Tensor) -> Tensor:
        """Return the normalized gated-convolution update without a residual."""

        normalized = self.norm(values)
        convolved = self.conv(normalized.transpose(1, 2))
        gated = F.glu(convolved, dim=1).transpose(1, 2)
        return self.dropout(gated)

    def prefill(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Run a sequence and retain normalized convolution history."""

        update, history = self.prefill_update(values)
        return values + update, history

    def prefill_update(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Return a branch-only update and its normalized convolution history."""

        normalized = self.norm(values)
        convolved = self.conv(normalized.transpose(1, 2))
        gated = F.glu(convolved, dim=1).transpose(1, 2)
        update = self.dropout(gated)
        history_length = self.conv.left_padding
        history = normalized[:, -history_length:, :] if history_length else normalized[:, :0, :]
        if history.shape[1] < history_length:
            history = F.pad(
                history.transpose(1, 2),
                (history_length - history.shape[1], 0),
            ).transpose(1, 2)
        return update, history

    def decode_step(self, values: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        """Process one position from a cache produced by :meth:`prefill`."""

        update, next_history = self.decode_step_update(values, history)
        return values + update, next_history

    def decode_step_update(
        self,
        values: Tensor,
        history: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return one branch-only update and advance convolution history."""

        if values.ndim != 3 or values.shape[1] != 1:
            raise ValueError("incremental CNN input must have shape [batch, 1, width]")
        normalized = self.norm(values)
        required = self.conv.left_padding
        if history.shape != (values.shape[0], required, values.shape[2]):
            raise ValueError("incremental CNN history has an incompatible shape")
        window = torch.cat((history, normalized), dim=1)
        convolved = self.conv.conv(window.transpose(1, 2))
        gated = F.glu(convolved, dim=1).transpose(1, 2)
        update = self.dropout(gated)
        next_history = window[:, -required:, :] if required else window[:, :0, :]
        return update, next_history


class ResidualSwiGLUBlock(nn.Module):
    """Pre-RMSNorm residual SwiGLU channel mixer."""

    def __init__(
        self,
        d_model: int,
        *,
        hidden_dim: int,
        dropout: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=rms_norm_eps)
        self.gate_projection = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_projection = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_projection = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.norm(values)
        gated = F.silu(self.gate_projection(normalized))
        gated = gated * self.up_projection(normalized)
        return values + self.dropout(self.down_projection(gated))


class RMSAttentionBlock(nn.Module):
    """Pre-RMSNorm residual RoPE causal self-attention mixer."""

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        dropout: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=rms_norm_eps)
        self.attention = CausalSelfAttention(
            d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, values: Tensor) -> Tensor:
        return values + self.mix(values)

    def mix(self, values: Tensor) -> Tensor:
        """Return the normalized attention update without a residual."""

        return self.attention(self.norm(values))

    def prefill(self, values: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Attend over a prompt and return the rotated attention cache."""

        update, cache = self.prefill_update(values)
        return values + update, cache

    def prefill_update(
        self,
        values: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Return a branch-only attention update and its KV cache."""

        return self.attention.prefill(self.norm(values))

    def decode_step(
        self,
        values: Tensor,
        cache: tuple[Tensor, Tensor],
        *,
        position: int,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Attend one position over cached keys and values."""

        update, cache = self.decode_step_update(
            values,
            cache,
            position=position,
        )
        return values + update, cache

    def decode_step_update(
        self,
        values: Tensor,
        cache: tuple[Tensor, Tensor],
        *,
        position: int,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Return one branch-only attention update and advance its KV cache."""

        return self.attention.decode_step(
            self.norm(values),
            cache,
            position=position,
        )


@dataclass(slots=True)
class ModelXCache:
    """Incremental state for Model X's two CNN and two attention mixers."""

    token_ids: Tensor
    cnn: list[Tensor]
    attention: list[tuple[Tensor, Tensor]]


class ModelXLM(CausalLanguageModel):
    """Alternating local/global mixers with a SwiGLU after every mixer."""

    def __init__(self, config: ModelXConfig | None = None) -> None:
        super().__init__()
        model_config = config or ModelXConfig()
        if not isinstance(model_config, ModelXConfig):
            raise TypeError("ModelXLM requires a ModelXConfig")

        self.config = model_config
        self.token_embedding = nn.Embedding(
            model_config.vocab_size,
            model_config.d_model,
        )
        self.cnn_blocks = nn.ModuleList(
            RMSGatedCNNBlock(
                model_config.d_model,
                kernel_size=model_config.kernel_size,
                dilation=dilation,
                dropout=model_config.dropout,
                rms_norm_eps=model_config.rms_norm_eps,
            )
            for dilation in model_config.cnn_dilations
        )
        self.attention_blocks = nn.ModuleList(
            RMSAttentionBlock(
                model_config.d_model,
                num_heads=model_config.num_heads,
                dropout=model_config.dropout,
                rms_norm_eps=model_config.rms_norm_eps,
            )
            for _ in range(2)
        )
        self.feedforward_blocks = nn.ModuleList(
            ResidualSwiGLUBlock(
                model_config.d_model,
                hidden_dim=model_config.swiglu_dim,
                dropout=model_config.dropout,
                rms_norm_eps=model_config.rms_norm_eps,
            )
            for _ in range(4)
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
        values = self.feedforward_blocks[0](self.cnn_blocks[0](values))
        values = self.feedforward_blocks[1](self.attention_blocks[0](values))
        values = self.feedforward_blocks[2](self.cnn_blocks[1](values))
        values = self.feedforward_blocks[3](self.attention_blocks[1](values))
        return self.lm_head(self.final_norm(values))

    def prefill(self, input_ids: Tensor) -> tuple[Tensor, ModelXCache]:
        """Populate every convolution history and attention KV cache."""

        input_ids = input_ids[:, -self.config.context_length :]
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids)
        cnn_caches: list[Tensor] = []
        attention_caches: list[tuple[Tensor, Tensor]] = []

        values, cnn_cache = self.cnn_blocks[0].prefill(values)
        cnn_caches.append(cnn_cache)
        values = self.feedforward_blocks[0](values)
        values, attention_cache = self.attention_blocks[0].prefill(values)
        attention_caches.append(attention_cache)
        values = self.feedforward_blocks[1](values)
        values, cnn_cache = self.cnn_blocks[1].prefill(values)
        cnn_caches.append(cnn_cache)
        values = self.feedforward_blocks[2](values)
        values, attention_cache = self.attention_blocks[1].prefill(values)
        attention_caches.append(attention_cache)
        values = self.feedforward_blocks[3](values)

        return self.lm_head(self.final_norm(values)), ModelXCache(
            token_ids=input_ids,
            cnn=cnn_caches,
            attention=attention_caches,
        )

    def decode_step(
        self,
        input_ids: Tensor,
        cache: ModelXCache,
    ) -> tuple[Tensor, ModelXCache]:
        """Decode one token, rebuilding all caches at context rollover."""

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step input must have shape [batch, 1]")
        if not isinstance(cache, ModelXCache):
            raise ValueError("incremental cache has an incompatible structure")
        if (
            cache.token_ids.ndim != 2
            or cache.token_ids.shape[1] < 1
            or cache.token_ids.shape[1] > self.config.context_length
        ):
            raise ValueError("incremental cache has an incompatible token window")
        if cache.token_ids.shape[0] != input_ids.shape[0]:
            raise ValueError("decode_step batch size differs from the cache")
        if len(cache.cnn) != 2 or len(cache.attention) != 2:
            raise ValueError("incremental cache has an incompatible structure")

        token_ids = torch.cat((cache.token_ids, input_ids), dim=1)
        if token_ids.shape[1] > self.config.context_length:
            return self.prefill(token_ids[:, -self.config.context_length :])

        position = cache.token_ids.shape[1]
        values = self.token_embedding(input_ids)
        cnn_caches: list[Tensor] = []
        attention_caches: list[tuple[Tensor, Tensor]] = []

        values, cnn_cache = self.cnn_blocks[0].decode_step(values, cache.cnn[0])
        cnn_caches.append(cnn_cache)
        values = self.feedforward_blocks[0](values)
        values, attention_cache = self.attention_blocks[0].decode_step(
            values,
            cache.attention[0],
            position=position,
        )
        attention_caches.append(attention_cache)
        values = self.feedforward_blocks[1](values)
        values, cnn_cache = self.cnn_blocks[1].decode_step(values, cache.cnn[1])
        cnn_caches.append(cnn_cache)
        values = self.feedforward_blocks[2](values)
        values, attention_cache = self.attention_blocks[1].decode_step(
            values,
            cache.attention[1],
            position=position,
        )
        attention_caches.append(attention_cache)
        values = self.feedforward_blocks[3](values)

        return self.lm_head(self.final_norm(values)), ModelXCache(
            token_ids=token_ids,
            cnn=cnn_caches,
            attention=attention_caches,
        )


def _build_model_x(config: ModelConfig) -> CausalLanguageModel:
    if not isinstance(config, ModelXConfig):
        raise TypeError("model_x requires ModelXConfig")
    return ModelXLM(config)


register_model("model_x", _build_model_x)


__all__ = [
    "ModelXCache",
    "ModelXLM",
    "RMSAttentionBlock",
    "RMSGatedCNNBlock",
    "ResidualSwiGLUBlock",
]
