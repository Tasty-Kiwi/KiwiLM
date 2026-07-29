"""Model Z-P: fixed parallel gated-convolution and attention fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kiwilm.config import ModelConfig, ModelZParallelConfig
from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.components import initialize_weights, validate_input_ids
from kiwilm.models.model_x import (
    ResidualSwiGLUBlock,
    RMSAttentionBlock,
    RMSGatedCNNBlock,
)
from kiwilm.models.registry import register_model

PARALLEL_BRANCH_SCALE = 1.0 / math.sqrt(2.0)


class ModelZParallelBlock(nn.Module):
    """Fuse local and global updates computed from the same residual input."""

    def __init__(
        self,
        d_model: int,
        *,
        kernel_size: int,
        dilation: int,
        num_heads: int,
        swiglu_dim: int,
        dropout: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.cnn = RMSGatedCNNBlock(
            d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout,
            rms_norm_eps=rms_norm_eps,
        )
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

    def branch_updates(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Return branch-only CNN and attention updates for one block."""

        return self.cnn.mix(values), self.attention.mix(values)

    def forward(self, values: Tensor) -> Tensor:
        cnn_update, attention_update = self.branch_updates(values)
        values = values + PARALLEL_BRANCH_SCALE * (cnn_update + attention_update)
        return self.feedforward(values)

    def prefill(
        self,
        values: Tensor,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Run both branches and populate their independent incremental caches."""

        cnn_update, cnn_cache = self.cnn.prefill_update(values)
        attention_update, attention_cache = self.attention.prefill_update(values)
        values = values + PARALLEL_BRANCH_SCALE * (cnn_update + attention_update)
        return self.feedforward(values), cnn_cache, attention_cache

    def decode_step(
        self,
        values: Tensor,
        cnn_cache: Tensor,
        attention_cache: tuple[Tensor, Tensor],
        *,
        position: int,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Decode one position through both branches from the same block input."""

        cnn_update, cnn_cache = self.cnn.decode_step_update(values, cnn_cache)
        attention_update, attention_cache = self.attention.decode_step_update(
            values,
            attention_cache,
            position=position,
        )
        values = values + PARALLEL_BRANCH_SCALE * (cnn_update + attention_update)
        return self.feedforward(values), cnn_cache, attention_cache


@dataclass(slots=True)
class ModelZParallelCache:
    """Raw token window plus two CNN histories and two attention KV caches."""

    token_ids: Tensor
    cnn: list[Tensor]
    attention: list[tuple[Tensor, Tensor]]


class ModelZParallelLM(CausalLanguageModel):
    """Two fixed parallel local/global blocks with wide SwiGLU refinements."""

    def __init__(self, config: ModelZParallelConfig | None = None) -> None:
        super().__init__()
        model_config = config or ModelZParallelConfig()
        if not isinstance(model_config, ModelZParallelConfig):
            raise TypeError("ModelZParallelLM requires a ModelZParallelConfig")

        self.config = model_config
        self.token_embedding = nn.Embedding(
            model_config.vocab_size,
            model_config.d_model,
        )
        self.blocks = nn.ModuleList(
            ModelZParallelBlock(
                model_config.d_model,
                kernel_size=model_config.kernel_size,
                dilation=dilation,
                num_heads=model_config.num_heads,
                swiglu_dim=model_config.swiglu_dim,
                dropout=model_config.dropout,
                rms_norm_eps=model_config.rms_norm_eps,
            )
            for dilation in model_config.cnn_dilations
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
    ) -> tuple[Tensor, ModelZParallelCache]:
        input_ids = input_ids[:, -self.config.context_length :]
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids)
        cnn_caches: list[Tensor] = []
        attention_caches: list[tuple[Tensor, Tensor]] = []
        for block in self.blocks:
            values, cnn_cache, attention_cache = block.prefill(values)
            cnn_caches.append(cnn_cache)
            attention_caches.append(attention_cache)
        return self.lm_head(self.final_norm(values)), ModelZParallelCache(
            token_ids=input_ids,
            cnn=cnn_caches,
            attention=attention_caches,
        )

    def decode_step(
        self,
        input_ids: Tensor,
        cache: ModelZParallelCache,
    ) -> tuple[Tensor, ModelZParallelCache]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step input must have shape [batch, 1]")
        if not isinstance(cache, ModelZParallelCache):
            raise ValueError("incremental cache has an incompatible structure")
        if (
            cache.token_ids.ndim != 2
            or cache.token_ids.shape[1] < 1
            or cache.token_ids.shape[1] > self.config.context_length
        ):
            raise ValueError("incremental cache has an incompatible token window")
        if cache.token_ids.shape[0] != input_ids.shape[0]:
            raise ValueError("decode_step batch size differs from the cache")
        if len(cache.cnn) != len(self.blocks) or len(cache.attention) != len(self.blocks):
            raise ValueError("incremental cache has an incompatible structure")

        token_ids = torch.cat((cache.token_ids, input_ids), dim=1)
        if token_ids.shape[1] > self.config.context_length:
            return self.prefill(token_ids[:, -self.config.context_length :])

        position = cache.token_ids.shape[1]
        values = self.token_embedding(input_ids)
        cnn_caches: list[Tensor] = []
        attention_caches: list[tuple[Tensor, Tensor]] = []
        for block, cnn_cache, attention_cache in zip(
            self.blocks,
            cache.cnn,
            cache.attention,
            strict=True,
        ):
            values, cnn_cache, attention_cache = block.decode_step(
                values,
                cnn_cache,
                attention_cache,
                position=position,
            )
            cnn_caches.append(cnn_cache)
            attention_caches.append(attention_cache)
        return self.lm_head(self.final_norm(values)), ModelZParallelCache(
            token_ids=token_ids,
            cnn=cnn_caches,
            attention=attention_caches,
        )

    @torch.no_grad()
    def branch_diagnostics(self, input_ids: Tensor) -> list[dict[str, float]]:
        """Measure complementarity and scale of each fixed parallel branch."""

        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self.token_embedding(input_ids)
        diagnostics: list[dict[str, float]] = []
        for index, block in enumerate(self.blocks):
            cnn_update, attention_update = block.branch_updates(values)
            cnn_rms = _rms(cnn_update)
            attention_rms = _rms(attention_update)
            residual_rms = _rms(values)
            merged_update = PARALLEL_BRANCH_SCALE * (cnn_update + attention_update)
            diagnostics.append(
                {
                    "block": float(index + 1),
                    "dilation": float(block.cnn.conv.conv.dilation[0]),
                    "cnn_rms": cnn_rms,
                    "attention_rms": attention_rms,
                    "cnn_to_attention_rms": cnn_rms / max(attention_rms, 1e-12),
                    "branch_cosine_similarity": float(
                        F.cosine_similarity(
                            cnn_update.float(),
                            attention_update.float(),
                            dim=-1,
                            eps=1e-12,
                        )
                        .mean()
                        .item()
                    ),
                    "merged_update_to_residual_rms": (
                        _rms(merged_update) / max(residual_rms, 1e-12)
                    ),
                }
            )
            values = block.feedforward(values + merged_update)
        return diagnostics


def _rms(values: Tensor) -> float:
    return float(values.float().square().mean().sqrt().item())


def _build_model_z_parallel(config: ModelConfig) -> CausalLanguageModel:
    if not isinstance(config, ModelZParallelConfig):
        raise TypeError("model_z_parallel requires ModelZParallelConfig")
    return ModelZParallelLM(config)


register_model("model_z_parallel", _build_model_z_parallel)


__all__ = [
    "PARALLEL_BRANCH_SCALE",
    "ModelZParallelBlock",
    "ModelZParallelCache",
    "ModelZParallelLM",
]
