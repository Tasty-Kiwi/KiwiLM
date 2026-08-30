"""KiwiLM 2 fixed hybrid backbone and its structured Slim control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kiwilm.config import (
    KiwiLM2Config,
    KiwiLM2SlimConfig,
    KiwiLM2SlimV3Config,
    ModelConfig,
)
from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.components import initialize_weights, validate_input_ids
from kiwilm.models.registry import register_model


class RMSNorm(nn.Module):
    """RMS normalization computed in float32 with a learned gain."""

    def __init__(self, width: int, *, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, values: Tensor) -> Tensor:
        inverse_rms = torch.rsqrt(values.float().square().mean(dim=-1, keepdim=True) + self.eps).to(
            values.dtype
        )
        return values * inverse_rms * self.weight.to(values.dtype)


class CachedRotaryEmbedding(nn.Module):
    """Precomputed RoPE cosines and sines for the complete context window."""

    def __init__(self, head_dim: int, max_positions: int, *, base: float) -> None:
        super().__init__()
        positions = torch.arange(max_positions, dtype=torch.float32)
        frequencies = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inverse_frequencies = 1.0 / (base ** (frequencies / head_dim))
        angles = torch.outer(positions, inverse_frequencies)
        self.register_buffer("cosines", angles.cos(), persistent=False)
        self.register_buffer("sines", angles.sin(), persistent=False)

    def forward(self, values: Tensor, *, offset: int = 0) -> Tensor:
        length = values.shape[-2]
        end = offset + length
        if end > self.cosines.shape[0]:
            raise ValueError("RoPE positions exceed the cached context window")
        cosines = self.cosines[offset:end].to(values.dtype)[None, None]
        sines = self.sines[offset:end].to(values.dtype)[None, None]
        even, odd = values[..., 0::2], values[..., 1::2]
        return torch.stack(
            (even * cosines - odd * sines, odd * cosines + even * sines),
            dim=-1,
        ).flatten(start_dim=-2)


class KiwiLM2GQA(nn.Module):
    """Bias-free grouped-query attention with cached RoPE and compact KV state."""

    def __init__(self, config: KiwiLM2Config) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.num_query_heads = config.num_query_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_query_heads
        self.dropout = config.dropout
        kv_width = self.num_kv_heads * self.head_dim
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, kv_width, bias=False)
        self.value = nn.Linear(config.d_model, kv_width, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = CachedRotaryEmbedding(
            self.head_dim, config.context_length, base=config.rope_base
        )

    def _split(self, values: Tensor, heads: int) -> Tensor:
        batch, length, _ = values.shape
        return values.view(batch, length, heads, self.head_dim).transpose(1, 2)

    def _project(self, values: Tensor, offset: int) -> tuple[Tensor, Tensor, Tensor]:
        query = self.rope(self._split(self.query(values), self.num_query_heads), offset=offset)
        key = self.rope(self._split(self.key(values), self.num_kv_heads), offset=offset)
        value = self._split(self.value(values), self.num_kv_heads)
        return query, key, value

    def _attend(self, query: Tensor, key: Tensor, value: Tensor, *, causal: bool) -> Tensor:
        dropout = self.dropout if self.training else 0.0
        if self.num_query_heads != self.num_kv_heads:
            if query.device.type == "cuda":
                return F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=dropout,
                    is_causal=causal,
                    enable_gqa=True,
                )
            repeats = self.num_query_heads // self.num_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout, is_causal=causal
        )

    def _merge(self, values: Tensor) -> Tensor:
        batch, _, length, _ = values.shape
        return values.transpose(1, 2).contiguous().view(batch, length, self.d_model)

    def forward(self, values: Tensor) -> Tensor:
        query, key, value = self._project(values, 0)
        return self.output(self._merge(self._attend(query, key, value, causal=True)))

    def prefill(self, values: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        query, key, value = self._project(values, 0)
        update = self.output(self._merge(self._attend(query, key, value, causal=True)))
        return update, (key, value)

    def decode_step(
        self, values: Tensor, cache: tuple[Tensor, Tensor], *, position: int
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        cached_key, cached_value = cache
        expected = (values.shape[0], self.num_kv_heads, position, self.head_dim)
        if cached_key.shape != expected or cached_value.shape != expected:
            raise ValueError("GQA cache has an incompatible shape")
        query, key, value = self._project(values, position)
        key = torch.cat((cached_key, key), dim=2)
        value = torch.cat((cached_value, value), dim=2)
        update = self.output(self._merge(self._attend(query, key, value, causal=False)))
        return update, (key, value)


class XXLCausalGatedConv(nn.Module):
    """Gated pointwise projections around a large causal depthwise convolution."""

    def __init__(self, d_model: int, kernel_size: int, *, dropout: float) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.input = nn.Linear(d_model, 2 * d_model, bias=False)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=True)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _finish(self, convolved: Tensor, gate: Tensor) -> Tensor:
        return self.dropout(self.output(convolved * F.silu(gate)))

    def forward(self, values: Tensor) -> Tensor:
        projected, gate = self.input(values).chunk(2, dim=-1)
        padded = F.pad(projected.transpose(1, 2), (self.kernel_size - 1, 0))
        convolved = self.depthwise(padded).transpose(1, 2)
        return self._finish(convolved, gate)

    def prefill(self, values: Tensor) -> tuple[Tensor, Tensor]:
        projected, gate = self.input(values).chunk(2, dim=-1)
        padded = F.pad(projected.transpose(1, 2), (self.kernel_size - 1, 0))
        convolved = self.depthwise(padded).transpose(1, 2)
        history_length = self.kernel_size - 1
        padded_history = padded.transpose(1, 2)
        history = (
            padded_history[:, -history_length:]
            if history_length
            else padded_history[:, :0]
        )
        return self._finish(convolved, gate), history

    def decode_step(self, values: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        projected, gate = self.input(values).chunk(2, dim=-1)
        expected = (values.shape[0], self.kernel_size - 1, values.shape[2])
        if history.shape != expected:
            raise ValueError("convolution cache has an incompatible shape")
        window = torch.cat((history, projected), dim=1)
        convolved = self.depthwise(window.transpose(1, 2)).transpose(1, 2)
        return self._finish(convolved, gate), window[:, 1:]


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        *,
        dropout: float,
        residual_gate_init: float | None = None,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        if residual_gate_init is None:
            self.register_parameter("residual_logit", None)
        else:
            logit = math.log(residual_gate_init / (1.0 - residual_gate_init))
            self.residual_logit = nn.Parameter(torch.tensor(logit))

    @property
    def effective_residual_scale(self) -> Tensor | None:
        """Return the bounded scalar applied to this branch, when enabled."""

        if self.residual_logit is None:
            return None
        return torch.sigmoid(self.residual_logit)

    def forward(self, values: Tensor) -> Tensor:
        output = self.dropout(self.down(F.silu(self.gate(values)) * self.up(values)))
        scale = self.effective_residual_scale
        return output if scale is None else output * scale


def fast_walsh_hadamard(values: Tensor) -> Tensor:
    """Apply an orthonormal Walsh-Hadamard transform along the last axis."""

    width = values.shape[-1]
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    output = values
    stride = 1
    while stride < width:
        pairs = output.reshape(*output.shape[:-1], -1, 2, stride)
        left, right = pairs.unbind(dim=-2)
        output = torch.cat((left + right, left - right), dim=-1).reshape_as(output)
        stride *= 2
    return output / math.sqrt(width)


class HadamardMLP(nn.Module):
    """Original minimal learned diagonal/FWHT mixer used by Slim v1.

    This is intentionally not parameter-matched to SwiGLU. Two learned diagonal
    affine stages surround orthonormal transforms; SiLU supplies non-linearity.
    """

    def __init__(self, width: int, *, dropout: float) -> None:
        super().__init__()
        self.input_scale = nn.Parameter(torch.ones(width))
        self.input_bias = nn.Parameter(torch.zeros(width))
        self.output_scale = nn.Parameter(torch.ones(width))
        self.output_bias = nn.Parameter(torch.zeros(width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        mixed = fast_walsh_hadamard(values * self.input_scale + self.input_bias)
        mixed = F.silu(mixed)
        mixed = fast_walsh_hadamard(mixed * self.output_scale + self.output_bias)
        return self.dropout(mixed)


class GatedHadamardMLP(nn.Module):
    """Gated three-transform Hadamard mixer used by KiwiLM 2 Slim v2."""

    def __init__(self, width: int, *, dropout: float, residual_scale: float) -> None:
        super().__init__()
        self.gate_scale = nn.Parameter(torch.empty(width))
        self.gate_bias = nn.Parameter(torch.zeros(width))
        self.value_scale = nn.Parameter(torch.empty(width))
        self.value_bias = nn.Parameter(torch.zeros(width))
        self.output_scale = nn.Parameter(torch.empty(width))
        self.output_bias = nn.Parameter(torch.zeros(width))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            for scale in (self.gate_scale, self.value_scale, self.output_scale):
                scale.bernoulli_(0.5).mul_(2).sub_(1)

    def forward(self, values: Tensor) -> Tensor:
        gate = fast_walsh_hadamard(values * self.gate_scale + self.gate_bias)
        value = fast_walsh_hadamard(values * self.value_scale + self.value_bias)
        mixed = F.silu(gate) * value
        mixed = fast_walsh_hadamard(mixed * self.output_scale + self.output_bias)
        return self.dropout(mixed) * self.residual_scale


class NGramEmbedding(nn.Module):
    """Hashed bigram and trigram lookup tables derived only from visible IDs."""

    def __init__(self, config: KiwiLM2Config) -> None:
        super().__init__()
        self.bigram = nn.Embedding(config.bigram_buckets, config.d_model)
        self.trigram = nn.Embedding(config.trigram_buckets, config.d_model)
        self.bigram_buckets = config.bigram_buckets
        self.trigram_buckets = config.trigram_buckets

    @staticmethod
    def _previous(input_ids: Tensor, distance: int) -> Tensor:
        return F.pad(input_ids, (distance, 0), value=0)[:, : input_ids.shape[1]]

    def indices(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        previous = self._previous(input_ids, 1)
        previous2 = self._previous(input_ids, 2)
        bigram = (previous * 1_000_003 + input_ids * 9_176 + 17) % self.bigram_buckets
        trigram = (
            previous2 * 1_000_003 + previous * 9_176 + input_ids * 131 + 29
        ) % self.trigram_buckets
        return bigram, trigram

    def forward(self, input_ids: Tensor) -> Tensor:
        bigram, trigram = self.indices(input_ids)
        return self.bigram(bigram) + self.trigram(trigram)


class KiwiLM2Block(nn.Module):
    """Shared pre-RMSNorm mixer/MLP residual block."""

    def __init__(
        self,
        config: KiwiLM2Config,
        mixer: KiwiLM2GQA | XXLCausalGatedConv,
        *,
        mlp_kind: str,
    ) -> None:
        super().__init__()
        self.mixer_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mixer = mixer
        self.mlp_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        if mlp_kind == "hadamard":
            if not isinstance(config, (KiwiLM2SlimConfig, KiwiLM2SlimV3Config)):
                raise TypeError("Hadamard blocks require a Slim configuration")
            self.mlp: nn.Module = (
                GatedHadamardMLP(
                    config.d_model,
                    dropout=config.dropout,
                    residual_scale=1.0 / math.sqrt(2 * len(config.mixer_schedule)),
                )
                if config.hadamard_variant == "gated_v2"
                else HadamardMLP(config.d_model, dropout=config.dropout)
            )
        elif mlp_kind == "swiglu":
            residual_gate_init = (
                config.swiglu_residual_gate_init
                if isinstance(config, KiwiLM2SlimV3Config)
                else None
            )
            self.mlp = SwiGLU(
                config.d_model,
                config.swiglu_dim,
                dropout=config.dropout,
                residual_gate_init=residual_gate_init,
            )
        else:
            raise ValueError("mlp_kind must be 'hadamard' or 'swiglu'")

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.mixer(self.mixer_norm(values))
        return values + self.mlp(self.mlp_norm(values))

    def prefill(self, values: Tensor) -> tuple[Tensor, Any]:
        update, cache = self.mixer.prefill(self.mixer_norm(values))
        values = values + update
        return values + self.mlp(self.mlp_norm(values)), cache

    def decode_step(self, values: Tensor, cache: Any, *, position: int) -> tuple[Tensor, Any]:
        if isinstance(self.mixer, KiwiLM2GQA):
            update, cache = self.mixer.decode_step(
                self.mixer_norm(values), cache, position=position
            )
        else:
            update, cache = self.mixer.decode_step(self.mixer_norm(values), cache)
        values = values + update
        return values + self.mlp(self.mlp_norm(values)), cache


@dataclass(slots=True)
class KiwiLM2Cache:
    token_ids: Tensor
    mixers: list[Any]


class KiwiLM2LM(CausalLanguageModel):
    """Fixed GQA/XXL-convolution KiwiLM 2 language model."""

    def __init__(self, config: KiwiLM2Config | None = None) -> None:
        super().__init__()
        model_config = config or KiwiLM2Config()
        if not isinstance(model_config, KiwiLM2Config):
            raise TypeError("KiwiLM2LM requires a KiwiLM2Config")
        self.config = model_config
        self.token_embedding = nn.Embedding(model_config.vocab_size, model_config.d_model)
        self.ngram_embedding = NGramEmbedding(model_config)
        self.embedding_scale = math.sqrt(model_config.d_model / 3)
        conv_index = 0
        blocks: list[KiwiLM2Block] = []
        if isinstance(model_config, KiwiLM2SlimV3Config):
            mlp_schedule = model_config.mlp_schedule
        elif isinstance(model_config, KiwiLM2SlimConfig):
            mlp_schedule = ("hadamard",) * len(model_config.mixer_schedule)
        else:
            mlp_schedule = ("swiglu",) * len(model_config.mixer_schedule)
        for mixer_name, mlp_kind in zip(
            model_config.mixer_schedule, mlp_schedule, strict=True
        ):
            if mixer_name == "gqa":
                mixer: KiwiLM2GQA | XXLCausalGatedConv = KiwiLM2GQA(model_config)
            else:
                mixer = XXLCausalGatedConv(
                    model_config.d_model,
                    model_config.conv_kernel_sizes[conv_index],
                    dropout=model_config.dropout,
                )
                conv_index += 1
            blocks.append(KiwiLM2Block(model_config, mixer, mlp_kind=mlp_kind))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(model_config.d_model, eps=model_config.rms_norm_eps)
        self.lm_head = nn.Linear(model_config.d_model, model_config.vocab_size, bias=False)
        self.apply(initialize_weights)
        for module in self.modules():
            if isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        if isinstance(model_config, KiwiLM2SlimConfig):
            for module in self.modules():
                if isinstance(module, HadamardMLP):
                    nn.init.ones_(module.input_scale)
                    nn.init.ones_(module.output_scale)
        residual_std = 0.02 / math.sqrt(2 * len(self.blocks))
        for block in self.blocks:
            output = block.mixer.output
            nn.init.normal_(output.weight, mean=0.0, std=residual_std)
            if isinstance(block.mlp, SwiGLU):
                nn.init.normal_(block.mlp.down.weight, mean=0.0, std=residual_std)
        if model_config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _embed(self, input_ids: Tensor) -> Tensor:
        return (
            self.token_embedding(input_ids) + self.ngram_embedding(input_ids)
        ) * self.embedding_scale

    def forward(self, input_ids: Tensor) -> Tensor:
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self._embed(input_ids)
        for block in self.blocks:
            values = block(values)
        return self.lm_head(self.final_norm(values))

    def prefill(self, input_ids: Tensor) -> tuple[Tensor, KiwiLM2Cache]:
        input_ids = input_ids[:, -self.config.context_length :]
        validate_input_ids(input_ids, context_length=self.config.context_length)
        values = self._embed(input_ids)
        caches: list[Any] = []
        for block in self.blocks:
            values, cache = block.prefill(values)
            caches.append(cache)
        return self.lm_head(self.final_norm(values)), KiwiLM2Cache(input_ids, caches)

    def decode_step(self, input_ids: Tensor, cache: KiwiLM2Cache) -> tuple[Tensor, KiwiLM2Cache]:
        if input_ids.ndim == 1:
            input_ids = input_ids[:, None]
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step input must have shape [batch, 1]")
        if not isinstance(cache, KiwiLM2Cache) or len(cache.mixers) != len(self.blocks):
            raise ValueError("incremental cache has an incompatible structure")
        token_ids = torch.cat((cache.token_ids, input_ids), dim=1)
        if token_ids.shape[1] > self.config.context_length:
            return self.prefill(token_ids[:, -self.config.context_length :])
        # Recreate the last position so its n-gram lookup sees cached predecessors.
        values = self._embed(token_ids)[:, -1:]
        position = cache.token_ids.shape[1]
        caches: list[Any] = []
        for block, block_cache in zip(self.blocks, cache.mixers, strict=True):
            values, block_cache = block.decode_step(values, block_cache, position=position)
            caches.append(block_cache)
        return self.lm_head(self.final_norm(values)), KiwiLM2Cache(token_ids, caches)


def _build_kiwilm2(config: ModelConfig) -> CausalLanguageModel:
    if not isinstance(config, KiwiLM2Config) or isinstance(
        config, (KiwiLM2SlimConfig, KiwiLM2SlimV3Config)
    ):
        raise TypeError("kiwilm2 requires KiwiLM2Config")
    return KiwiLM2LM(config)


def _build_kiwilm2_slim(config: ModelConfig) -> CausalLanguageModel:
    if not isinstance(config, KiwiLM2SlimConfig):
        raise TypeError("kiwilm2_slim requires KiwiLM2SlimConfig")
    return KiwiLM2LM(config)


def _build_kiwilm2_slim_v3(config: ModelConfig) -> CausalLanguageModel:
    if not isinstance(config, KiwiLM2SlimV3Config):
        raise TypeError("kiwilm2_slim_v3 requires KiwiLM2SlimV3Config")
    return KiwiLM2LM(config)


register_model("kiwilm2", _build_kiwilm2)
register_model("kiwilm2_slim", _build_kiwilm2_slim)
register_model("kiwilm2_slim_v3", _build_kiwilm2_slim_v3)


__all__ = [
    "CachedRotaryEmbedding",
    "GatedHadamardMLP",
    "HadamardMLP",
    "KiwiLM2Block",
    "KiwiLM2Cache",
    "KiwiLM2GQA",
    "KiwiLM2LM",
    "NGramEmbedding",
    "RMSNorm",
    "SwiGLU",
    "XXLCausalGatedConv",
    "fast_walsh_hadamard",
]
