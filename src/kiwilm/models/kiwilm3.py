"""M3 bidirectional encoder: full attention, symmetric gated BiConv and dense FFNs.

No n-gram path, causal mask, GQA, Hadamard, autoregressive cache or clean-target
input exists here. Phase 4 will supply corruption, MASK policy and a loss.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kiwilm.models.encoder import BidirectionalEncoder
from kiwilm.models.kiwilm2 import CachedRotaryEmbedding, RMSNorm, SwiGLU
from kiwilm.v3.config import KiwiLM3Config


class FullBidirectionalAttention(nn.Module):
    """Equal-width Q/K/V heads with RoPE and a key-validity mask, never a causal mask."""

    def __init__(self, config: KiwiLM3Config) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.dropout = config.dropout
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = CachedRotaryEmbedding(
            self.head_dim, config.context_length, base=config.rope_base
        )

    def forward(self, values: Tensor, attention_mask: Tensor) -> Tensor:
        batch, length, width = values.shape

        def split(projected: Tensor) -> Tensor:
            return projected.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        query, key = self.rope(split(self.query(values))), self.rope(split(self.key(values)))
        value = split(self.value(values))
        # Fully padded rows get a zero-valued sentinel key, avoiding undefined all-masked
        # softmax on other backends. Their outputs are still zeroed below.
        first_position = torch.arange(length, device=values.device)[None] == 0
        allowed_keys = attention_mask | (~attention_mask.any(dim=1, keepdim=True) & first_position)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed_keys[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        merged = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(merged) * attention_mask[..., None]


class GatedBidirectionalConv(nn.Module):
    """Same-length, symmetric odd-kernel depthwise convolution with pointwise gates."""

    def __init__(self, width: int, kernel_size: int, *, dropout: float) -> None:
        super().__init__()
        if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("BiConv kernel must be a positive odd integer")
        self.kernel_size = kernel_size
        self.input = nn.Linear(width, 2 * width, bias=False)
        self.depthwise = nn.Conv1d(
            width, width, kernel_size, padding=kernel_size // 2, groups=width, bias=True
        )
        self.output = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, attention_mask: Tensor) -> Tensor:
        values = values * attention_mask[..., None]
        projected, gate = self.input(values).chunk(2, dim=-1)
        convolved = self.depthwise(projected.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.output(convolved * F.silu(gate))) * attention_mask[..., None]


class NoiseConditioning(nn.Module):
    """Fixed sinusoidal features of a normalized [0,1] level, then a learned MLP."""

    def __init__(self, config: KiwiLM3Config) -> None:
        super().__init__()
        half = config.noise_embedding_dim // 2
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half).float() / half)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.input = nn.Linear(config.noise_embedding_dim, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)

    def forward(self, level: Tensor) -> Tensor:
        angles = level.float()[:, None] * self.frequencies.float()[None] * 1_000
        features = torch.cat((angles.cos(), angles.sin()), dim=-1).to(self.input.weight.dtype)
        return self.output(F.silu(self.input(features)))


class KiwiLM3Block(nn.Module):
    def __init__(self, config: KiwiLM3Config, mixer: str, kernel_size: int | None) -> None:
        super().__init__()
        self.mixer_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        if mixer == "attention":
            self.mixer = FullBidirectionalAttention(config)
        elif mixer == "biconv" and kernel_size is not None:
            self.mixer = GatedBidirectionalConv(config.d_model, kernel_size, dropout=config.dropout)
        else:
            raise ValueError("unknown encoder mixer or missing BiConv kernel")
        self.mlp = SwiGLU(config.d_model, config.swiglu_dim, dropout=config.dropout)

    def forward(self, values: Tensor, attention_mask: Tensor) -> Tensor:
        values = values + self.mixer(self.mixer_norm(values), attention_mask)
        values = values + self.mlp(self.mlp_norm(values))
        return values * attention_mask[..., None]


class KiwiLM3Encoder(BidirectionalEncoder):
    def __init__(self, config: KiwiLM3Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.noise_conditioning = NoiseConditioning(config)
        kernels = iter(config.conv_kernel_sizes)
        self.blocks = nn.ModuleList(
            [
                KiwiLM3Block(config, mixer, next(kernels) if mixer == "biconv" else None)
                for mixer in config.mixer_schedule
            ]
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.reconstruction_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.reconstruction_head.weight = self.token_embedding.weight
        self.reset_parameters()

    def _apply(self, fn, recurse=True):
        # Device transfers (notably XLA) can replace aliased Parameters separately.
        # Re-tie before any caller constructs an optimizer on the transferred model.
        super()._apply(fn, recurse=recurse)
        self.reconstruction_head.weight = self.token_embedding.weight
        return self

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is self.reconstruction_head:
                continue  # tied weight was already initialized as the embedding
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        residual_std = 0.02 / math.sqrt(2 * self.config.num_blocks)
        for block in self.blocks:
            nn.init.normal_(block.mixer.output.weight, std=residual_std)
            nn.init.normal_(block.mlp.down.weight, std=residual_std)

    def _mask(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        if (
            input_ids.ndim != 2
            or not input_ids.shape[0]
            or not 0 < input_ids.shape[1] <= self.config.context_length
        ):
            raise ValueError(
                "input_ids must be nonempty [batch, sequence] within the context window"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")
        # Embedding enforces vocabulary bounds on accelerator inputs as well.
        if input_ids.device.type == "cpu" and (
            input_ids.min().item() < 0 or input_ids.max().item() >= self.config.vocab_size
        ):
            raise ValueError("input_ids contains an ID outside the configured vocabulary")
        if attention_mask is not None:
            if (
                attention_mask.dtype != torch.bool
                or attention_mask.shape != input_ids.shape
                or attention_mask.device != input_ids.device
            ):
                raise ValueError(
                    "attention_mask must be boolean [batch, sequence] on the input device"
                )
            return attention_mask
        if self.config.pad_token_id is not None:
            return input_ids != self.config.pad_token_id
        return torch.ones_like(input_ids, dtype=torch.bool)

    def _noise(self, noise_level: Tensor | float | None, input_ids: Tensor) -> Tensor:
        batch = input_ids.shape[0]
        if noise_level is None:
            return torch.zeros(batch, device=input_ids.device)
        if isinstance(noise_level, bool) or not isinstance(noise_level, (Tensor, float, int)):
            raise ValueError("noise_level must be a normalized scalar or floating [batch] tensor")
        if isinstance(noise_level, Tensor) and not noise_level.is_floating_point():
            raise ValueError("noise_level tensor must be floating point")
        level = torch.as_tensor(noise_level)
        if level.ndim == 0:
            level = level.expand(batch)
        if level.shape != (batch,):
            raise ValueError("noise_level must be scalar or have shape [batch]")
        if not torch.isfinite(level).all().item() or ((level < 0) | (level > 1)).any().item():
            raise ValueError("noise_level must be finite and in [0,1]")
        return level.to(device=input_ids.device, dtype=torch.float32)

    def encode(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        noise_level: Tensor | float | None = None,
    ) -> Tensor:
        valid = self._mask(input_ids, attention_mask)
        level = self._noise(noise_level, input_ids)
        values = self.token_embedding(input_ids)
        condition = self.noise_conditioning(level).to(values.dtype)
        values = (values + condition[:, None]) * valid[..., None]
        for block in self.blocks:
            values = block(values, valid)
        return self.final_norm(values) * valid[..., None]

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        noise_level: Tensor | float | None = None,
    ) -> Tensor:
        return self.reconstruction_head(
            self.encode(input_ids, attention_mask=attention_mask, noise_level=noise_level)
        )
