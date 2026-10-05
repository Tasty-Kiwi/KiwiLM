"""Recoverable Bernoulli absorbing-mask corruption; no clean-target side channel."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MaskingConfig:
    vocab_size: int = 32_001
    mask_id: int = 32_000
    pad_id: int = 0
    protected_token_ids: tuple[int, ...] = (0, 1, 2, 3)
    min_noise: float = 0.01
    full_mask_probability: float = 0.1

    def __post_init__(self) -> None:
        if type(self.vocab_size) is not int or self.vocab_size < 5:
            raise ValueError("vocab_size must include content and control tokens")
        if type(self.mask_id) is not int or self.mask_id != self.vocab_size - 1:
            raise ValueError("MASK must be the final appended vocabulary ID")
        if type(self.pad_id) is not int or not 0 <= self.pad_id < self.mask_id:
            raise ValueError("pad_id must be inside the clean vocabulary")
        protected = tuple(self.protected_token_ids)
        if any(type(value) is not int or not 0 <= value < self.mask_id for value in protected):
            raise ValueError("protected IDs must be inside the clean vocabulary")
        if len(set(protected)) != len(protected) or self.pad_id not in protected:
            raise ValueError("protected IDs must be unique and include padding")
        object.__setattr__(self, "protected_token_ids", protected)
        for name in ("min_noise", "full_mask_probability"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.min_noise < 1 or not 0 <= self.full_mask_probability <= 1:
            raise ValueError("min_noise must be in (0,1); full_mask_probability in [0,1]")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["protected_token_ids"] = list(self.protected_token_ids)
        return result


@dataclass(frozen=True)
class CorruptedBatch:
    input_ids: Tensor
    attention_mask: Tensor
    masked_positions: Tensor
    eligible_positions: Tensor
    noise_level: Tensor


def _boolean_mask(mask: Tensor | None, ids: Tensor, name: str, default: Tensor) -> Tensor:
    if mask is None:
        return default
    if mask.dtype != torch.bool or mask.shape != ids.shape or mask.device != ids.device:
        raise ValueError(f"{name} must be boolean [batch, sequence] on the input device")
    return mask


def corrupt_tokens(
    clean_ids: Tensor,
    config: MaskingConfig,
    *,
    generator: torch.Generator,
    attention_mask: Tensor | None = None,
    protected_mask: Tensor | None = None,
    noise_level: Tensor | float | None = None,
) -> CorruptedBatch:
    """Sample levels on CPU with explicit RNG, transfer masks to the input device.

    Unforced Bernoulli draws can mask zero eligible tokens. Padding/control IDs
    and caller-protected slots are never targets. No random replacement/80-10-10
    policy, target-dependent selection or forced-one-mask bias is applied.
    """
    if (
        clean_ids.ndim != 2
        or not all(clean_ids.shape)
        or clean_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("clean_ids must be nonempty integer [batch, sequence]")
    if clean_ids.min().item() < 0 or clean_ids.max().item() >= config.mask_id:
        raise ValueError("clean IDs must be inside the vocabulary and cannot contain MASK")
    if generator.device.type != "cpu":
        raise ValueError("corruption requires an explicit CPU generator for portable recovery")
    valid = _boolean_mask(attention_mask, clean_ids, "attention_mask", clean_ids != config.pad_id)
    valid = valid & (clean_ids != config.pad_id)  # never promote padding to content
    protected = _boolean_mask(protected_mask, clean_ids, "protected_mask", torch.zeros_like(valid))
    eligible = valid & ~protected
    for token_id in config.protected_token_ids:
        eligible = eligible & (clean_ids != token_id)
    batch_size = clean_ids.shape[0]
    if noise_level is None:
        levels = config.min_noise + (1 - config.min_noise) * torch.rand(
            batch_size, generator=generator
        )
        full = torch.rand(batch_size, generator=generator) < config.full_mask_probability
        levels = torch.where(full, 1.0, levels)
    else:
        if isinstance(noise_level, bool) or not isinstance(noise_level, (Tensor, int, float)):
            raise ValueError("noise_level must be scalar or floating [batch]")
        if isinstance(noise_level, Tensor) and not noise_level.is_floating_point():
            raise ValueError("noise_level tensor must be floating")
        levels = torch.as_tensor(noise_level).detach().to(device="cpu", dtype=torch.float32)
        if levels.ndim == 0:
            levels = levels.expand(batch_size)
        if (
            levels.shape != (batch_size,)
            or not torch.isfinite(levels).all().item()
            or ((levels < 0) | (levels > 1)).any().item()
        ):
            raise ValueError("noise_level must be finite [batch] or scalar in [0,1]")
    selected = torch.rand(clean_ids.shape, generator=generator) < levels[:, None]
    masked = selected.to(clean_ids.device) & eligible
    corrupted = clean_ids.clone().masked_fill(masked, config.mask_id)
    return CorruptedBatch(corrupted, valid, masked, eligible, levels.to(clean_ids.device))
