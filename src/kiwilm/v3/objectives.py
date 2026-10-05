"""Unweighted masked-token CE baseline, not an ELBO or autoregressive perplexity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from kiwilm.models.kiwilm3 import KiwiLM3Encoder
from kiwilm.v3.masking import CorruptedBatch, MaskingConfig, corrupt_tokens

OBJECTIVE = "variable-noise-masked-ce-v1"


@dataclass(frozen=True)
class ReconstructionResult:
    loss: Tensor
    loss_sum: Tensor
    correct: Tensor
    masked_tokens: Tensor


def masked_reconstruction_loss(
    logits: Tensor,
    clean_ids: Tensor,
    masked_positions: Tensor,
    *,
    mask_id: int,
) -> ReconstructionResult:
    """Sum CE over selected positions / number selected across the entire batch.

    The appended MASK output is excluded from clean-token prediction support.
    Every selected token has weight one, irrespective of sampled noise level.
    Nonselected targets are ignored, including padding and protected prompts.
    Zero-mask batches return graph-connected zero and must skip optimizer.step.
    """
    if (
        logits.ndim != 3
        or logits.shape[:2] != clean_ids.shape
        or logits.shape[-1] != mask_id + 1
        or not logits.is_floating_point()
    ):
        raise ValueError("logits must be floating [batch, sequence, clean_vocab + MASK]")
    if (
        clean_ids.dtype not in (torch.int32, torch.int64)
        or clean_ids.device != logits.device
        or masked_positions.device != logits.device
        or masked_positions.dtype != torch.bool
        or masked_positions.shape != clean_ids.shape
    ):
        raise ValueError("targets and boolean masked_positions must match logits")
    selected_targets = clean_ids[masked_positions]
    if selected_targets.numel() and (
        selected_targets.min().item() < 0 or selected_targets.max().item() >= mask_id
    ):
        raise ValueError("selected targets cannot be MASK or outside the clean vocabulary")
    # Dense reduction avoids accelerator-incompatible dynamic selected-logit shapes.
    targets = clean_ids.long().masked_fill(~masked_positions, -100)
    values = F.cross_entropy(
        logits[..., :mask_id].float().transpose(1, 2), targets, reduction="none", ignore_index=-100
    )
    count = masked_positions.sum()
    total = values.sum()
    correct = ((logits[..., :mask_id].argmax(-1) == clean_ids) & masked_positions).sum()
    return ReconstructionResult(total / count.clamp_min(1), total, correct, count)


def denoising_forward(
    model: KiwiLM3Encoder,
    clean_ids: Tensor,
    config: MaskingConfig,
    *,
    generator: torch.Generator,
    attention_mask: Tensor | None = None,
    protected_mask: Tensor | None = None,
    noise_level: Tensor | float | None = None,
) -> tuple[ReconstructionResult, CorruptedBatch]:
    """Only corrupted IDs, validity and noise enter the encoder. No shifted labels."""
    if model.config.vocab_size != config.vocab_size or model.config.pad_token_id != config.pad_id:
        raise ValueError("encoder vocabulary/padding must match the masking configuration")
    corrupted = corrupt_tokens(
        clean_ids,
        config,
        generator=generator,
        attention_mask=attention_mask,
        protected_mask=protected_mask,
        noise_level=noise_level,
    )
    logits = model(
        corrupted.input_ids,
        attention_mask=corrupted.attention_mask,
        noise_level=corrupted.noise_level,
    )
    result = masked_reconstruction_loss(
        logits, clean_ids, corrupted.masked_positions, mask_id=config.mask_id
    )
    return result, corrupted
