"""One-batch health diagnostics for KiwiLM 2 smoke runs."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from kiwilm.models.kiwilm2 import KiwiLM2LM


def _rms(values: Tensor) -> float:
    return float(values.detach().float().square().mean().sqrt())


def _gradient_norm(parameters: Any) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(squared)


def model_health_report(
    model: KiwiLM2LM, input_ids: Tensor, targets: Tensor
) -> dict[str, Any]:
    """Probe finiteness, mixer/MLP activations, gradients, and n-gram hashes."""

    if not isinstance(model, KiwiLM2LM):
        raise TypeError("model_health_report requires a KiwiLM2LM")
    if targets.shape != input_ids.shape:
        raise ValueError("targets must match input_ids")
    activations: dict[str, float] = {}
    handles = []

    def capture(name: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Tensor) -> None:
            activations[name] = _rms(output)

        return hook

    for index, block in enumerate(model.blocks):
        handles.append(block.mixer.register_forward_hook(capture(f"{index}.mixer")))
        handles.append(block.mlp.register_forward_hook(capture(f"{index}.mlp")))
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        logits = model(input_ids)
        loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        loss.backward()
        bigram, trigram = model.ngram_embedding.indices(input_ids)
        total_hashes = input_ids.numel()
        block_health = []
        for index, block in enumerate(model.blocks):
            block_health.append(
                {
                    "index": index,
                    "mixer": model.config.mixer_schedule[index],
                    "mixer_output_rms": activations[f"{index}.mixer"],
                    "mlp_output_rms": activations[f"{index}.mlp"],
                    "mixer_gradient_norm": _gradient_norm(block.mixer.parameters()),
                    "mlp_gradient_norm": _gradient_norm(block.mlp.parameters()),
                }
            )
        return {
            "loss": float(loss.detach()),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "logits_rms": _rms(logits),
            "blocks": block_health,
            "ngram": {
                "bigram_unique_fraction": bigram.unique().numel() / total_hashes,
                "trigram_unique_fraction": trigram.unique().numel() / total_hashes,
                "bigram_table_rms": _rms(model.ngram_embedding.bigram.weight),
                "trigram_table_rms": _rms(model.ngram_embedding.trigram.weight),
                "bigram_gradient_norm": _gradient_norm(
                    model.ngram_embedding.bigram.parameters()
                ),
                "trigram_gradient_norm": _gradient_norm(
                    model.ngram_embedding.trigram.parameters()
                ),
            },
        }
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)


__all__ = ["model_health_report"]
