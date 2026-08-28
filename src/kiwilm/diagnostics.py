"""One-batch health diagnostics for KiwiLM 2 smoke runs."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from kiwilm.models.kiwilm2 import GatedHadamardMLP, KiwiLM2LM


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
    residuals: dict[str, float] = {}
    handles = []

    def capture(name: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Tensor) -> None:
            activations[name] = _rms(output)

        return hook

    def capture_input(name: str) -> Callable[..., None]:
        def hook(_module: Any, inputs: tuple[Tensor, ...]) -> None:
            residuals[name] = _rms(inputs[0])

        return hook

    def capture_output(name: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Tensor) -> None:
            residuals[name] = _rms(output)

        return hook

    for index, block in enumerate(model.blocks):
        handles.append(block.register_forward_pre_hook(capture_input(f"{index}.input")))
        handles.append(block.mixer.register_forward_hook(capture(f"{index}.mixer")))
        handles.append(
            block.mlp_norm.register_forward_pre_hook(capture_input(f"{index}.post_mixer"))
        )
        handles.append(block.mlp.register_forward_hook(capture(f"{index}.mlp")))
        handles.append(block.register_forward_hook(capture_output(f"{index}.output")))
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
            residual_scale = (
                float(block.mlp.residual_scale.detach())
                if isinstance(block.mlp, GatedHadamardMLP)
                else None
            )
            block_health.append(
                {
                    "index": index,
                    "mixer": model.config.mixer_schedule[index],
                    "residual_input_rms": residuals[f"{index}.input"],
                    "post_mixer_residual_rms": residuals[f"{index}.post_mixer"],
                    "post_mlp_residual_rms": residuals[f"{index}.output"],
                    "mixer_output_rms": activations[f"{index}.mixer"],
                    "mlp_output_rms": activations[f"{index}.mlp"],
                    "mlp_residual_scale": residual_scale,
                    "mixer_gradient_norm": _gradient_norm(block.mixer.parameters()),
                    "mlp_gradient_norm": _gradient_norm(block.mlp.parameters()),
                }
            )
        finite_values = all(
            math.isfinite(float(value))
            for block in block_health
            for value in block.values()
            if isinstance(value, float)
        )
        nonzero_gradients = all(
            block[gradient] > 0
            for block in block_health
            for gradient in ("mixer_gradient_norm", "mlp_gradient_norm")
        )
        bounded_residual_steps = all(
            block["post_mlp_residual_rms"] <= 1.5 * block["post_mixer_residual_rms"]
            for block in block_health
        )
        first_mlp_gradient = block_health[0]["mlp_gradient_norm"]
        deepest_gradient_ratio = (
            block_health[-1]["mlp_gradient_norm"] / first_mlp_gradient
            if first_mlp_gradient > 0
            else 0.0
        )
        residual_scales = [
            block["mlp_residual_scale"]
            for block in block_health
            if block["mlp_residual_scale"] is not None
        ]
        bounded_residual_scales = all(abs(scale) <= 1.0 for scale in residual_scales)
        health_checks = {
            "finite": finite_values and bool(torch.isfinite(logits).all()),
            "nonzero_gradients": nonzero_gradients,
            "bounded_residual_steps": bounded_residual_steps,
            "deepest_to_first_mlp_gradient_ratio": deepest_gradient_ratio,
            "deepest_gradient_ratio_passed": deepest_gradient_ratio >= 0.1,
            "bounded_residual_scales": bounded_residual_scales,
        }
        return {
            "loss": float(loss.detach()),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "logits_rms": _rms(logits),
            "blocks": block_health,
            "health_checks": health_checks,
            "health_passed": all(
                value
                for name, value in health_checks.items()
                if name != "deepest_to_first_mlp_gradient_ratio"
            ),
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
