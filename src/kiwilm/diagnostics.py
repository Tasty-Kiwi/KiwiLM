"""One-batch health diagnostics for KiwiLM 2 smoke runs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from kiwilm.models.kiwilm2 import GatedHadamardMLP, HadamardMLP, KiwiLM2LM, SwiGLU


def _rms(values: Tensor) -> float:
    return float(values.detach().float().square().mean().sqrt())


def _gradient_norm(parameters: Any) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(squared)


def _mlp_residual_scale(module: Any) -> float | None:
    if isinstance(module, GatedHadamardMLP):
        return float(module.residual_scale.detach())
    if isinstance(module, SwiGLU):
        scale = module.effective_residual_scale
        return float(scale.detach()) if scale is not None else None
    return None


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.inf


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
            residual_scale = _mlp_residual_scale(block.mlp)
            post_mixer_rms = residuals[f"{index}.post_mixer"]
            post_mlp_rms = residuals[f"{index}.output"]
            mlp_output_rms = activations[f"{index}.mlp"]
            block_health.append(
                {
                    "index": index,
                    "mixer": model.config.mixer_schedule[index],
                    "mlp_type": (
                        "hadamard"
                        if isinstance(block.mlp, (HadamardMLP, GatedHadamardMLP))
                        else "swiglu"
                        if isinstance(block.mlp, SwiGLU)
                        else type(block.mlp).__name__
                    ),
                    "residual_input_rms": residuals[f"{index}.input"],
                    "post_mixer_residual_rms": post_mixer_rms,
                    "post_mlp_residual_rms": post_mlp_rms,
                    "mixer_output_rms": activations[f"{index}.mixer"],
                    "mlp_output_rms": mlp_output_rms,
                    "mlp_update_to_residual_rms": _ratio(
                        mlp_output_rms, post_mixer_rms
                    ),
                    "residual_amplification": _ratio(
                        post_mlp_rms, post_mixer_rms
                    ),
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
            block["residual_amplification"] <= 1.5 for block in block_health
        )
        family_gradient_ratios: dict[str, float] = {}
        family_output_rms_ratios: dict[str, float] = {}
        for mlp_type in ("hadamard", "swiglu"):
            family_blocks = [
                block for block in block_health if block["mlp_type"] == mlp_type
            ]
            if family_blocks:
                first_gradient = family_blocks[0]["mlp_gradient_norm"]
                family_gradient_ratios[mlp_type] = (
                    family_blocks[-1]["mlp_gradient_norm"] / first_gradient
                    if first_gradient > 0
                    else 0.0
                )
                first_output_rms = family_blocks[0]["mlp_output_rms"]
                family_output_rms_ratios[mlp_type] = (
                    family_blocks[-1]["mlp_output_rms"] / first_output_rms
                    if first_output_rms > 0
                    else 0.0
                )
        deepest_gradient_ratio = (
            next(iter(family_gradient_ratios.values()))
            if len(family_gradient_ratios) == 1
            else None
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
            "family_gradient_ratios_passed": all(
                ratio >= 0.1 for ratio in family_gradient_ratios.values()
            ),
            "bounded_residual_scales": bounded_residual_scales,
        }
        return {
            "loss": float(loss.detach()),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "logits_rms": _rms(logits),
            "blocks": block_health,
            "mlp_family_gradient_ratios": family_gradient_ratios,
            "mlp_family_output_rms_ratios": family_output_rms_ratios,
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


def model_residual_report(model: KiwiLM2LM, input_ids: Tensor) -> dict[str, Any]:
    """Capture forward-only block residual contributions for training telemetry."""

    if not isinstance(model, KiwiLM2LM):
        raise TypeError("model_residual_report requires a KiwiLM2LM")
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
        handles.append(
            block.mlp_norm.register_forward_pre_hook(
                capture_input(f"{index}.post_mixer")
            )
        )
        handles.append(block.mlp.register_forward_hook(capture(f"{index}.mlp")))
        handles.append(block.register_forward_hook(capture_output(f"{index}.output")))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(input_ids)
        blocks = []
        for index, block in enumerate(model.blocks):
            post_mixer_rms = residuals[f"{index}.post_mixer"]
            post_mlp_rms = residuals[f"{index}.output"]
            mlp_output_rms = activations[f"{index}.mlp"]
            blocks.append(
                {
                    "index": index,
                    "mlp_type": (
                        "hadamard"
                        if isinstance(block.mlp, (HadamardMLP, GatedHadamardMLP))
                        else "swiglu"
                    ),
                    "post_mixer_residual_rms": post_mixer_rms,
                    "mlp_output_rms": mlp_output_rms,
                    "post_mlp_residual_rms": post_mlp_rms,
                    "mlp_update_to_residual_rms": _ratio(
                        mlp_output_rms, post_mixer_rms
                    ),
                    "residual_amplification": _ratio(
                        post_mlp_rms, post_mixer_rms
                    ),
                    "mlp_residual_scale": _mlp_residual_scale(block.mlp),
                }
            )
        return {
            "logits_finite": bool(torch.isfinite(logits).all()),
            "logits_rms": _rms(logits),
            "blocks": blocks,
        }
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def aggregate_health_reports(
    reports: Sequence[Mapping[str, Any]], *, residual_threshold: float = 1.5
) -> dict[str, Any]:
    """Aggregate deterministic multi-batch health reports by block."""

    if not reports:
        raise ValueError("reports cannot be empty")
    first_blocks = reports[0].get("blocks")
    if not isinstance(first_blocks, list) or not first_blocks:
        raise ValueError("each report must contain block diagnostics")
    block_count = len(first_blocks)
    if any(len(report.get("blocks", [])) != block_count for report in reports):
        raise ValueError("health reports must contain the same block count")
    metric_names = (
        "post_mixer_residual_rms",
        "mlp_output_rms",
        "post_mlp_residual_rms",
        "mlp_update_to_residual_rms",
        "residual_amplification",
    )
    blocks = []
    for index in range(block_count):
        source_blocks = [report["blocks"][index] for report in reports]
        metrics: dict[str, Any] = {}
        for name in metric_names:
            values = [float(block[name]) for block in source_blocks]
            entry = {
                "median": _quantile(values, 0.5),
                "p90": _quantile(values, 0.9),
                "p95": _quantile(values, 0.95),
                "maximum": max(values),
            }
            if name == "residual_amplification":
                failures = sum(value > residual_threshold for value in values)
                entry["threshold"] = residual_threshold
                entry["threshold_failure_count"] = failures
                entry["threshold_failure_rate"] = failures / len(values)
            metrics[name] = entry
        scales = [
            float(block["mlp_residual_scale"])
            for block in source_blocks
            if block.get("mlp_residual_scale") is not None
        ]
        blocks.append(
            {
                "index": index,
                "mixer": source_blocks[0]["mixer"],
                "mlp_type": source_blocks[0]["mlp_type"],
                "metrics": metrics,
                "mlp_residual_scale": (
                    {
                        "median": _quantile(scales, 0.5),
                        "p90": _quantile(scales, 0.9),
                        "p95": _quantile(scales, 0.95),
                        "minimum": min(scales),
                        "maximum": max(scales),
                    }
                    if scales
                    else None
                ),
            }
        )
    family_names = sorted(
        {
            str(name)
            for report in reports
            for name in report.get("mlp_family_gradient_ratios", {})
        }
    )
    family_gradient_minimum = {
        name: min(
            float(report["mlp_family_gradient_ratios"][name])
            for report in reports
            if name in report.get("mlp_family_gradient_ratios", {})
        )
        for name in family_names
    }
    family_output_rms_minimum = {
        name: min(
            float(report["mlp_family_output_rms_ratios"][name])
            for report in reports
            if name in report.get("mlp_family_output_rms_ratios", {})
        )
        for name in family_names
    }
    return {
        "batch_count": len(reports),
        "health_pass_count": sum(bool(report.get("health_passed")) for report in reports),
        "all_finite": all(
            bool(report.get("health_checks", {}).get("finite"))
            for report in reports
        ),
        "all_nonzero_gradients": all(
            bool(report.get("health_checks", {}).get("nonzero_gradients"))
            for report in reports
        ),
        "minimum_mlp_family_gradient_ratios": family_gradient_minimum,
        "minimum_mlp_family_output_rms_ratios": family_output_rms_minimum,
        "blocks": blocks,
    }


def residual_growth_reproduced(
    aggregate: Mapping[str, Any],
    *,
    block_index: int = 9,
    threshold: float = 1.5,
    minimum_failures: int = 10,
) -> bool:
    """Apply the frozen pre-smoke residual-growth audit gate."""

    blocks = aggregate.get("blocks")
    if not isinstance(blocks, list) or block_index >= len(blocks):
        raise ValueError("aggregate does not contain the requested block")
    amplification = blocks[block_index]["metrics"]["residual_amplification"]
    return bool(
        float(amplification["p90"]) > threshold
        and int(amplification["threshold_failure_count"]) >= minimum_failures
    )


def cached_generation_parity_report(
    model: KiwiLM2LM,
    input_ids: Tensor,
    *,
    rtol: float = 2e-3,
    atol: float = 2e-3,
) -> dict[str, Any]:
    """Compare cached decoding with full forward, including context rollover."""

    if not isinstance(model, KiwiLM2LM):
        raise TypeError("cached_generation_parity_report requires a KiwiLM2LM")
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("parity input must have shape [batch, time] with time >= 2")
    tokens = input_ids[:1, -model.config.context_length :]
    prefix = tokens[:, :-1]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _, cache = model.prefill(prefix)
            cached, cache = model.decode_step(tokens[:, -1:], cache)
            expected = model(tokens)[:, -1:]
            direct_difference = float(
                (cached.float() - expected.float()).abs().max()
            )
            direct_passed = bool(torch.allclose(cached, expected, rtol=rtol, atol=atol))

            rollover_token = tokens[:, :1]
            cached_rollover, _ = model.decode_step(rollover_token, cache)
            cached_rollover = cached_rollover[:, -1:]
            rollover_window = torch.cat((tokens, rollover_token), dim=1)[
                :, -model.config.context_length :
            ]
            expected_rollover = model(rollover_window)[:, -1:]
            rollover_difference = float(
                (cached_rollover.float() - expected_rollover.float()).abs().max()
            )
            rollover_passed = bool(
                torch.allclose(
                    cached_rollover, expected_rollover, rtol=rtol, atol=atol
                )
            )
        return {
            "passed": direct_passed and rollover_passed,
            "direct_passed": direct_passed,
            "rollover_passed": rollover_passed,
            "direct_max_absolute_difference": direct_difference,
            "rollover_max_absolute_difference": rollover_difference,
            "rtol": rtol,
            "atol": atol,
        }
    finally:
        model.train(was_training)


__all__ = [
    "aggregate_health_reports",
    "cached_generation_parity_report",
    "model_health_report",
    "model_residual_report",
    "residual_growth_reproduced",
]
