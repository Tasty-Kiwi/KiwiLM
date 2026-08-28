"""Same-device eager/compiled training-step benchmarks for KiwiLM 2 Slim."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any

import torch
from torch.nn import functional as F

from kiwilm.config import KiwiLM2Config, KiwiLM2SlimConfig
from kiwilm.models.kiwilm2 import KiwiLM2LM


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device, precision: str) -> Any:
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=precision != "fp32",
    )


def _measure(
    config: KiwiLM2Config,
    *,
    device: torch.device,
    batch_size: int,
    precision: str,
    compiled: bool,
    warmup_iterations: int,
    measured_iterations: int,
    compile_backend: str | None,
) -> dict[str, Any]:
    torch.manual_seed(17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(17)
    model = KiwiLM2LM(config).to(device).train()
    if compiled:
        compile_kwargs: dict[str, Any] = {"dynamic": False}
        if compile_backend is not None:
            compile_kwargs["backend"] = compile_backend
        model.compile(**compile_kwargs)
    generator = torch.Generator(device=device).manual_seed(29)
    inputs = torch.randint(
        config.vocab_size,
        (batch_size, config.context_length),
        device=device,
        generator=generator,
    )
    targets = torch.randint(
        config.vocab_size,
        (batch_size, config.context_length),
        device=device,
        generator=generator,
    )

    def step() -> float:
        model.zero_grad(set_to_none=True)
        with _autocast(device, precision):
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        return float(loss.detach())

    _synchronize(device)
    warmup_started = time.perf_counter()
    reference_loss = 0.0
    for _ in range(warmup_iterations):
        reference_loss = step()
    _synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started

    durations: list[float] = []
    for _ in range(measured_iterations):
        _synchronize(device)
        started = time.perf_counter()
        reference_loss = step()
        _synchronize(device)
        durations.append(time.perf_counter() - started)
    gradient_norms = [
        parameter.grad.detach().float().norm()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    reference_gradient_norm = float(torch.stack(gradient_norms).norm())
    tokens = batch_size * config.context_length
    median_seconds = statistics.median(durations)
    result = {
        "compiled": compiled,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "warmup_seconds": warmup_seconds,
        "median_step_seconds": median_seconds,
        "median_tokens_per_second": tokens / median_seconds,
        "loss": reference_loss,
        "gradient_norm": reference_gradient_norm,
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def select_slim_runtime(benchmark: dict[str, Any]) -> tuple[str, str]:
    """Select compiled only when it is valid, fastest, and beats Dense eager."""

    eager = benchmark["slim_eager"]
    dense = benchmark["dense_eager"]
    compiled = benchmark.get("slim_compiled")
    if not isinstance(compiled, dict):
        return "eager", "compiled benchmark was unavailable"
    if not benchmark.get("compiled_parity", False):
        return "eager", "compiled eager/parity check failed"
    compiled_speed = float(compiled["median_tokens_per_second"])
    eager_speed = float(eager["median_tokens_per_second"])
    dense_speed = float(dense["median_tokens_per_second"])
    if compiled_speed <= eager_speed:
        return "eager", "compiled Slim was not the fastest Slim path"
    if compiled_speed <= dense_speed:
        return "eager", "compiled Slim did not beat Dense eager"
    return "compiled", "compiled Slim was fastest and beat Dense eager"


def benchmark_slim_runtime(
    dense_config: KiwiLM2Config,
    slim_config: KiwiLM2SlimConfig,
    *,
    device: torch.device,
    batch_size: int,
    precision: str,
    warmup_iterations: int = 3,
    measured_iterations: int = 10,
    compile_backend: str | None = None,
) -> dict[str, Any]:
    """Benchmark Dense eager and gated Slim eager/compiled on one device."""

    if slim_config.hadamard_variant != "gated_v2":
        raise ValueError("compile selection is defined only for gated Slim v2")
    common = {
        "device": device,
        "batch_size": batch_size,
        "precision": precision,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "compile_backend": compile_backend,
    }
    result: dict[str, Any] = {
        "dense_eager": _measure(dense_config, compiled=False, **common),
        "slim_eager": _measure(slim_config, compiled=False, **common),
        "slim_compiled": None,
        "compiled_error": None,
    }
    try:
        compiled = _measure(slim_config, compiled=True, **common)
        result["slim_compiled"] = compiled
        eager = result["slim_eager"]
        result["compiled_parity"] = math.isclose(
            float(eager["loss"]),
            float(compiled["loss"]),
            rel_tol=2e-3,
            abs_tol=2e-3,
        ) and math.isclose(
            float(eager["gradient_norm"]),
            float(compiled["gradient_norm"]),
            rel_tol=5e-3,
            abs_tol=5e-3,
        )
    except Exception as error:  # pragma: no cover - backend-specific failure path
        result["compiled_parity"] = False
        result["compiled_error"] = f"{type(error).__name__}: {error}"
    selected, reason = select_slim_runtime(result)
    result["selected_runtime"] = selected
    result["selection_reason"] = reason
    fastest_slim = max(
        float(result["slim_eager"]["median_tokens_per_second"]),
        float((result["slim_compiled"] or {}).get("median_tokens_per_second", 0.0)),
    )
    dense_speed = float(result["dense_eager"]["median_tokens_per_second"])
    result["promotion_throughput_ratio"] = fastest_slim / dense_speed
    result["promotion_throughput_passed"] = fastest_slim >= 1.05 * dense_speed
    return result


__all__ = ["benchmark_slim_runtime", "select_slim_runtime"]
