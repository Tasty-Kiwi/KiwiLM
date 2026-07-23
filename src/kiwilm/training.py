"""Architecture-independent next-token training for KiwiLM."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from kiwilm.checkpoint import (
    CheckpointCompatibilityError,
    load_checkpoint,
    save_checkpoint,
)
from kiwilm.config import ModelConfig
from kiwilm.data import PreparedTokenData
from kiwilm.generation import generate as generate_text
from kiwilm.models import build_model


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimizer and reporting settings for the fast TinyStories profile."""

    max_steps: int = 2_000
    batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_batches: int = 20
    checkpoint_interval: int = 500
    log_interval: int = 10
    sample_prompt: str = "Once upon a time"
    sample_tokens: int = 64
    seed: int = 42

    def __post_init__(self) -> None:
        _positive_int("max_steps", self.max_steps)
        _positive_int("batch_size", self.batch_size)
        _positive_int("grad_accum_steps", self.grad_accum_steps)
        _positive_int("eval_batches", self.eval_batches)
        _non_negative_int("warmup_steps", self.warmup_steps)
        _non_negative_int("eval_interval", self.eval_interval)
        _non_negative_int("checkpoint_interval", self.checkpoint_interval)
        _non_negative_int("log_interval", self.log_interval)
        _non_negative_int("sample_tokens", self.sample_tokens)
        if not isinstance(self.sample_prompt, str) or not self.sample_prompt:
            raise ValueError("sample_prompt must be a non-empty string")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.min_lr) or not 0 <= self.min_lr <= self.lr:
            raise ValueError("min_lr must be finite and in [0, lr]")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.beta2) or not 0 < self.beta2 < 1:
            raise ValueError("beta2 must be finite and in (0, 1)")
        if not math.isfinite(self.grad_clip) or self.grad_clip < 0:
            raise ValueError("grad_clip must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` in CUDA, MPS, CPU priority order."""

    if isinstance(requested, torch.device):
        device = requested
    elif requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    else:
        device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    """Seed Python and all available PyTorch device generators."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
        and torch.backends.mps.is_available()
    ):
        torch.mps.manual_seed(seed)


def learning_rate_at_step(step: int, config: TrainConfig) -> float:
    """Return the learning rate for a zero-based optimizer step."""

    if step < 0:
        raise ValueError("step must be non-negative")
    effective_warmup_steps = min(config.warmup_steps, config.max_steps)
    if effective_warmup_steps and step < effective_warmup_steps:
        return config.lr * (step + 1) / effective_warmup_steps
    if config.max_steps <= effective_warmup_steps:
        return config.lr
    decay_progress = (step - effective_warmup_steps) / (
        config.max_steps - effective_warmup_steps
    )
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return config.min_lr + coefficient * (config.lr - config.min_lr)


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate mean next-token cross entropy."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must match the logits batch and sequence dimensions")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data: PreparedTokenData | Any,
    *,
    batch_size: int,
    context_length: int,
    num_batches: int = 20,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
    split: str = "validation",
) -> dict[str, float]:
    """Measure mean loss and perplexity over sampled validation windows."""

    _positive_int("batch_size", batch_size)
    _positive_int("context_length", context_length)
    _positive_int("num_batches", num_batches)
    resolved_device = (
        _model_device(model) if device is None else torch.device(device)
    )
    model.to(resolved_device)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    try:
        for _ in range(num_batches):
            inputs, targets = _get_batch(
                data,
                split,
                batch_size=batch_size,
                context_length=context_length,
                device=resolved_device,
                generator=generator,
            )
            total_loss += float(next_token_loss(model(inputs), targets))
    finally:
        model.train(was_training)

    validation_loss = total_loss / num_batches
    try:
        perplexity = math.exp(validation_loss)
    except OverflowError:
        perplexity = math.inf
    return {
        "validation_loss": validation_loss,
        "perplexity": perplexity,
    }


def train(
    model_config: ModelConfig,
    data: PreparedTokenData,
    output_dir: str | Path,
    train_config: TrainConfig | None = None,
    *,
    device: str | torch.device = "auto",
    resume_from: str | Path | None = None,
    model: nn.Module | None = None,
    log_fn: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Train a model and return a JSON-serializable run summary."""

    settings = train_config or TrainConfig()
    resolved_device = choose_device(device)
    run_directory = Path(output_dir)
    run_directory.mkdir(parents=True, exist_ok=True)
    latest_path = run_directory / "latest.pt"
    best_path = run_directory / "best.pt"
    metrics_path = run_directory / "metrics.jsonl"

    seed_everything(settings.seed)
    network = build_model(model_config) if model is None else model
    network.to(resolved_device)
    network.train()
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    if log_fn is not None:
        log_fn(
            f"model={model_config.architecture} "
            f"parameters={parameter_count:,} device={resolved_device}"
        )
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=settings.lr,
        betas=(0.9, settings.beta2),
        weight_decay=settings.weight_decay,
    )
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(settings.seed)
    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(settings.seed + 1)
    generators = {"train": train_generator, "validation": eval_generator}

    completed_step = 0
    best_validation_loss = math.inf
    best_validation_perplexity = math.inf
    if resume_from is not None:
        _validate_resume_settings(resume_from, settings)
        checkpoint = load_checkpoint(
            resume_from,
            model=network,
            optimizer=optimizer,
            expected_model_config=model_config,
            expected_data_fingerprint=data.fingerprint,
            generators=generators,
            batcher=data,
            map_location="cpu",
        )
        completed_step = int(checkpoint["step"])
        checkpoint_metrics = checkpoint.get("metrics") or {}
        best_validation_loss = float(
            checkpoint_metrics.get("best_validation_loss", math.inf)
        )
        best_validation_perplexity = float(
            checkpoint_metrics.get("best_validation_perplexity", math.inf)
        )
        _truncate_metrics_after(metrics_path, completed_step)

    metric_mode = "a" if resume_from is not None else "w"
    last_log_time = time.perf_counter()
    tokens_since_log = 0
    latest_train_loss: float | None = None
    latest_validation: dict[str, float] | None = None
    generated_sample: str | None = None
    with metrics_path.open(metric_mode, encoding="utf-8") as metric_stream:
        while completed_step < settings.max_steps:
            step_index = completed_step
            current_lr = learning_rate_at_step(step_index, settings)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(settings.grad_accum_steps):
                inputs, targets = _get_batch(
                    data,
                    "train",
                    batch_size=settings.batch_size,
                    context_length=model_config.context_length,
                    device=resolved_device,
                    generator=train_generator,
                )
                loss = next_token_loss(network(inputs), targets)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite training loss at step {completed_step + 1}"
                    )
                (loss / settings.grad_accum_steps).backward()
                accumulated_loss += (
                    loss.detach().item() / settings.grad_accum_steps
                )
                tokens_since_log += targets.numel()

            if settings.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), settings.grad_clip
                )
            optimizer.step()
            completed_step += 1
            latest_train_loss = accumulated_loss

            should_log = (
                completed_step == 1
                or completed_step == settings.max_steps
                or (
                    settings.log_interval > 0
                    and completed_step % settings.log_interval == 0
                )
            )
            if should_log:
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-12)
                tokens_per_second = tokens_since_log / elapsed
                train_metrics = {
                    "event": "train",
                    "step": completed_step,
                    "train_loss": latest_train_loss,
                    "learning_rate": current_lr,
                    "tokens_per_second": tokens_per_second,
                }
                _write_metric(metric_stream, train_metrics)
                if log_fn is not None:
                    log_fn(
                        f"step {completed_step}/{settings.max_steps} "
                        f"loss={latest_train_loss:.4f} "
                        f"lr={current_lr:.3g} tok/s={tokens_per_second:,.0f}"
                    )
                last_log_time = now
                tokens_since_log = 0

            should_evaluate = (
                completed_step == settings.max_steps
                or (
                    settings.eval_interval > 0
                    and completed_step % settings.eval_interval == 0
                )
            )
            if should_evaluate:
                evaluation_started = time.perf_counter()
                latest_validation = evaluate(
                    network,
                    data,
                    batch_size=settings.batch_size,
                    context_length=model_config.context_length,
                    num_batches=settings.eval_batches,
                    device=resolved_device,
                    generator=eval_generator,
                )
                evaluation_metrics = {
                    "event": "validation",
                    "step": completed_step,
                    **latest_validation,
                }
                _write_metric(metric_stream, evaluation_metrics)
                if log_fn is not None:
                    log_fn(
                        f"validation step {completed_step} "
                        f"loss={latest_validation['validation_loss']:.4f} "
                        f"ppl={latest_validation['perplexity']:.2f}"
                    )
                last_log_time += time.perf_counter() - evaluation_started
                if (
                    latest_validation["validation_loss"]
                    < best_validation_loss
                ):
                    best_validation_loss = latest_validation["validation_loss"]
                    best_validation_perplexity = latest_validation["perplexity"]
                    save_checkpoint(
                        best_path,
                        model=network,
                        optimizer=optimizer,
                        step=completed_step,
                        model_config=model_config,
                        train_config=settings,
                        data_fingerprint=data.fingerprint,
                        generators=generators,
                        batcher=data,
                        metrics=_checkpoint_metrics(
                            latest_train_loss,
                            latest_validation,
                            best_validation_loss,
                            best_validation_perplexity,
                        ),
                    )

            should_checkpoint = (
                completed_step == settings.max_steps
                or (
                    settings.checkpoint_interval > 0
                    and completed_step % settings.checkpoint_interval == 0
                )
            )
            if should_checkpoint:
                save_checkpoint(
                    latest_path,
                    model=network,
                    optimizer=optimizer,
                    step=completed_step,
                    model_config=model_config,
                    train_config=settings,
                    data_fingerprint=data.fingerprint,
                    generators=generators,
                    batcher=data,
                    metrics=_checkpoint_metrics(
                        latest_train_loss,
                        latest_validation,
                        best_validation_loss,
                        best_validation_perplexity,
                    ),
                )

        if not latest_path.exists() or completed_step != settings.max_steps:
            save_checkpoint(
                latest_path,
                model=network,
                optimizer=optimizer,
                step=completed_step,
                model_config=model_config,
                train_config=settings,
                data_fingerprint=data.fingerprint,
                generators=generators,
                batcher=data,
                metrics=_checkpoint_metrics(
                    latest_train_loss,
                    latest_validation,
                    best_validation_loss,
                    best_validation_perplexity,
                ),
            )

        tokenizer = getattr(data, "tokenizer", None)
        if settings.sample_tokens and tokenizer is not None:
            generated_sample = generate_text(
                network,
                tokenizer,
                settings.sample_prompt,
                max_new_tokens=settings.sample_tokens,
                context_length=model_config.context_length,
                temperature=0,
                seed=settings.seed,
                device=resolved_device,
            )
            _write_metric(
                metric_stream,
                {
                    "event": "sample",
                    "step": completed_step,
                    "prompt": settings.sample_prompt,
                    "text": generated_sample,
                },
            )
            if log_fn is not None:
                log_fn(f"sample: {generated_sample}")

    resolved_latest_path = str(latest_path.resolve())
    resolved_best_path = str(best_path.resolve()) if best_path.exists() else None
    return {
        "step": completed_step,
        "best_validation_loss": (
            best_validation_loss if math.isfinite(best_validation_loss) else None
        ),
        "best_validation_perplexity": (
            best_validation_perplexity
            if math.isfinite(best_validation_perplexity)
            else None
        ),
        "checkpoint_paths": {
            "latest": resolved_latest_path,
            "best": resolved_best_path,
        },
        "latest_checkpoint": resolved_latest_path,
        "best_checkpoint": resolved_best_path,
        "metrics_path": str(metrics_path.resolve()),
        "device": str(resolved_device),
        "parameter_count": parameter_count,
        "sample": generated_sample,
    }


def _validate_resume_settings(
    checkpoint_path: str | Path,
    settings: TrainConfig,
) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("resume checkpoint must contain a mapping")
    saved = payload.get("train_config")
    if not isinstance(saved, Mapping):
        raise CheckpointCompatibilityError(
            "resume checkpoint does not contain a training configuration"
        )
    current = settings.to_dict()
    allowed_changes = {
        "eval_interval",
        "checkpoint_interval",
        "log_interval",
        "sample_prompt",
        "sample_tokens",
    }
    incompatible = sorted(
        name
        for name, value in current.items()
        if name not in allowed_changes and saved.get(name) != value
    )
    if incompatible:
        raise CheckpointCompatibilityError(
            "resume training configuration differs for locked fields: "
            + ", ".join(incompatible)
        )
    completed_step = payload.get("step")
    if not isinstance(completed_step, int) or completed_step < 0:
        raise ValueError("resume checkpoint has an invalid training step")
    if completed_step > settings.max_steps:
        raise CheckpointCompatibilityError(
            "resume checkpoint step exceeds the requested max_steps"
        )


def _checkpoint_metrics(
    train_loss: float | None,
    validation: Mapping[str, float] | None,
    best_loss: float,
    best_perplexity: float,
) -> dict[str, Any]:
    return {
        "train_loss": train_loss,
        "validation_loss": (
            validation.get("validation_loss") if validation is not None else None
        ),
        "validation_perplexity": (
            validation.get("perplexity") if validation is not None else None
        ),
        "best_validation_loss": best_loss,
        "best_validation_perplexity": best_perplexity,
    }


def _truncate_metrics_after(metrics_path: Path, completed_step: int) -> None:
    """Discard metric records newer than the checkpoint being resumed."""

    if not metrics_path.exists():
        return
    retained: list[str] = []
    for line_number, line in enumerate(
        metrics_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid metrics JSON at {metrics_path}:{line_number}"
            ) from error
        if not isinstance(record, Mapping) or not isinstance(record.get("step"), int):
            raise ValueError(
                f"invalid metrics record at {metrics_path}:{line_number}"
            )
        if record["step"] <= completed_step:
            retained.append(json.dumps(dict(record), sort_keys=True))

    descriptor, temporary_name = tempfile.mkstemp(
        dir=metrics_path.parent,
        prefix=f".{metrics_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            if retained:
                stream.write("\n".join(retained) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, metrics_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_metric(stream: Any, values: Mapping[str, Any]) -> None:
    stream.write(json.dumps(dict(values), sort_keys=True) + "\n")
    stream.flush()


def _get_batch(
    data: Any,
    split: str,
    *,
    batch_size: int,
    context_length: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    get_batch = getattr(data, "get_batch", None)
    if get_batch is None:
        get_batch = getattr(data, "batch", None)
    if get_batch is None:
        raise TypeError("data must expose get_batch()")
    inputs, targets = get_batch(
        split,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
        generator=generator,
    )
    return inputs, targets


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
