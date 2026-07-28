"""Safe, resumable training checkpoints.

The checkpoint format intentionally contains only tensors and Python primitive
containers so it can be loaded with ``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_FORMAT_VERSION = 1


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint does not belong to the requested run."""


def config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a model or training configuration to safe primitive values."""

    if config is None:
        return {}
    if isinstance(config, Mapping):
        value = dict(config)
    elif hasattr(config, "to_dict"):
        value = config.to_dict()
    elif is_dataclass(config):
        value = asdict(config)
    else:
        raise TypeError(
            "configuration must be a mapping, dataclass, or expose to_dict()"
        )
    return _normalise(value)


def capture_rng_state() -> dict[str, Any]:
    """Capture process RNG state used by the training runtime."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and torch.backends.mps.is_available()
    ):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore an RNG snapshot produced by :func:`capture_rng_state`."""

    if "python" in state:
        random.setstate(_nested_tuple(state["python"]))
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if (
        "mps" in state
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    step: int,
    model_config: Any | None = None,
    train_config: Any | None = None,
    data_fingerprint: Any = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    generators: Mapping[str, torch.Generator] | None = None,
    batcher: Any | None = None,
    metrics: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save all state needed to resume at an optimizer boundary."""

    if step < 0:
        raise ValueError("step must be non-negative")

    resolved_model_config = (
        model_config if model_config is not None else getattr(model, "config", None)
    )
    batcher_state: dict[str, Any] = {}
    if generators:
        batcher_state["generators"] = {
            name: generator.get_state() for name, generator in generators.items()
        }
    if batcher is not None and hasattr(batcher, "state_dict"):
        batcher_state["owner"] = batcher.state_dict()

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": int(step),
        "model_config": config_to_dict(resolved_model_config),
        "train_config": config_to_dict(train_config),
        "data_fingerprint": _normalise(data_fingerprint),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "rng_state": capture_rng_state(),
        "batcher_state": batcher_state,
        "metrics": _normalise(dict(metrics or {})),
        "training_state": _normalise(dict(training_state or {})),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    expected_model_config: Any | None = None,
    expected_data_fingerprint: Any = None,
    generators: Mapping[str, torch.Generator] | None = None,
    batcher: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint after validating its architecture and prepared data."""

    source = Path(path)
    payload = torch.load(source, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {source} does not contain a mapping")
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {version!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )

    resolved_expected_config = (
        expected_model_config
        if expected_model_config is not None
        else getattr(model, "config", None)
    )
    if resolved_expected_config is not None:
        expected = config_to_dict(resolved_expected_config)
        actual = _normalise(payload.get("model_config", {}))
        if (
            isinstance(actual, dict)
            and actual.get("architecture") == "modern_transformer"
        ):
            actual = {**actual, "architecture": "model_y"}
        if actual != expected:
            raise CheckpointCompatibilityError(
                "checkpoint model configuration does not match the requested model"
            )

    if expected_data_fingerprint is not None:
        expected_fingerprint = _normalise(expected_data_fingerprint)
        actual_fingerprint = _normalise(payload.get("data_fingerprint"))
        if actual_fingerprint != expected_fingerprint:
            raise CheckpointCompatibilityError(
                "checkpoint data fingerprint does not match the prepared dataset"
            )

    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    batcher_state = payload.get("batcher_state") or {}
    saved_generators = batcher_state.get("generators") or {}
    if generators:
        for name, generator in generators.items():
            if name in saved_generators:
                generator.set_state(saved_generators[name])
    if (
        batcher is not None
        and "owner" in batcher_state
        and hasattr(batcher, "load_state_dict")
    ):
        batcher.load_state_dict(batcher_state["owner"])

    rng_state = payload.get("rng_state")
    if restore_rng and rng_state:
        restore_rng_state(rng_state)
    return payload


def _normalise(value: Any) -> Any:
    """Recursively reduce values to weights-only-loadable primitives."""

    if is_dataclass(value):
        return _normalise(asdict(value))
    if isinstance(value, Enum):
        return _normalise(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalise(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None), torch.Tensor)):
        return value
    raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")


def _nested_tuple(value: Any) -> Any:
    """Make older list-normalized Python RNG snapshots acceptable."""

    if isinstance(value, (list, tuple)):
        return tuple(_nested_tuple(item) for item in value)
    return value
