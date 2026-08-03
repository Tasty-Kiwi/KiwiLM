"""Checkpoint reconstruction shared by evaluation, generation, and comparison."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from kiwilm.checkpoint import load_checkpoint
from kiwilm.config import ModelConfig
from kiwilm.models import build_model
from kiwilm.safetensors_io import load_safetensors_model


def load_trained_model(
    checkpoint_path: str | Path,
    *,
    data_fingerprint: str | None,
    device: torch.device,
) -> tuple[nn.Module, ModelConfig]:
    """Rebuild a checkpoint, optionally validating its prepared dataset."""

    source = Path(checkpoint_path)
    if source.is_dir() or source.suffix == ".safetensors":
        return load_safetensors_model(
            source,
            data_fingerprint=data_fingerprint,
            device=device,
        )

    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    serialized_config = payload.get("model_config")
    if not isinstance(serialized_config, dict):
        raise ValueError("checkpoint does not contain a model configuration")
    config = ModelConfig.from_dict(serialized_config)
    model = build_model(config)
    load_checkpoint(
        source,
        model=model,
        expected_model_config=config,
        expected_data_fingerprint=data_fingerprint,
        map_location="cpu",
        restore_rng=False,
    )
    model.to(device)
    model.eval()
    return model, config
