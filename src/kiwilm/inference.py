"""Checkpoint reconstruction shared by evaluation, generation, and comparison."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from kiwilm.checkpoint import load_checkpoint
from kiwilm.config import ModelConfig
from kiwilm.models import build_model


def load_trained_model(
    checkpoint_path: str | Path,
    *,
    data_fingerprint: str,
    device: torch.device,
) -> tuple[nn.Module, ModelConfig]:
    """Rebuild an architecture from its checkpoint and validate its dataset."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    serialized_config = payload.get("model_config")
    if not isinstance(serialized_config, dict):
        raise ValueError("checkpoint does not contain a model configuration")
    config = ModelConfig.from_dict(serialized_config)
    model = build_model(config)
    load_checkpoint(
        checkpoint_path,
        model=model,
        expected_model_config=config,
        expected_data_fingerprint=data_fingerprint,
        map_location="cpu",
        restore_rng=False,
    )
    model.to(device)
    model.eval()
    return model, config
