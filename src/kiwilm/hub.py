"""Dataset-free, revision-pinned Hugging Face inference for custom KiwiLM bundles."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torch import nn

from kiwilm.config import ModelConfig
from kiwilm.safetensors_io import BUNDLE_FILES, load_safetensors_model
from kiwilm.tokenizer import ByteBPETokenizer


def load_pretrained(
    repo_id: str, *, revision: str, device: torch.device | None = None,
    cache_dir: str | Path | None = None, local_files_only: bool = False,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, ByteBPETokenizer, ModelConfig]:
    """Download only inference files at an immutable Hub commit; execute no remote code."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("revision must be a full 40-character Hugging Face commit SHA")
    root = Path(snapshot_download(
        repo_id=repo_id, repo_type="model", revision=revision,
        allow_patterns=list(BUNDLE_FILES), cache_dir=cache_dir,
        local_files_only=local_files_only,
    ))
    model, config = load_safetensors_model(
        root, data_fingerprint=None, device=device if device is not None else torch.device("cpu"),
        dtype=dtype,
    )
    return model, ByteBPETokenizer.load(root / "tokenizer.json"), config
