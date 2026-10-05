"""Inference-only M3 weights, never mistaken for a resumable diffusion checkpoint.

This separate format leaves V2 loaders and their serialized schema untouched.
Tokenizer bytes/objective/optimizer/noise-sampling state are not bundled here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from kiwilm.models.kiwilm3 import KiwiLM3Encoder
from kiwilm.v3.config import KiwiLM3Config

FORMAT = "kiwilm3-encoder-weights-v1"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _checksum(value: str | None) -> None:
    if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("tokenizer checksum must be a lowercase SHA256 digest")


def save_encoder_weights(
    model: KiwiLM3Encoder,
    path: str | Path,
    *,
    tokenizer_sha256: str | None = None,
    dtype: str | None = None,
) -> Path:
    """Atomically create one .pt or .safetensors file, refusing any overwrite.

    Native weights default to lossless FP32, Safetensors to BF16. Inference
    reconstructs FP32 weights by default.
    """
    destination = Path(path)
    dtype = dtype or ("bf16" if destination.suffix == ".safetensors" else "fp32")
    if destination.suffix not in {".pt", ".safetensors"} or dtype not in DTYPES:
        raise ValueError("choose .pt or .safetensors and dtype fp32 or bf16")
    _checksum(tokenizer_sha256)
    if not isinstance(model, KiwiLM3Encoder):
        raise TypeError("encoder weights require a KiwiLM3Encoder")
    if model.reconstruction_head.weight is not model.token_embedding.weight:
        raise ValueError("encoder reconstruction head must remain tied")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"encoder weights already exist: {destination}")
    state = {
        name: value.detach().to(device="cpu", dtype=DTYPES[dtype]).contiguous()
        for name, value in model.state_dict().items()
        if name != "reconstruction_head.weight"
    }
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("cannot publish non-finite encoder weights")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": FORMAT,
        "config": json.dumps(model.config.to_dict(), sort_keys=True),
        "weights_dtype": dtype,
        "tokenizer_sha256": tokenizer_sha256 or "",
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=".encoder-", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if destination.suffix == ".pt":
            with temporary.open("wb") as stream:
                torch.save({"metadata": metadata, "model_state_dict": state}, stream)
        else:
            save_file(state, temporary, metadata=metadata)
        # Atomic, exclusive publication: another writer cannot be overwritten in a race.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_encoder_weights(
    path: str | Path,
    *,
    expected_config: KiwiLM3Config | None = None,
    expected_tokenizer_sha256: str | None = None,
    expected_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[KiwiLM3Encoder, KiwiLM3Config]:
    """Reconstruct a V3 encoder; require externally supplied provenance when available."""
    source = Path(path)
    _checksum(expected_tokenizer_sha256)
    _checksum(expected_sha256)
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("encoder weights checksum mismatch")
    if source.suffix == ".safetensors":
        with safe_open(source, framework="pt", device="cpu") as stream:
            metadata = stream.metadata() or {}
        state = load_file(source, device="cpu")
    elif source.suffix == ".pt":
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("not an M3 encoder weight artifact")
        metadata, state = payload.get("metadata", {}), payload.get("model_state_dict", {})
    else:
        raise ValueError("encoder weights must be .pt or .safetensors")
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != FORMAT
        or metadata.get("weights_dtype") not in DTYPES
    ):
        raise ValueError(
            "not an M3 encoder weight artifact; V2/resume checkpoints use different loaders"
        )
    config = KiwiLM3Config.from_dict(json.loads(metadata["config"]))
    if expected_config is not None and config != expected_config:
        raise ValueError("encoder configuration mismatch")
    if (
        expected_tokenizer_sha256 is not None
        and metadata.get("tokenizer_sha256") != expected_tokenizer_sha256
    ):
        raise ValueError("encoder tokenizer checksum mismatch")
    if (
        not isinstance(state, dict)
        or "token_embedding.weight" not in state
        or "reconstruction_head.weight" in state
    ):
        raise ValueError("encoder artifact must store the tied embedding exactly once")
    dtype = DTYPES[metadata["weights_dtype"]]
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype != dtype
        or not torch.isfinite(value).all().item()
        for value in state.values()
    ):
        raise ValueError("encoder tensors disagree with storage dtype or contain non-finite values")
    model = KiwiLM3Encoder(config)
    state["reconstruction_head.weight"] = state["token_embedding.weight"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, config
