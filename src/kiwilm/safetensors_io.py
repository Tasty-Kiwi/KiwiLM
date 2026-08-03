"""Portable, inference-only Safetensors bundles for KiwiLM checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from kiwilm.checkpoint import CHECKPOINT_FORMAT_VERSION
from kiwilm.config import ModelConfig
from kiwilm.models import build_model
from kiwilm.tokenizer import ByteBPETokenizer

SAFETENSORS_FORMAT = "kiwilm-safetensors-v1"
MODEL_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.json"
TOKENIZER_FILE = "tokenizer.json"
MANIFEST_FILE = "manifest.json"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def export_safetensors_bundle(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer_path: str | Path,
    expected_data_fingerprint: str | None = None,
    expected_tokenizer_sha256: str | None = None,
    variant: str,
) -> dict[str, Any]:
    """Export model weights and reconstruction metadata without optimizer state."""

    checkpoint = Path(checkpoint_path)
    destination = Path(output_dir)
    tokenizer_source = Path(tokenizer_path)
    if destination.exists():
        raise FileExistsError(f"Safetensors output already exists: {destination}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported KiwiLM checkpoint format")
    serialized_config = payload.get("model_config")
    if not isinstance(serialized_config, Mapping):
        raise ValueError("checkpoint does not contain a model configuration")
    config = ModelConfig.from_dict(dict(serialized_config))
    data_fingerprint = payload.get("data_fingerprint")
    if not isinstance(data_fingerprint, str):
        raise ValueError("checkpoint does not contain a data fingerprint")
    if (
        expected_data_fingerprint is not None
        and data_fingerprint != expected_data_fingerprint
    ):
        raise ValueError("checkpoint data fingerprint does not match export data")

    tokenizer = ByteBPETokenizer.load(tokenizer_source)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match checkpoint model")
    tokenizer_sha256 = sha256_file(tokenizer_source)
    if (
        expected_tokenizer_sha256 is not None
        and tokenizer_sha256 != expected_tokenizer_sha256
    ):
        raise ValueError("tokenizer checksum does not match prepared metadata")

    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise ValueError("checkpoint model state must map names to tensors")
    model = build_model(config)
    model.load_state_dict(state, strict=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        # Clone every entry so tied tensors remain independently addressable by
        # ordinary state_dict loaders. The in-memory model re-establishes tying.
        portable_state = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in state.items()
        }
        tensor_metadata = {
            "format": SAFETENSORS_FORMAT,
            "architecture": config.architecture,
            "model_config": json.dumps(
                config.to_dict(), sort_keys=True, separators=(",", ":")
            ),
            "data_fingerprint": data_fingerprint,
            "checkpoint_sha256": sha256_file(checkpoint),
            "step": str(payload.get("step", "")),
            "variant": variant,
        }
        save_file(
            portable_state,
            temporary / MODEL_FILE,
            metadata=tensor_metadata,
        )
        shutil.copyfile(tokenizer_source, temporary / TOKENIZER_FILE)
        _write_json(temporary / CONFIG_FILE, config.to_dict())

        metadata = {
            "format": SAFETENSORS_FORMAT,
            "variant": variant,
            "architecture": config.architecture,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "step": payload.get("step"),
            "tokens_seen": _nested_value(payload, "training_state", "tokens_seen"),
            "data_fingerprint": data_fingerprint,
            "checkpoint_sha256": sha256_file(checkpoint),
            "tokenizer_sha256": tokenizer_sha256,
            "model_config": config.to_dict(),
            "train_config": payload.get("train_config"),
            "metrics": payload.get("metrics"),
            "initialization": _nested_value(
                payload, "training_state", "initialization"
            ),
        }
        _write_json(temporary / METADATA_FILE, metadata)
        files = {
            name: {
                "sha256": sha256_file(temporary / name),
                "bytes": (temporary / name).stat().st_size,
            }
            for name in (MODEL_FILE, CONFIG_FILE, METADATA_FILE, TOKENIZER_FILE)
        }
        manifest = {"format": SAFETENSORS_FORMAT, "variant": variant, "files": files}
        _write_json(temporary / MANIFEST_FILE, manifest)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def read_safetensors_metadata(path: str | Path) -> dict[str, str]:
    """Read and validate KiwiLM metadata embedded in a Safetensors file."""

    model_path = _resolve_model_path(path)
    with safe_open(model_path, framework="pt", device="cpu") as stream:
        metadata = stream.metadata() or {}
    if metadata.get("format") != SAFETENSORS_FORMAT:
        raise ValueError("unsupported KiwiLM Safetensors format")
    return dict(metadata)


def load_safetensors_model(
    path: str | Path,
    *,
    data_fingerprint: str | None,
    device: torch.device,
) -> tuple[nn.Module, ModelConfig]:
    """Reconstruct a KiwiLM model from a portable Safetensors bundle."""

    model_path = _resolve_model_path(path)
    metadata = read_safetensors_metadata(model_path)
    actual_fingerprint = metadata.get("data_fingerprint")
    if data_fingerprint is not None and actual_fingerprint != data_fingerprint:
        raise ValueError("Safetensors data fingerprint does not match requested data")
    try:
        serialized_config = json.loads(metadata["model_config"])
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError("Safetensors metadata has an invalid model configuration") from error
    if not isinstance(serialized_config, dict):
        raise ValueError("Safetensors model configuration must be an object")
    config = ModelConfig.from_dict(serialized_config)
    model = build_model(config)
    model.load_state_dict(load_file(model_path, device="cpu"), strict=True)
    model.to(device)
    model.eval()
    return model, config


def _resolve_model_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / MODEL_FILE if candidate.is_dir() else candidate


def _nested_value(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CONFIG_FILE",
    "MANIFEST_FILE",
    "METADATA_FILE",
    "MODEL_FILE",
    "SAFETENSORS_FORMAT",
    "TOKENIZER_FILE",
    "export_safetensors_bundle",
    "load_safetensors_model",
    "read_safetensors_metadata",
    "sha256_file",
]
