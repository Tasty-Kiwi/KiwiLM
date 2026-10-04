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
BUNDLE_FILES = (MODEL_FILE, CONFIG_FILE, METADATA_FILE, TOKENIZER_FILE, MANIFEST_FILE)
EXPORT_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def export_provenance(
    prepared_metadata: Mapping[str, Any], provenance_path: Path | None = None,
) -> dict[str, str]:
    """Validate tokenizer provenance independently from the checkpoint's data recipe."""
    tokenizer = prepared_metadata.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or not isinstance(tokenizer.get("sha256"), str):
        raise ValueError("prepared data does not contain a tokenizer checksum")
    fingerprint = prepared_metadata.get("fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("prepared data does not contain a fingerprint")
    result = {
        "tokenizer_dataset_fingerprint": fingerprint,
        "tokenizer_sha256": tokenizer["sha256"],
        "checkpoint_data_fingerprint": fingerprint,
    }
    if provenance_path is not None:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(provenance, Mapping):
            raise ValueError("checkpoint provenance must be an object")
        checkpoint_fingerprint = provenance.get("data_fingerprint")
        if not isinstance(checkpoint_fingerprint, str):
            raise ValueError("checkpoint provenance lacks a data_fingerprint")
        if provenance.get("tokenizer_sha256") != tokenizer["sha256"]:
            raise ValueError("checkpoint provenance tokenizer checksum does not match")
        result["checkpoint_data_fingerprint"] = checkpoint_fingerprint
        result["checkpoint_provenance_sha256"] = sha256_file(provenance_path)
    return result


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
    provenance: Mapping[str, str] | None = None,
    dtype: str = "bf16",
) -> dict[str, Any]:
    """Export model weights and reconstruction metadata without optimizer state."""

    checkpoint = Path(checkpoint_path)
    destination = Path(output_dir)
    tokenizer_source = Path(tokenizer_path)
    if dtype not in EXPORT_DTYPES:
        raise ValueError("export dtype must be bf16 or fp32")
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
    if not all(torch.isfinite(tensor).all().item() for tensor in state.values()):
        raise ValueError("checkpoint model weights must all be finite")
    if config.tie_embeddings and not torch.equal(
        state["token_embedding.weight"], state["lm_head.weight"]
    ):
        raise ValueError("checkpoint tied embedding/head weights disagree")
    model.load_state_dict(state, strict=True)
    if provenance is not None and (
        provenance.get("checkpoint_data_fingerprint") != data_fingerprint
        or provenance.get("tokenizer_sha256") != tokenizer_sha256
    ):
        raise ValueError("export provenance does not match checkpoint/tokenizer")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        # Clone every entry so tied tensors remain independently addressable by
        # ordinary state_dict loaders. The in-memory model re-establishes tying.
        portable_state = {
            name: tensor.detach().cpu().to(
                dtype=EXPORT_DTYPES[dtype] if tensor.is_floating_point() else tensor.dtype,
            ).contiguous().clone()
            for name, tensor in state.items()
        }
        if not all(torch.isfinite(tensor).all().item() for tensor in portable_state.values()):
            raise ValueError("export dtype conversion produced non-finite weights")
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
            "weights_dtype": dtype,
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
            "weights_dtype": dtype,
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
            "export_provenance": dict(provenance) if provenance is not None else None,
        }
        _write_json(temporary / METADATA_FILE, metadata)
        files = {
            name: {
                "sha256": sha256_file(temporary / name),
                "bytes": (temporary / name).stat().st_size,
            }
            for name in (MODEL_FILE, CONFIG_FILE, METADATA_FILE, TOKENIZER_FILE)
        }
        manifest = {"format": SAFETENSORS_FORMAT, "variant": variant,
                    "weights_dtype": dtype, "files": files}
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
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, ModelConfig]:
    """Reconstruct a KiwiLM model from a portable Safetensors bundle."""

    model_path = _resolve_model_path(path)
    if Path(path).is_dir() or (model_path.parent / MANIFEST_FILE).exists():
        verify_safetensors_bundle(model_path.parent)
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
    # Preserve portable FP32 execution by default; storage precision is independent.
    model = build_model(config).to(dtype=dtype)
    state = load_file(model_path, device="cpu")
    stored_dtype = metadata.get("weights_dtype")
    if stored_dtype is not None and (
        stored_dtype not in EXPORT_DTYPES
        or any(t.is_floating_point() and t.dtype != EXPORT_DTYPES[stored_dtype]
               for t in state.values())
    ):
        raise ValueError("Safetensors tensor dtype does not match weights_dtype metadata")
    if config.tie_embeddings and not torch.equal(
        state["token_embedding.weight"], state["lm_head.weight"]
    ):
        raise ValueError("Safetensors tied embedding/head weights disagree")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, config


def verify_safetensors_bundle(path: str | Path) -> dict[str, Any]:
    """Check all inference files and reconstruction metadata before using a bundle."""
    root = Path(path)
    manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != SAFETENSORS_FORMAT:
        raise ValueError("unsupported KiwiLM bundle manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(BUNDLE_FILES) - {MANIFEST_FILE}:
        raise ValueError("bundle manifest must describe exactly the four inference files")
    for name, details in files.items():
        target = root / name
        if (not isinstance(details, dict) or not target.is_file()
                or target.stat().st_size != details.get("bytes")
                or sha256_file(target) != details.get("sha256")):
            raise ValueError(f"bundle integrity check failed for {name}")
    embedded = read_safetensors_metadata(root / MODEL_FILE)
    config = json.loads((root / CONFIG_FILE).read_text(encoding="utf-8"))
    metadata = json.loads((root / METADATA_FILE).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise ValueError("bundle configuration/metadata must be objects")
    if (config != json.loads(embedded["model_config"])
            or metadata.get("model_config") != config
            or metadata.get("format") != SAFETENSORS_FORMAT
            or metadata.get("variant") != manifest.get("variant")
            or metadata.get("variant") != embedded.get("variant")
            or metadata.get("tokenizer_sha256") != files[TOKENIZER_FILE]["sha256"]):
        raise ValueError("bundle reconstruction metadata disagrees")
    for key in ("architecture", "checkpoint_sha256", "data_fingerprint"):
        if metadata.get(key) != embedded.get(key):
            raise ValueError(f"bundle metadata disagrees on {key}")
    if embedded.get("weights_dtype") is not None and (
        embedded["weights_dtype"] not in EXPORT_DTYPES
        or metadata.get("weights_dtype") != embedded["weights_dtype"]
        or manifest.get("weights_dtype") != embedded["weights_dtype"]
    ):
        raise ValueError("bundle storage dtype metadata disagrees")
    if ByteBPETokenizer.load(root / TOKENIZER_FILE).vocab_size != config.get("vocab_size"):
        raise ValueError("bundle tokenizer vocabulary does not match the model")
    return manifest


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
    "BUNDLE_FILES",
    "CONFIG_FILE",
    "MANIFEST_FILE",
    "METADATA_FILE",
    "MODEL_FILE",
    "SAFETENSORS_FORMAT",
    "TOKENIZER_FILE",
    "export_provenance",
    "export_safetensors_bundle",
    "load_safetensors_model",
    "read_safetensors_metadata",
    "sha256_file",
    "verify_safetensors_bundle",
]
