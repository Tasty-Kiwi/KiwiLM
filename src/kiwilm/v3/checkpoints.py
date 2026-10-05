"""Exclusive local M4 state files; NOT production latest/previous Drive backups."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from kiwilm.checkpoint import capture_rng_state, restore_rng_state
from kiwilm.v3.trainer import DenoisingTrainer

FORMAT = "kiwilm3-m4-local-training-v1"


def _validate_live_contract(trainer: DenoisingTrainer) -> None:
    contract = trainer.contract
    if (
        contract["model"] != trainer.model.config.to_dict()
        or contract["training"] != asdict(trainer.config)
        or contract["masking"] != trainer.masking.to_dict()
        or contract["data_fingerprint"] != trainer.data.fingerprint
        or contract["tokenizer_sha256"] != trainer.tokenizer.fingerprint
        or hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        != trainer.identity
    ):
        raise ValueError("live trainer settings changed; construct a new locked experiment")


def save_training_state(trainer: DenoisingTrainer, path: str | Path) -> Path:
    """Save only between complete steps. Refuse overwrite or artifact reinterpretation."""
    destination = Path(path)
    _validate_live_contract(trainer)
    if not trainer.at_step_boundary:
        raise RuntimeError("cannot checkpoint a failed or incomplete optimizer step")
    if not all(torch.isfinite(p).all().item() for p in trainer.model.parameters()):
        raise FloatingPointError("cannot checkpoint nonfinite model weights")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    payload = {
        "format": FORMAT,
        "contract": trainer.contract,
        "identity": trainer.identity,
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "step": trainer.step,
        "optimizer_steps": trainer.optimizer_steps,
        "tokens_seen": trainer.tokens_seen,
        "data_rng": trainer.data_generator.get_state(),
        "noise_rng": trainer.noise_generator.get_state(),
        "global_rng": capture_rng_state(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".m4-state-", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_training_state(
    trainer: DenoisingTrainer, path: str | Path, *, expected_sha256: str
) -> None:
    """Verify external file integrity and the complete resume contract before loading."""
    source = Path(path)
    _validate_live_contract(trainer)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("M4 checkpoint checksum mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("not an M4 training checkpoint; V2/M3 artifacts use different loaders")
    if payload.get("contract") != trainer.contract or payload.get("identity") != trainer.identity:
        raise ValueError(
            "M4 resume contract mismatch (model, objective, tokenizer, data, code or runtime)"
        )
    for name in ("model", "optimizer", "data_rng", "noise_rng", "global_rng"):
        if name not in payload:
            raise ValueError(f"M4 checkpoint missing required {name}")
    step, updates, tokens = (
        payload.get(name) for name in ("step", "optimizer_steps", "tokens_seen")
    )
    if (
        any(type(value) is not int for value in (step, updates, tokens))
        or not 0 <= updates <= step <= trainer.config.max_steps
        or tokens < 0
        or tokens > step * trainer.config.batch_size * trainer.config.context_length
    ):
        raise ValueError("M4 checkpoint progress is inconsistent")
    if not isinstance(payload["global_rng"], dict) or not {"python", "torch"}.issubset(
        payload["global_rng"]
    ):
        raise ValueError("M4 checkpoint missing global RNG recovery state")
    # Validate generator states before mutating live trainer objects.
    for name in ("data_rng", "noise_rng"):
        torch.Generator().set_state(payload[name])
    trainer.model.load_state_dict(payload["model"], strict=True)
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.data_generator.set_state(payload["data_rng"])
    trainer.noise_generator.set_state(payload["noise_rng"])
    restore_rng_state(payload["global_rng"])
    trainer.step, trainer.optimizer_steps, trainer.tokens_seen = step, updates, tokens
    trainer.at_step_boundary = True
