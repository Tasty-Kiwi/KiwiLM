"""Validated job specifications for KiwiLM 2 Colab training."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from kiwilm.data import PreparedTokenData

PHASE_TOKENS = {
    "smoke": 50_000_000,
    "architecture": 250_000_000,
    "final-500m": 500_000_000,
    "final-1b": 1_000_000_000,
}
ARCHITECTURES = {"kiwilm2", "kiwilm2_slim"}
OPTIMIZERS = {"adamw", "muon"}


def checkpoint_backup_key(job: dict[str, Any]) -> str:
    """Return a readable key that changes with resume-locked job settings."""

    locked_names = (
        "phase",
        "architecture",
        "hadamard_variant",
        "optimizer",
        "muon_lr",
        "max_tokens",
        "max_steps",
        "warmup_tokens",
        "batch_size",
        "grad_accum_steps",
        "learning_rate",
        "min_learning_rate",
        "precision",
        "seed",
        "context_length",
        "d_model",
        "data_cache_key",
    )
    locked = {name: job.get(name) for name in locked_names}
    # Smoke historically used 50 evaluation batches without serializing the
    # value into the job. Preserve those backup keys while separating the more
    # robust architecture/final evaluation profile.
    if job.get("eval_batches", 50) != 50:
        locked["eval_batches"] = job["eval_batches"]
    digest = hashlib.sha256(
        json.dumps(locked, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    prefix = f"{job.get('phase')}-{job.get('architecture')}-{job.get('optimizer')}"
    if job.get("hadamard_variant"):
        prefix += f"-{job['hadamard_variant']}"
    if job.get("optimizer") == "muon":
        prefix += f"-{job.get('muon_lr')}"
    return f"{prefix}-{digest}".replace(".", "p")


def build_colab_job(
    data_dir: str | Path | None = None,
    *,
    phase: str,
    architecture: str,
    optimizer: str = "adamw",
    muon_lr: float = 0.02,
    max_tokens: int | None = None,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    learning_rate: float = 3e-4,
    min_learning_rate: float = 3e-5,
    precision: str = "fp16",
    compile_policy: str = "auto",
    seed: int = 42,
    allow_data_token_mismatch: bool = False,
    drive_backups: bool = True,
    drive_root: str = "/content/drive/MyDrive/KiwiLM2",
) -> dict[str, Any]:
    """Return a JSON-compatible job, optionally validating local prepared data."""

    if phase not in PHASE_TOKENS:
        raise ValueError(f"unknown KiwiLM 2 phase: {phase}")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown KiwiLM 2 architecture: {architecture}")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer: {optimizer}")
    if optimizer == "muon" and architecture != "kiwilm2":
        raise ValueError("Muon is restricted to the dense KiwiLM 2 variant")
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError("Colab precision must be fp16, bf16, or fp32")
    if compile_policy not in {"auto", "eager", "compiled"}:
        raise ValueError("compile_policy must be auto, eager, or compiled")
    for name, value in (
        ("batch_size", batch_size),
        ("grad_accum_steps", grad_accum_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("learning_rate", learning_rate),
        ("min_learning_rate", min_learning_rate),
        ("muon_lr", muon_lr),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if min_learning_rate > learning_rate:
        raise ValueError("min_learning_rate cannot exceed learning_rate")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(drive_backups, bool):
        raise TypeError("drive_backups must be a boolean")
    drive_path = PurePosixPath(drive_root) if isinstance(drive_root, str) else None
    if (
        drive_path is None
        or drive_path.parts[:3] != ("/", "content", "drive")
        or len(drive_path.parts) < 4
        or ".." in drive_path.parts
    ):
        raise ValueError("drive_root must be an absolute path below /content/drive")

    resolved_tokens = max_tokens or PHASE_TOKENS[phase]
    if resolved_tokens < 1:
        raise ValueError("max_tokens must be positive")
    data_fingerprint = None
    tokenizer_sha256 = None
    prepared_tokens = resolved_tokens
    if data_dir is not None:
        data = PreparedTokenData(data_dir)
        metadata = data.metadata
        config = metadata.get("config")
        if not isinstance(config, dict):
            raise ValueError("prepared data does not contain configuration metadata")
        if config.get("python_edu_included") is not False:
            raise ValueError("KiwiLM 2 data must explicitly exclude Python-Edu")
        if config.get("source_configs") != ["fineweb-edu-dedup", "cosmopedia-v2"]:
            raise ValueError("KiwiLM 2 data has the wrong SmolLM source configuration")
        tokenizer = metadata.get("tokenizer")
        if not isinstance(tokenizer, dict) or tokenizer.get("vocab_size") != 32_000:
            raise ValueError("KiwiLM 2 Colab jobs require the frozen 32K tokenizer")
        splits = metadata.get("splits")
        train_split = splits.get("train") if isinstance(splits, dict) else None
        if not isinstance(train_split, dict) or not isinstance(train_split.get("tokens"), int):
            raise ValueError("prepared data does not declare its training token count")
        prepared_tokens = int(train_split["tokens"])
        if prepared_tokens != resolved_tokens and not allow_data_token_mismatch:
            raise ValueError(
                f"prepared split has {prepared_tokens} tokens, but the {phase} job "
                f"requests {resolved_tokens}; explicitly allow a mismatch only for debugging"
            )
        data_fingerprint = data.fingerprint
        tokenizer_sha256 = tokenizer.get("sha256")
    tokens_per_step = batch_size * grad_accum_steps * 512
    required_steps = math.ceil(resolved_tokens / tokens_per_step)
    warmup_tokens = min(max(1, resolved_tokens // 50), resolved_tokens)
    eval_batches = 50 if phase == "smoke" else 200
    return {
        "schema_version": 2,
        "phase": phase,
        "architecture": architecture,
        "hadamard_variant": "gated_v2" if architecture == "kiwilm2_slim" else None,
        "optimizer": optimizer,
        "muon_lr": muon_lr,
        "max_tokens": resolved_tokens,
        "max_steps": required_steps + 100,
        "warmup_tokens": warmup_tokens,
        "eval_batches": eval_batches,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "precision": precision,
        "compile_policy": compile_policy,
        "seed": seed,
        "context_length": 512,
        "d_model": 512,
        "data_fingerprint": data_fingerprint,
        "prepared_train_tokens": prepared_tokens,
        "tokenizer_sha256": tokenizer_sha256,
        "prepare_data_in_vm": data_dir is None,
        "validation_tokens": 2_000_000,
        "tokenizer_train_documents": 100_000,
        "validation_documents_per_source": 10_000,
        "fineweb_probability": 0.7,
        "drive_backups": drive_backups,
        "drive_root": drive_root,
        "data_cache_key": f"{phase}-{resolved_tokens}-seed{seed}",
    }


__all__ = [
    "ARCHITECTURES",
    "OPTIMIZERS",
    "PHASE_TOKENS",
    "build_colab_job",
    "checkpoint_backup_key",
]
