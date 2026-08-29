"""Provenance and promotion rules for the KiwiLM 2 Slim v3 ablation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from kiwilm.config import KiwiLM2SlimV3Config, ModelConfig

SLIM_V3_ROLES = ("dense", "slim_v2", "h7s3", "h6s4")
_EXPECTED_ARCHITECTURES = {
    "dense": "kiwilm2",
    "slim_v2": "kiwilm2_slim",
    "h7s3": "kiwilm2_slim_v3",
    "h6s4": "kiwilm2_slim_v3",
}
_SHARED_MODEL_FIELDS = (
    "vocab_size",
    "context_length",
    "d_model",
    "dropout",
    "tie_embeddings",
    "mixer_schedule",
    "conv_kernel_sizes",
    "num_query_heads",
    "num_kv_heads",
    "swiglu_dim",
    "bigram_buckets",
    "trigram_buckets",
    "rms_norm_eps",
    "rope_base",
)
_FROZEN_TRAIN_FIELDS = (
    "max_steps",
    "batch_size",
    "grad_accum_steps",
    "lr",
    "min_lr",
    "warmup_tokens",
    "max_tokens",
    "batch_mode",
    "precision",
    "weight_decay",
    "beta2",
    "optimizer",
    "grad_clip",
    "seed",
)


def validate_slim_v3_smoke_checkpoints(
    checkpoints: Mapping[str, str | Path],
    *,
    data_fingerprint: str,
    tokenizer_vocab_size: int,
) -> dict[str, Any]:
    """Validate the four controlled smoke checkpoints before comparison."""

    if set(checkpoints) != set(SLIM_V3_ROLES):
        raise ValueError(f"checkpoints must contain exactly: {', '.join(SLIM_V3_ROLES)}")
    records: dict[str, dict[str, Any]] = {}
    reference_model: dict[str, Any] | None = None
    reference_train: dict[str, Any] | None = None
    for role in SLIM_V3_ROLES:
        path = Path(checkpoints[role])
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(payload, dict):
            raise ValueError(f"{role} checkpoint must contain a mapping")
        if payload.get("data_fingerprint") != data_fingerprint:
            raise ValueError(f"{role} checkpoint data fingerprint does not match")
        raw_config = payload.get("model_config")
        if not isinstance(raw_config, dict):
            raise ValueError(f"{role} checkpoint lacks model_config")
        config = ModelConfig.from_dict(raw_config)
        if config.architecture != _EXPECTED_ARCHITECTURES[role]:
            raise ValueError(
                f"{role} checkpoint architecture must be {_EXPECTED_ARCHITECTURES[role]!r}"
            )
        if config.vocab_size != tokenizer_vocab_size:
            raise ValueError(f"{role} checkpoint tokenizer vocabulary does not match")
        if role in {"h7s3", "h6s4"}:
            if not isinstance(config, KiwiLM2SlimV3Config):
                raise ValueError(f"{role} checkpoint is not a Slim v3 configuration")
            expected_upper = 3 if role == "h7s3" else 4
            if config.upper_swiglu_blocks != expected_upper:
                raise ValueError(
                    f"{role} checkpoint must use {expected_upper} upper SwiGLU blocks"
                )
        train_config = payload.get("train_config")
        if not isinstance(train_config, dict):
            raise ValueError(f"{role} checkpoint lacks train_config")
        training_state = payload.get("training_state")
        if not isinstance(training_state, dict):
            raise ValueError(f"{role} checkpoint lacks training_state")
        if training_state.get("tokens_seen") != 50_000_000:
            raise ValueError(f"{role} checkpoint must contain exactly 50M training tokens")
        if train_config.get("optimizer") != "adamw":
            raise ValueError(f"{role} checkpoint must use AdamW")
        model_values = {name: getattr(config, name) for name in _SHARED_MODEL_FIELDS}
        train_values = {name: train_config.get(name) for name in _FROZEN_TRAIN_FIELDS}
        if reference_model is None:
            reference_model = model_values
            reference_train = train_values
        else:
            if model_values != reference_model:
                raise ValueError(f"{role} checkpoint does not share the frozen backbone")
            if train_values != reference_train:
                raise ValueError(f"{role} checkpoint does not share the frozen training setup")
        records[role] = {
            "checkpoint": str(path),
            "architecture": config.architecture,
            "upper_swiglu_blocks": getattr(config, "upper_swiglu_blocks", None),
            "tokens_seen": training_state["tokens_seen"],
            "train_config": train_values,
        }
        del payload
    return {
        "data_fingerprint": data_fingerprint,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "checkpoints": records,
    }


def select_slim_v3_candidate(
    *,
    dense_tokens_per_second: float | None,
    h7s3_validation_loss: float | None,
    h6s4_validation_loss: float | None,
    h6s4_tokens_per_second: float | None,
    h7s3_health_passed: bool,
    h6s4_health_passed: bool,
    h7s3_parity_passed: bool,
    h6s4_parity_passed: bool,
) -> dict[str, Any]:
    """Apply the frozen Slim v3 promotion gate to measured smoke results."""

    measurements = {
        "dense_tokens_per_second": dense_tokens_per_second,
        "h7s3_validation_loss": h7s3_validation_loss,
        "h6s4_validation_loss": h6s4_validation_loss,
        "h6s4_tokens_per_second": h6s4_tokens_per_second,
    }
    if not all(
        (
            h7s3_health_passed,
            h6s4_health_passed,
            h7s3_parity_passed,
            h6s4_parity_passed,
        )
    ):
        return {
            "selected": None,
            "reason": "both Slim v3 candidates must pass health and parity checks",
            "measurements": measurements,
        }
    if any(
        value is None or not math.isfinite(value) or value <= 0
        for value in measurements.values()
    ):
        return {
            "selected": None,
            "reason": "all required validation-loss and throughput measurements are required",
            "measurements": measurements,
        }
    assert dense_tokens_per_second is not None
    assert h7s3_validation_loss is not None
    assert h6s4_validation_loss is not None
    assert h6s4_tokens_per_second is not None
    loss_improvement = h7s3_validation_loss - h6s4_validation_loss
    dense_throughput_ratio = h6s4_tokens_per_second / dense_tokens_per_second
    loss_gate_passed = loss_improvement >= 0.03
    throughput_gate_passed = dense_throughput_ratio >= 1.10
    selected = "h6s4" if loss_gate_passed and throughput_gate_passed else "h7s3"
    return {
        "selected": selected,
        "reason": (
            "H6/S4 passed both promotion gates"
            if selected == "h6s4"
            else "H6/S4 did not pass both gates; select the cheaper H7/S3 schedule"
        ),
        "loss_improvement_h6s4_over_h7s3": loss_improvement,
        "h6s4_to_dense_throughput_ratio": dense_throughput_ratio,
        "loss_gate": {"minimum": 0.03, "passed": loss_gate_passed},
        "throughput_gate": {"minimum": 1.10, "passed": throughput_gate_passed},
        "measurements": measurements,
    }


__all__ = [
    "SLIM_V3_ROLES",
    "select_slim_v3_candidate",
    "validate_slim_v3_smoke_checkpoints",
]
