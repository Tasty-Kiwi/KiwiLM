#!/usr/bin/env python3
"""Audit Dense and Slim v3 residual health before gated smoke training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from kiwilm.config import KiwiLM2SlimV3Config
from kiwilm.data import PreparedTokenData
from kiwilm.diagnostics import (
    aggregate_health_reports,
    cached_generation_parity_report,
    model_health_report,
    residual_growth_reproduced,
)
from kiwilm.inference import load_trained_model
from kiwilm.models import KiwiLM2LM
from kiwilm.training import choose_device

EXPECTED_TOKENS = 250_000_000


def _checkpoint_metadata(path: Path, *, fingerprint: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} does not contain a mapping")
    if payload.get("data_fingerprint") != fingerprint:
        raise ValueError(f"checkpoint {path} has a different data fingerprint")
    state = payload.get("training_state")
    tokens_seen = state.get("tokens_seen") if isinstance(state, dict) else None
    if tokens_seen != EXPECTED_TOKENS:
        raise ValueError(
            f"checkpoint {path} must contain exactly {EXPECTED_TOKENS} training tokens"
        )
    train_config = payload.get("train_config")
    if (
        not isinstance(train_config, dict)
        or train_config.get("max_tokens") != EXPECTED_TOKENS
    ):
        raise ValueError(f"checkpoint {path} does not use the 250M training budget")
    return {
        "checkpoint": str(path),
        "step": payload.get("step"),
        "tokens_seen": tokens_seen,
        "train_config": train_config,
    }


def _dtype(precision: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]


def _audit_model(
    checkpoint: Path,
    *,
    data_dir: Path,
    fingerprint: str,
    seeds: tuple[int, ...],
    batches_per_seed: int,
    batch_size: int,
    context_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[KiwiLM2LM, dict[str, Any]]:
    model, config = load_trained_model(
        checkpoint, data_fingerprint=fingerprint, device=device
    )
    if not isinstance(model, KiwiLM2LM):
        raise TypeError("residual audit requires KiwiLM2LM checkpoints")
    model.to(dtype=dtype)
    reports = []
    parity_inputs = None
    for seed in seeds:
        data = PreparedTokenData(data_dir, seed=seed, expected_fingerprint=fingerprint)
        for _ in range(batches_per_seed):
            inputs, targets = data.get_batch(
                "validation",
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            if parity_inputs is None:
                parity_inputs = inputs
            reports.append(model_health_report(model, inputs, targets))
    assert parity_inputs is not None
    return model, {
        "architecture": config.architecture,
        "model_config": config.to_dict(),
        "aggregate": aggregate_health_reports(reports),
        "cached_generation": cached_generation_parity_report(model, parity_inputs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--h6s4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(141, 142))
    parser.add_argument("--batches-per-seed", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    args = parser.parse_args()
    if args.batches_per_seed < 1 or args.batch_size < 1:
        raise ValueError("batch counts and batch size must be positive")
    seeds = tuple(args.seeds)
    if not seeds or any(isinstance(seed, bool) for seed in seeds):
        raise ValueError("at least one integer seed is required")

    source = PreparedTokenData(args.data_dir)
    device = choose_device(args.device)
    checkpoints = {
        "dense": _checkpoint_metadata(args.dense, fingerprint=source.fingerprint),
        "h6s4": _checkpoint_metadata(args.h6s4, fingerprint=source.fingerprint),
    }
    dense_model, dense = _audit_model(
        args.dense,
        data_dir=args.data_dir,
        fingerprint=source.fingerprint,
        seeds=seeds,
        batches_per_seed=args.batches_per_seed,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
        dtype=_dtype(args.precision),
    )
    if dense_model.config.architecture != "kiwilm2":
        raise ValueError("--dense must be a dense KiwiLM 2 checkpoint")
    del dense_model
    h6_model, h6s4 = _audit_model(
        args.h6s4,
        data_dir=args.data_dir,
        fingerprint=source.fingerprint,
        seeds=seeds,
        batches_per_seed=args.batches_per_seed,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
        dtype=_dtype(args.precision),
    )
    h6_config = h6_model.config
    if (
        not isinstance(h6_config, KiwiLM2SlimV3Config)
        or h6_config.upper_swiglu_blocks != 4
        or h6_config.swiglu_residual_gate_init is not None
    ):
        raise ValueError("--h6s4 must be the ungated H6/S4 Slim v3 checkpoint")
    del h6_model

    if checkpoints["dense"]["train_config"] != checkpoints["h6s4"]["train_config"]:
        raise ValueError("Dense and H6/S4 checkpoints have different training controls")

    reproduced = residual_growth_reproduced(h6s4["aggregate"])
    controls_match = (
        seeds == (141, 142)
        and args.batches_per_seed == 50
        and args.batch_size == 2
        and args.context_length == 512
    )
    result = {
        "schema_version": 1,
        "data_fingerprint": source.fingerprint,
        "audit": {
            "seeds": list(seeds),
            "batches_per_seed": args.batches_per_seed,
            "total_batches": len(seeds) * args.batches_per_seed,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "device": str(device),
            "precision": args.precision,
            "residual_threshold": 1.5,
            "minimum_failures": 10,
        },
        "checkpoints": checkpoints,
        "models": {"dense": dense, "h6s4": h6s4},
        "residual_growth_reproduced": reproduced,
        "frozen_controls_match": controls_match,
        "gated_smoke_authorized": reproduced and controls_match,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
