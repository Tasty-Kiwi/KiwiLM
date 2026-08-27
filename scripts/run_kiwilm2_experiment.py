#!/usr/bin/env python3
"""Run the controlled KiwiLM 2 / Slim comparison on one frozen dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kiwilm.config import KiwiLM2Config, KiwiLM2SlimConfig
from kiwilm.data import PreparedTokenData
from kiwilm.diagnostics import model_health_report
from kiwilm.inference import load_trained_model
from kiwilm.model_profile import profile_kiwilm2
from kiwilm.models import KiwiLM2LM, build_model
from kiwilm.training import TrainConfig, choose_device, train

PHASE_TOKENS = {
    "smoke": 50_000_000,
    "architecture": 250_000_000,
    "final-500m": 500_000_000,
    "final-1b": 1_000_000_000,
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--phase", choices=tuple(PHASE_TOKENS), default="smoke")
    result.add_argument("--max-tokens", type=int)
    result.add_argument("--max-steps", type=int, default=1_000_000)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--grad-accum-steps", type=int, default=1)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--min-learning-rate", type=float, default=3e-5)
    result.add_argument("--warmup-tokens", type=int, default=1_000_000)
    result.add_argument("--precision", choices=("fp32", "fp16", "bf16", "auto"), default="auto")
    result.add_argument("--device", default="auto")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--eval-interval", type=int, default=500)
    result.add_argument("--eval-batches", type=int, default=50)
    result.add_argument("--checkpoint-interval", type=int, default=500)
    result.add_argument(
        "--muon-lrs",
        type=float,
        nargs="*",
        default=(),
        help="optional KiwiLM 2-only Muon sweep, e.g. 0.01 0.02 0.04",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    resolved_device = choose_device(args.device)
    max_tokens = args.max_tokens or PHASE_TOKENS[args.phase]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = PreparedTokenData(args.data_dir, seed=args.seed)
    common: dict[str, Any] = {
        "vocab_size": source.tokenizer.vocab_size,
        "context_length": 512,
        "d_model": 512,
        "dropout": 0.0,
        "tie_embeddings": True,
    }
    configs = {
        "kiwilm2-adamw": KiwiLM2Config(**common),
        "kiwilm2-slim-adamw": KiwiLM2SlimConfig(**common),
    }
    settings = TrainConfig(
        max_steps=args.max_steps,
        max_tokens=max_tokens,
        warmup_tokens=min(args.warmup_tokens, max_tokens),
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        precision=args.precision,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
    )
    manifest: dict[str, Any] = {
        "phase": args.phase,
        "data_dir": str(args.data_dir.resolve()),
        "data_fingerprint": source.fingerprint,
        "tokenizer_vocab_size": source.tokenizer.vocab_size,
        "max_tokens": max_tokens,
        "shared_train_config": settings.to_dict(),
        "runs": {},
    }
    for label, config in configs.items():
        model = build_model(config)
        assert isinstance(model, KiwiLM2LM)
        profile = profile_kiwilm2(model)
        del model
        summary = train(
            config,
            PreparedTokenData(args.data_dir, seed=args.seed),
            args.output_dir / label,
            settings,
            device=resolved_device,
        )
        trained, _ = load_trained_model(
            summary["latest_checkpoint"],
            data_fingerprint=source.fingerprint,
            device=resolved_device,
        )
        diagnostic_data = PreparedTokenData(args.data_dir, seed=args.seed + 99)
        diagnostic_inputs, diagnostic_targets = diagnostic_data.get_batch(
            "validation",
            batch_size=min(args.batch_size, 2),
            context_length=config.context_length,
            device=next(trained.parameters()).device,
            generator=None,
        )
        assert isinstance(trained, KiwiLM2LM)
        health = model_health_report(trained, diagnostic_inputs, diagnostic_targets)
        del trained
        manifest["runs"][label] = {
            "config": config.to_dict(),
            "profile": profile,
            "summary": summary,
            "health": health,
        }
    for muon_lr in args.muon_lrs:
        label = f"kiwilm2-muon-{muon_lr:g}"
        config = KiwiLM2Config(**common)
        muon_settings = TrainConfig(
            **{
                **asdict(settings),
                "optimizer": "muon",
                "muon_lr": muon_lr,
            }
        )
        summary = train(
            config,
            PreparedTokenData(args.data_dir, seed=args.seed),
            args.output_dir / label,
            muon_settings,
            device=resolved_device,
        )
        trained, _ = load_trained_model(
            summary["latest_checkpoint"],
            data_fingerprint=source.fingerprint,
            device=resolved_device,
        )
        diagnostic_data = PreparedTokenData(args.data_dir, seed=args.seed + 99)
        diagnostic_inputs, diagnostic_targets = diagnostic_data.get_batch(
            "validation",
            batch_size=min(args.batch_size, 2),
            context_length=config.context_length,
            device=next(trained.parameters()).device,
            generator=None,
        )
        assert isinstance(trained, KiwiLM2LM)
        health = model_health_report(trained, diagnostic_inputs, diagnostic_targets)
        del trained
        manifest["runs"][label] = {
            "config": config.to_dict(),
            "optimizer": muon_settings.to_dict(),
            "summary": summary,
            "health": health,
        }
    manifest_path = args.output_dir / "experiment.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path.resolve()), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
