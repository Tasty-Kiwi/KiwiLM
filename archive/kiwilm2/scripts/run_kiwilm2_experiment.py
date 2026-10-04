#!/usr/bin/env python3
"""Run the controlled KiwiLM 2 / Slim comparison on one frozen dataset."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kiwilm.config import KiwiLM2Config, KiwiLM2SlimConfig, KiwiLM2SlimV3Config
from kiwilm.data import PreparedTokenData
from kiwilm.diagnostics import (
    cached_generation_parity_report,
    model_health_report,
    model_residual_report,
)
from kiwilm.inference import load_trained_model
from kiwilm.model_profile import profile_kiwilm2
from kiwilm.models import KiwiLM2LM, build_model
from kiwilm.residual_gate import (
    validate_residual_audit_authorization,
    validate_residual_gate_promotion_override,
)
from kiwilm.training import TrainConfig, choose_device, train

PHASE_TOKENS = {
    "smoke": 50_000_000,
    "architecture": 250_000_000,
    "final-500m": 500_000_000,
    "final-1b": 1_000_000_000,
}
CANDIDATES = (
    "dense",
    "slim-v2",
    "slim-v3-h6s4",
    "slim-v3-h6s4-gate-025",
    "slim-v3-h6s4-gate-050",
)
GATED_CANDIDATES = {
    "slim-v3-h6s4-gate-025",
    "slim-v3-h6s4-gate-050",
}


def _uses_residual_gate(config: Any) -> bool:
    return (
        isinstance(config, KiwiLM2SlimV3Config)
        and config.swiglu_residual_gate_init is not None
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--phase", choices=tuple(PHASE_TOKENS), default="smoke")
    result.add_argument("--max-tokens", type=int)
    result.add_argument(
        "--max-steps",
        type=int,
        help="optimizer-step ceiling; defaults to the token budget plus 100 steps",
    )
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--grad-accum-steps", type=int, default=1)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--min-learning-rate", type=float, default=3e-5)
    result.add_argument(
        "--warmup-tokens",
        type=int,
        help="defaults to two percent of the token budget",
    )
    result.add_argument("--precision", choices=("fp32", "fp16", "bf16", "auto"), default="auto")
    result.add_argument("--device", default="auto")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--eval-interval", type=int, default=500)
    result.add_argument(
        "--eval-batches",
        type=int,
        help="defaults to 50 for smoke and 200 for larger phases",
    )
    result.add_argument("--checkpoint-interval", type=int, default=500)
    result.add_argument(
        "--slim-compile-mode",
        choices=("eager", "compiled"),
        default="eager",
        help="runtime for the gated Slim candidate",
    )
    result.add_argument(
        "--candidates",
        nargs="+",
        choices=CANDIDATES,
        default=("dense", "slim-v2"),
        help="AdamW candidates to train; v3-only runs can omit existing controls",
    )
    result.add_argument(
        "--residual-audit",
        type=Path,
        help="required authorization JSON for either residual-gated candidate",
    )
    result.add_argument(
        "--promotion-override",
        type=Path,
        help="required manual decision record for a gated 250M confirmation",
    )
    result.add_argument(
        "--resume-existing",
        action="store_true",
        help="resume each candidate from its output directory's latest.pt",
    )
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
    if len(set(args.candidates)) != len(args.candidates):
        raise ValueError("--candidates cannot contain duplicates")
    resolved_device = choose_device(args.device)
    max_tokens = args.max_tokens if args.max_tokens is not None else PHASE_TOKENS[args.phase]
    if args.batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("batch_size and grad_accum_steps must be positive")
    tokens_per_step = args.batch_size * args.grad_accum_steps * 512
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else math.ceil(max_tokens / tokens_per_step) + 100
    )
    warmup_tokens = (
        args.warmup_tokens if args.warmup_tokens is not None else max(1, max_tokens // 50)
    )
    eval_batches = (
        args.eval_batches
        if args.eval_batches is not None
        else (50 if args.phase == "smoke" else 200)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = PreparedTokenData(args.data_dir, seed=args.seed)
    gated_requested = bool(set(args.candidates) & GATED_CANDIDATES)
    residual_audit = None
    promotion_override = None
    if gated_requested:
        if args.residual_audit is None:
            raise ValueError("gated candidates require --residual-audit")
        residual_audit = validate_residual_audit_authorization(
            args.residual_audit,
            fingerprint=source.fingerprint,
            phase=args.phase,
        )
        if args.phase == "architecture":
            gated_candidates = set(args.candidates) & GATED_CANDIDATES
            if gated_candidates != {"slim-v3-h6s4-gate-050"}:
                raise ValueError(
                    "the manual 250M promotion authorizes only slim-v3-h6s4-gate-050"
                )
            if args.promotion_override is None:
                raise ValueError(
                    "gated architecture runs require --promotion-override"
                )
            promotion_override = validate_residual_gate_promotion_override(
                args.promotion_override,
                fingerprint=source.fingerprint,
                candidate="slim-v3-h6s4-gate-050",
            )
    common: dict[str, Any] = {
        "vocab_size": source.tokenizer.vocab_size,
        "context_length": 512,
        "d_model": 512,
        "dropout": 0.0,
        "tie_embeddings": True,
    }
    available_configs = {
        "dense": ("kiwilm2-adamw", KiwiLM2Config(**common)),
        "slim-v2": (
            "kiwilm2-slim-gated-v2-adamw",
            KiwiLM2SlimConfig(**common),
        ),
        "slim-v3-h6s4": (
            "kiwilm2-slim-v3-h6-s4-adamw",
            KiwiLM2SlimV3Config(**common, upper_swiglu_blocks=4),
        ),
        "slim-v3-h6s4-gate-025": (
            "kiwilm2-slim-v3-h6-s4-gate-025-adamw",
            KiwiLM2SlimV3Config(
                **common,
                upper_swiglu_blocks=4,
                swiglu_residual_gate_init=0.25,
            ),
        ),
        "slim-v3-h6s4-gate-050": (
            "kiwilm2-slim-v3-h6-s4-gate-050-adamw",
            KiwiLM2SlimV3Config(
                **common,
                upper_swiglu_blocks=4,
                swiglu_residual_gate_init=0.5,
            ),
        ),
    }
    configs = dict(available_configs[candidate] for candidate in args.candidates)
    settings = TrainConfig(
        max_steps=max_steps,
        max_tokens=max_tokens,
        warmup_tokens=min(warmup_tokens, max_tokens),
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        precision=args.precision,
        eval_interval=args.eval_interval,
        eval_batches=eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
    )
    manifest: dict[str, Any] = {
        "phase": args.phase,
        "data_dir": str(args.data_dir.resolve()),
        "data_fingerprint": source.fingerprint,
        "tokenizer_vocab_size": source.tokenizer.vocab_size,
        "max_tokens": max_tokens,
        "slim_compile_mode": args.slim_compile_mode,
        "candidates": list(args.candidates),
        "resume_existing": args.resume_existing,
        "residual_audit": residual_audit,
        "promotion_override": promotion_override,
        "shared_train_config": settings.to_dict(),
        "runs": {},
    }
    for label, config in configs.items():
        model = build_model(config)
        assert isinstance(model, KiwiLM2LM)
        profile = profile_kiwilm2(model)
        del model
        run_dir = args.output_dir / label
        resume_from = run_dir / "latest.pt" if args.resume_existing else None
        if resume_from is not None and not resume_from.is_file():
            resume_from = None
        diagnostic_data = PreparedTokenData(args.data_dir, seed=141)
        diagnostic_inputs, _ = diagnostic_data.get_batch(
            "validation",
            batch_size=2,
            context_length=config.context_length,
            device=resolved_device,
        )

        def validation_diagnostic(
            network: Any,
            step: int,
            tokens_seen: int,
            diagnostic_batch: Any = diagnostic_inputs,
        ) -> dict[str, Any] | None:
            if step % 500 and tokens_seen < max_tokens:
                return None
            if not isinstance(network, KiwiLM2LM):
                raise TypeError("residual telemetry requires KiwiLM2LM")
            return model_residual_report(network, diagnostic_batch)

        summary = train(
            config,
            PreparedTokenData(args.data_dir, seed=args.seed),
            run_dir,
            settings,
            device=resolved_device,
            resume_from=resume_from,
            compile_model=(
                isinstance(config, (KiwiLM2SlimConfig, KiwiLM2SlimV3Config))
                and args.slim_compile_mode == "compiled"
            ),
            validation_diagnostic_fn=(
                validation_diagnostic if _uses_residual_gate(config) else None
            ),
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
        cached_generation = cached_generation_parity_report(trained, diagnostic_inputs)
        del trained
        run_record = {
            "config": config.to_dict(),
            "profile": profile,
            "summary": summary,
            "health": health,
            "cached_generation": cached_generation,
        }
        manifest["runs"][label] = run_record
        (run_dir / "summary.json").write_text(
            json.dumps(run_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
        run_dir = args.output_dir / label
        resume_from = run_dir / "latest.pt" if args.resume_existing else None
        if resume_from is not None and not resume_from.is_file():
            resume_from = None
        summary = train(
            config,
            PreparedTokenData(args.data_dir, seed=args.seed),
            run_dir,
            muon_settings,
            device=resolved_device,
            resume_from=resume_from,
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
        run_record = {
            "config": config.to_dict(),
            "optimizer": muon_settings.to_dict(),
            "summary": summary,
            "health": health,
        }
        manifest["runs"][label] = run_record
        (run_dir / "summary.json").write_text(
            json.dumps(run_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest_path = args.output_dir / "experiment.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path.resolve()), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
