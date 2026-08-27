#!/usr/bin/env python3
"""Write a validated KiwiLM 2 Colab job specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kiwilm.colab_kiwilm2 import ARCHITECTURES, OPTIMIZERS, PHASE_TOKENS, build_colab_job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(PHASE_TOKENS), required=True)
    parser.add_argument("--architecture", choices=tuple(sorted(ARCHITECTURES)), required=True)
    parser.add_argument("--optimizer", choices=tuple(sorted(OPTIMIZERS)), default="adamw")
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-data-token-mismatch", action="store_true")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/KiwiLM2")
    parser.add_argument("--no-drive-backups", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    job = build_colab_job(
        args.data_dir,
        phase=args.phase,
        architecture=args.architecture,
        optimizer=args.optimizer,
        muon_lr=args.muon_lr,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        precision=args.precision,
        seed=args.seed,
        allow_data_token_mismatch=args.allow_data_token_mismatch,
        drive_backups=not args.no_drive_backups,
        drive_root=args.drive_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
