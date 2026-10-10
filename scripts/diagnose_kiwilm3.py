"""Local-only V3 collapse checks: synthetic overfit and optional native checkpoint audit."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from kiwilm.v3.diagnostics import audit_checkpoint, overfit_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overfit-steps", type=int, default=500)
    parser.add_argument("--overfit-width", type=int, choices=[32, 512], default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer-data-dir",
        type=Path,
        help="Use only this prepared-data tokenizer for synthetic probes",
    )
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.0003, 0.0001])
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument(
        "--run-dir", type=Path, help="Native Drive generation containing manifest.json"
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--audit-batches", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, help="New JSON path; existing files are never overwritten"
    )
    args = parser.parse_args()
    if bool(args.run_dir) != bool(args.data_dir):
        parser.error("--run-dir and --data-dir must be supplied together")
    if args.skip_overfit and args.run_dir is None:
        parser.error("select at least one diagnostic")
    if not 1 <= args.threads <= 8 or not 1 <= args.audit_batches <= 4:
        parser.error("use 1-8 CPU threads and 1-4 audit batches")
    if not 0 <= args.seed < 2**32 - 1:
        parser.error("seed must be between 0 and 2**32-2")
    if not 1 <= args.overfit_steps <= 2000 or not 1 <= len(args.learning_rates) <= 4:
        parser.error("use 1-2000 overfit steps and at most four learning rates")
    if any(not math.isfinite(lr) or not 0 < lr <= 0.01 for lr in args.learning_rates):
        parser.error("learning rates must be finite and in (0, 0.01]")
    if not args.skip_overfit and args.overfit_width == 512 and args.overfit_steps > 300:
        parser.error("full-width local overfit requires explicit --overfit-steps <= 300")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    previous = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    try:
        result = {"schema_version": 1, "cloud_allocated": False, "checkpoint_modified": False}
        if not args.skip_overfit:
            result["overfit_probes"] = []
            for lr in args.learning_rates:
                print(
                    f"CPU overfit: width={args.overfit_width}, seed={args.seed}, "
                    f"lr={lr}, steps={args.overfit_steps}",
                    file=sys.stderr,
                    flush=True,
                )
                probe = overfit_probe(
                    steps=args.overfit_steps,
                    learning_rate=lr,
                    seed=args.seed,
                    width=args.overfit_width,
                    tokenizer_data_dir=args.tokenizer_data_dir,
                )
                result["overfit_probes"].append(probe)
                final = probe["trajectory"][-1]
                print(
                    f"Completed: loss={final['loss']:.6f}, accuracy={final['accuracy']:.2%}, "
                    f"strict_pass={probe['passed']}",
                    file=sys.stderr,
                    flush=True,
                )
        if args.run_dir is not None:
            result["native_checkpoint_audit"] = audit_checkpoint(
                args.run_dir, args.data_dir, batches=args.audit_batches
            )
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        print(text, end="")
    finally:
        torch.set_num_threads(previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
