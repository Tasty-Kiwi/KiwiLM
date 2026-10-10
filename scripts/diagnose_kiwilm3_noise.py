"""Local paired variable-noise versus fixed-15% diagnostic; never allocate cloud."""

import argparse
import json
import sys
from pathlib import Path

import torch

from kiwilm.v3.noise_diagnostic import noise_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("refusing to overwrite output")
    if not 1 <= args.steps <= 128 or not 1 <= args.threads <= 8:
        parser.error("use 1-128 steps and 1-8 CPU threads")
    previous = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    try:
        result = noise_diagnostic(
            args.run_dir,
            args.data_dir,
            steps=args.steps,
            seed=args.seed,
            progress=lambda value: print(value, file=sys.stderr, flush=True),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Saved {args.output}")
    finally:
        torch.set_num_threads(previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
