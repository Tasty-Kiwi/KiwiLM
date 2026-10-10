"""Read-only checkpoint counterfactuals and optional bounded fresh CPU corpus replays."""

import argparse
import json
import sys
from pathlib import Path

import torch

from kiwilm.v3.collapse_isolation import isolate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--replay-steps", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    if not 0 <= args.replay_steps <= 64 or not 1 <= args.threads <= 8:
        parser.error("use 0-64 replay steps and 1-8 CPU threads")
    previous = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    try:
        result = isolate_checkpoint(
            args.run_dir,
            args.data_dir,
            replay_steps=args.replay_steps,
            progress=lambda text: print(text, file=sys.stderr, flush=True),
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
