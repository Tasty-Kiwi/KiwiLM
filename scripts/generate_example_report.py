#!/usr/bin/env python3
"""Generate a Markdown prompt-suite report for one KiwiLM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kiwilm.data import PreparedTokenData
from kiwilm.example_report import generate_example_report
from kiwilm.training import choose_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("eval/story-consistency-prompts.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    data = PreparedTokenData(args.data_dir)
    summary = generate_example_report(
        args.checkpoint,
        data=data,
        suite_path=args.suite,
        output_path=args.output,
        device=device,
        title=args.title,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
