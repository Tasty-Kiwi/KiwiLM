#!/usr/bin/env python3
"""Evaluate one or more KiwiLM checkpoints on counterfactual retrieval probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kiwilm.data import PreparedTokenData
from kiwilm.inference import load_trained_model
from kiwilm.retrieval import (
    DEFAULT_RETRIEVAL_DISTANCES,
    DEFAULT_RETRIEVAL_PAIRS_PER_DISTANCE,
    build_retrieval_suite,
    evaluate_retrieval_model,
    write_retrieval_artifacts,
)
from kiwilm.training import choose_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--label",
        action="append",
        help="model label; repeat once per checkpoint (defaults to checkpoint stem)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/evaluations/context-retrieval"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument(
        "--distance",
        type=int,
        action="append",
        dest="distances",
        help="needle distance; repeat for multiple distances",
    )
    parser.add_argument(
        "--pairs-per-distance",
        type=int,
        default=DEFAULT_RETRIEVAL_PAIRS_PER_DISTANCE,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = args.label or [checkpoint.stem for checkpoint in args.checkpoint]
    if len(labels) != len(args.checkpoint):
        raise ValueError("--label must be omitted or repeated once per --checkpoint")
    if len(set(labels)) != len(labels):
        raise ValueError("retrieval model labels must be unique")

    data = PreparedTokenData(args.data_dir, seed=args.seed)
    device = choose_device(args.device)
    suite = build_retrieval_suite(
        data.tokenizer,
        context_length=args.context_length,
        distances=tuple(args.distances or DEFAULT_RETRIEVAL_DISTANCES),
        pairs_per_distance=args.pairs_per_distance,
        seed=args.seed,
    )
    evaluations = []
    for label, checkpoint in zip(labels, args.checkpoint, strict=True):
        model, config = load_trained_model(
            checkpoint,
            data_fingerprint=data.fingerprint,
            device=device,
        )
        if config.context_length < args.context_length:
            raise ValueError(
                f"{label} supports {config.context_length} tokens, below the requested "
                f"retrieval context length {args.context_length}"
            )
        evaluations.append(
            evaluate_retrieval_model(
                model,
                suite,
                label=label,
                architecture=config.architecture,
                checkpoint=checkpoint,
                device=device,
                batch_size=args.batch_size,
            )
        )

    artifacts = write_retrieval_artifacts(
        args.output_dir,
        suite=suite,
        evaluations=evaluations,
    )
    print(
        json.dumps(
            {
                **artifacts,
                "data_fingerprint": data.fingerprint,
                "device": str(device),
                "models": [evaluation["summary"] for evaluation in evaluations],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
