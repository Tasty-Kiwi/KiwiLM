#!/usr/bin/env python3
"""Validate four Slim v3 smoke checkpoints before producing comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kiwilm.data import PreparedTokenData
from kiwilm.slim_v3 import validate_slim_v3_smoke_checkpoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--slim-v2", type=Path, required=True)
    parser.add_argument("--h7s3", type=Path, required=True)
    parser.add_argument("--h6s4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = PreparedTokenData(args.data_dir)
    result = validate_slim_v3_smoke_checkpoints(
        {
            "dense": args.dense,
            "slim_v2": args.slim_v2,
            "h7s3": args.h7s3,
            "h6s4": args.h6s4,
        },
        data_fingerprint=data.fingerprint,
        tokenizer_vocab_size=data.tokenizer.vocab_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
