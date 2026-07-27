#!/usr/bin/env python3
"""Verify, reassemble, and safely extract chunked Colab artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kiwilm.colab_artifacts import reassemble_colab_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    extracted = reassemble_colab_artifacts(args.manifest, args.output_dir)
    print(
        json.dumps(
            {"extracted": [str(path) for path in extracted]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
