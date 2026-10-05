"""Local M3 qualification only: no optimizer, data download or cloud session."""

from __future__ import annotations

import argparse
import json

from kiwilm.v3.validation import qualify_encoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, choices=(12, 16), default=12)
    args = parser.parse_args()
    report = qualify_encoder(args.depth)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
