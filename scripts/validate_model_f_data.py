#!/usr/bin/env python3
"""Validate the exact frozen-tokenizer dataset expected by Model F."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kiwilm.data import PreparedTokenData

EXPECTED_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
EXPECTED_DATA_FINGERPRINT = (
    "6b2687870c402c5e70e677e8a6c88bb854786c8dcb963f9c734feb022862ed82"
)
EXPECTED_SOURCE_FINGERPRINT = (
    "d2f500e2a85cf7c1a1c1b292b2f186c04782e9443312aaea5f1dc08a561dc764"
)
EXPECTED_TOKENIZER_SHA256 = (
    "0127391ca334542dd206b0bef735b571d3739e5a399e89bbe0b42e79a09d9226"
)


def validate_model_f_data(data_dir: str | Path) -> PreparedTokenData:
    """Load and validate the prepared corpus before Colab allocation."""

    data = PreparedTokenData(
        data_dir,
        expected_fingerprint=EXPECTED_DATA_FINGERPRINT,
    )
    metadata = data.metadata
    tokenizer = metadata["tokenizer"]
    reused_from = tokenizer.get("reused_from")
    if metadata["dataset"].get("resolved_revision") != EXPECTED_REVISION:
        raise ValueError("prepared data uses the wrong TinyStories revision")
    if metadata["splits"]["train"].get("stories") != 750_000:
        raise ValueError("prepared data must contain 750,000 training stories")
    if metadata["splits"]["validation"].get("stories") != 10_000:
        raise ValueError("prepared data must contain 10,000 validation stories")
    if tokenizer.get("sha256") != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("prepared data does not use the frozen 550k tokenizer")
    if reused_from != {
        "dataset_fingerprint": EXPECTED_SOURCE_FINGERPRINT,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
    }:
        raise ValueError("prepared data has invalid frozen-tokenizer provenance")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    data = validate_model_f_data(args.data_dir)
    print(
        json.dumps(
            {
                "data_dir": str(args.data_dir.resolve()),
                "fingerprint": data.fingerprint,
                "train_stories": data.metadata["splits"]["train"]["stories"],
                "validation_stories": data.metadata["splits"]["validation"]["stories"],
                "tokenizer_sha256": data.metadata["tokenizer"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
