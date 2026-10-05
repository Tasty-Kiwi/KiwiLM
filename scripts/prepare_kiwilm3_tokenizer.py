"""Create an ID-preserving V3 tokenizer with a real MASK token, never overwrite V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kiwilm.tokenizer import ByteBPETokenizer
from kiwilm.v3.tokenizer import MaskBPETokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Existing BPE tokenizer.json")
    parser.add_argument("--output", required=True, type=Path, help="New V3 tokenizer.json path")
    args = parser.parse_args()
    tokenizer = MaskBPETokenizer.from_base(ByteBPETokenizer.load(args.source))
    tokenizer.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "vocab_size": tokenizer.vocab_size,
                "mask_id": tokenizer.mask_id,
                "tokenizer_sha256": tokenizer.fingerprint,
                "existing_ids_preserved": True,
                "training_started": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
