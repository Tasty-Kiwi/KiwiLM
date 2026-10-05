"""Fixed-slot bidirectional generation/infilling from V3 inference weights, without data."""

import argparse
import json
from pathlib import Path

from kiwilm.v3.sampling import SamplingConfig, generate_slots
from kiwilm.v3.tokenizer import MaskBPETokenizer
from kiwilm.v3.weights import load_encoder_weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="encoder.pt or encoder.safetensors, not training state",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--output-slots", type=int, default=32)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sampling = SamplingConfig(args.steps, args.temperature, args.top_k, args.seed)
    tokenizer = MaskBPETokenizer.load(args.tokenizer)
    model, _ = load_encoder_weights(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        expected_tokenizer_sha256=tokenizer.fingerprint,
    )
    print(
        json.dumps(
            generate_slots(
                model,
                tokenizer,
                args.prompt,
                suffix=args.suffix,
                output_slots=args.output_slots,
                config=sampling,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
