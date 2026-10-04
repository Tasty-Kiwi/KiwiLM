"""Generate from the published bundle without downloading a training dataset."""

from __future__ import annotations

import argparse

from kiwilm.generation import generate
from kiwilm.hub import load_pretrained
from kiwilm.training import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Tasty-Kiwi/KiwiLM-2")
    parser.add_argument("--revision", required=True, help="full immutable Hub commit SHA")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    device = choose_device(args.device)
    model, tokenizer, config = load_pretrained(
        args.repo_id, revision=args.revision, device=device,
    )
    print(generate(
        model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
        context_length=config.context_length, temperature=0.8, top_k=40,
        seed=args.seed, device=device, cache="auto",
    ))


if __name__ == "__main__":
    main()
