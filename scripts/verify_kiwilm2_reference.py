"""Offline frozen-V2 checkpoint/bundle regression; never train, upload or overwrite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kiwilm.generation import generate_tokens
from kiwilm.inference import load_trained_model
from kiwilm.safetensors_io import sha256_file

REFERENCE = Path(__file__).resolve().parents[1] / "releases/kiwilm2-1b/regression-reference.json"


def verify_reference(source: Path, kind: str) -> dict:
    reference = json.loads(REFERENCE.read_text())
    expected = reference[kind]
    weights = source / "model.safetensors" if kind == "bf16_bundle" else source
    if sha256_file(weights) != expected["sha256"]:
        raise ValueError(f"not the frozen V2 {kind}; checkpoint/bundle checksum mismatch")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        # Loading a model initializes throwaway weights; do not change caller RNG state.
        with torch.random.fork_rng(devices=[]):
            model, config = load_trained_model(
                source,
                data_fingerprint=reference["data_fingerprint"],
                device=torch.device("cpu"),
            )
        assert config.architecture == "kiwilm2"
        assert config.context_length == 512
        assert model.lm_head.weight is model.token_embedding.weight
        max_error = 0.0
        max_cache_error = 0.0
        with torch.inference_mode():
            for fixture in expected["fixtures"]:
                length = fixture["length"]
                ids = torch.tensor([[2] + [100 + i % 101 for i in range(length - 1)]])
                full = model(ids)
                cached, cache = model.prefill(ids)
                torch.testing.assert_close(cached, full, rtol=1e-5, atol=2e-5)
                decoded, _ = model.decode_step(torch.tensor([[201]]), cache)
                direct = model(torch.cat((ids, torch.tensor([[201]])), dim=1)[:, -512:])
                # Different GEMM/attention shapes can round differently in FP32.
                # Frozen-reference slices below use a tighter tolerance separately.
                torch.testing.assert_close(decoded[:, -1], direct[:, -1], rtol=1e-5, atol=1e-4)
                max_cache_error = max(
                    max_cache_error, float((decoded[:, -1] - direct[:, -1]).abs().max()),
                )
                for logits, name in ((full, "last_logits"), (decoded, "next_logits")):
                    selected = logits[0, -1, reference["logit_indices"]]
                    frozen = torch.tensor(fixture[name])
                    torch.testing.assert_close(selected, frozen, rtol=1e-5, atol=2e-5)
                    max_error = max(max_error, float((selected - frozen).abs().max()))
            for mode in ("auto", "off"):
                generated = generate_tokens(
                    model,
                    torch.tensor([[2, 100, 200]]),
                    max_new_tokens=8,
                    temperature=0,
                    cache=mode,
                )
                assert generated[0].tolist() == expected["greedy_ids"]
        if sha256_file(weights) != expected["sha256"]:
            raise ValueError("frozen weights changed during verification")
        return {
            "kind": kind,
            "sha256": expected["sha256"],
            "status": "passed",
            "max_selected_logit_error": max_error,
            "max_cache_logit_error": max_cache_error,
            "direct_cached_rollover": "passed",
            "greedy_cached_uncached": "passed",
        }
    finally:
        torch.set_num_threads(previous_threads)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.checkpoint is None and args.bundle is None:
        parser.error("provide --checkpoint and/or --bundle; no weights are downloaded")
    for source, kind in ((args.checkpoint, "training_checkpoint"), (args.bundle, "bf16_bundle")):
        if source is not None:
            print(json.dumps(verify_reference(source, kind)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
