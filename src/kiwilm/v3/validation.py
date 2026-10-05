"""Bounded CPU structural qualification, without an optimizer or training dataset."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.profile import profile_encoder
from kiwilm.v3.weights import load_encoder_weights, save_encoder_weights


def qualify_encoder(num_blocks: int = 12) -> dict:
    """Exercise full depth at tiny width; profile production width separately on meta."""
    if num_blocks not in (12, 16):
        raise ValueError("M3 qualification supports 12 or 16 blocks")
    config = KiwiLM3Config(
        num_blocks=num_blocks,
        vocab_size=97,
        context_length=16,
        d_model=16,
        num_heads=2,
        swiglu_dim=48,
        noise_embedding_dim=16,
    )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = build_encoder(config).eval()
            ids = torch.tensor([[2, 4, 5, 6, 3, 0]])
            changed = ids.clone()
            changed[0, 3] = 7
            with torch.no_grad():
                logits = model(ids, noise_level=0.5)
                right_context_delta = logits[:, 1] - model(changed, noise_level=0.5)[:, 1]
                noise_delta = logits - model(ids, noise_level=0.25)
                padded = ids.clone()
                padded[:, -1] = 11
                padding_error = (
                    (logits - model(padded, attention_mask=ids != 0, noise_level=0.5))
                    .abs()
                    .max()
                    .item()
                )
            model(ids, noise_level=0.5).square().mean().backward()
            finite_nonzero_gradients = all(
                p.grad is not None
                and torch.isfinite(p.grad).all().item()
                and p.grad.norm().item() > 0
                for p in model.parameters()
            )
            with tempfile.TemporaryDirectory(prefix="kiwilm3-qualification-") as directory:
                path = save_encoder_weights(model, Path(directory) / "encoder.pt")
                restored, _ = load_encoder_weights(path, expected_config=config)
                with torch.no_grad():
                    roundtrip_error = (logits - restored(ids, noise_level=0.5)).abs().max().item()
            checks = {
                "forward_finite": torch.isfinite(logits).all().item(),
                "all_gradients_finite_nonzero": finite_nonzero_gradients,
                "uses_right_context": right_context_delta.abs().max().item() > 1e-7,
                "noise_conditioning_changes_logits": noise_delta.abs().max().item() > 1e-7,
                "padding_isolated": padding_error == 0,
                "fp32_weight_roundtrip_exact": roundtrip_error == 0,
                "head_tied": restored.reconstruction_head.weight is restored.token_embedding.weight,
            }
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "phase": "M3",
        "device": "cpu",
        "trained": False,
        "qualification_config": config.to_dict(),
        "checks": checks,
        "passed": all(checks.values()),
        "full_width_static_profile": profile_encoder(KiwiLM3Config(num_blocks=num_blocks)),
        "limitations": (
            "Tiny-width CPU checks; no GPU/TPU, throughput, objective or recovery proof."
        ),
    }
