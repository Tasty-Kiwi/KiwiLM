"""Bounded CPU-only factorial probes; no checkpoint resume or production model changes."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch
from tokenizers import Tokenizer

from kiwilm.data import PreparedTokenData
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.accelerator import AcceleratorTrainConfig, AcceleratorTrainer
from kiwilm.v3.diagnostics import _audit_payload, context_probe, gradient_probe
from kiwilm.v3.masking import MaskingConfig
from kiwilm.v3.tokenizer import MaskBPETokenizer

CASES = (
    ("control", 1.0, 1.0),
    ("conditioning-off", 0.0, 1.0),
    ("mixer-quarter", 1.0, 0.25),
    ("conditioning-off-mixer-quarter", 0.0, 0.25),
)


@contextmanager
def interventions(model, *, conditioning_scale=1.0, mixer_scale=1.0):
    """Temporary differentiable output scaling; parameters/state dict stay untouched."""
    if conditioning_scale not in {0.0, 1.0} or mixer_scale not in {0.25, 1.0}:
        raise ValueError("select conditioning 0/1 and mixer scale 0.25/1")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("isolation probes require CPU")
    handles = []
    previous = model.training
    try:
        if conditioning_scale != 1:
            handles.append(
                model.noise_conditioning.register_forward_hook(
                    lambda module, args, output: output * conditioning_scale
                )
            )
        if mixer_scale != 1:
            for block in model.blocks:
                handles.append(
                    block.mixer.register_forward_hook(
                        lambda module, args, output: output * mixer_scale
                    )
                )
        yield
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous)


def _measure(model, clean, masking):
    gradient = gradient_probe(model, clean, masking)
    return {
        "gradient": gradient,
        "context": context_probe(model, clean, masking),
    }


def isolate_checkpoint(run_dir: Path, data_dir: Path, *, replay_steps=32, progress=None):
    if type(replay_steps) is not int or not 0 <= replay_steps <= 64:
        raise ValueError("local replay must be 0-64 steps per case")
    payload, document, manifest = _audit_payload(run_dir)
    contract = document["job"]["contract"]
    tokenizer = MaskBPETokenizer(Tokenizer.from_str(document["tokenizer_json"]))
    if tokenizer.fingerprint != contract["tokenizer_sha256"]:
        raise ValueError("tokenizer fingerprint mismatch")
    data = PreparedTokenData(data_dir, expected_fingerprint=contract["data_fingerprint"])
    tokenizer.assert_base_compatible(data.tokenizer)
    config = KiwiLM3Config.from_dict(contract["model"])
    masking = MaskingConfig(**contract["masking"])
    state = payload["model"]
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all().item() for t in state.values()):
        raise ValueError("checkpoint model must be finite FP32")
    if not torch.equal(state["token_embedding.weight"], state["reconstruction_head.weight"]):
        raise ValueError("checkpoint tied weights disagree")
    # Never restore the optimizer/RNG from the downloaded run.
    del payload["optimizer"]
    clean = data.get_batch(
        "validation",
        batch_size=1,
        context_length=config.context_length,
        generator=torch.Generator().manual_seed(contract["training"]["validation_seed"]),
    )[0]
    results = {
        "schema_version": 1,
        "scope": "CPU FP32, one fixed validation window; not accelerator parity or promotion",
        "checkpoint_sha256": manifest["files"]["latest.pt"]["sha256"],
        "job_identity": document["identity"],
        "checkpoint_steps": payload["step"],
        "checkpoint_tokens": payload["tokens_seen"],
        "original_runtime": contract["runtime"],
        "counterfactual_scope": "Inference interventions cannot establish the learning-time cause",
        "counterfactuals": [],
        "fresh_replays": [],
        "cloud_allocated": False,
        "checkpoint_modified": False,
    }
    with torch.random.fork_rng(devices=[]):
        model = build_encoder(config)
        model.load_state_dict(state, strict=True)
        for name, condition, mixer in CASES:
            if progress:
                progress(f"Frozen checkpoint: {name}")
            with interventions(model, conditioning_scale=condition, mixer_scale=mixer):
                results["counterfactuals"].append({"case": name, **_measure(model, clean, masking)})
        del model, state, payload
        if not replay_steps:
            return results
        original = AcceleratorTrainConfig(**contract["training"])
        if (
            original.max_tokens != 5_000_000
            or original.batch_size != 8
            or original.grad_accum_steps != 4
        ):
            raise ValueError(
                "fresh replay currently requires the original 5M 8x4 diagnostic controls"
            )
        # 2 rather than 32 windows/update => divide token horizon and warmup by 16.
        # This matches per-update LR, NOT original effective batch or token budget.
        replay = replace(
            original,
            device="cpu",
            precision="fp32",
            batch_size=2,
            grad_accum_steps=1,
            max_tokens=original.max_tokens // 16,
            warmup_tokens=original.warmup_tokens // 16,
        )
        results["replay_controls"] = replay.to_dict()
        results["replay_scope"] = (
            "Fresh random initialization; 2 vs 32 windows/update; matched per-update LR via "
            "16x shorter token schedule. Same native objective/data/model/optimizer/seed. "
            "Not the frozen Colab experiment, full-budget run, or saved-checkpoint continuation."
        )
        for name, condition, mixer in CASES:
            if progress:
                progress(f"Fresh CPU replay: {name}, {replay_steps} updates")
            trainer = AcceleratorTrainer(config, replay, tokenizer, data, masking=masking)
            trajectory = []
            training = []
            with interventions(trainer.model, conditioning_scale=condition, mixer_scale=mixer):
                trajectory.append({"step": 0, **_measure(trainer.model, clean, masking)})
                for step in range(1, replay_steps + 1):
                    row = trainer.train_step()
                    if row["loss"] is not None and not math.isfinite(row["loss"]):
                        raise FloatingPointError("nonfinite local replay")
                    training.append(row)
                    if step % 8 == 0 or step == replay_steps:
                        trajectory.append({"step": step, **_measure(trainer.model, clean, masking)})
                        if progress:
                            progress(f"{name}: update {step}, loss {row['loss']:.4f}")
            results["fresh_replays"].append(
                {
                    "case": name,
                    "conditioning_scale": condition,
                    "mixer_scale": mixer,
                    "steps": trainer.step,
                    "tokens_seen": trainer.tokens_seen,
                    "trajectory": trajectory,
                    "training": training,
                }
            )
            del trainer
    return results
