"""Paired CPU mask-policy investigation; no cloud, resume or production objective change."""

from __future__ import annotations

import hashlib
import math
import platform
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch
from tokenizers import Tokenizer

import kiwilm.v3.accelerator as engine
from kiwilm.data import PreparedTokenData
from kiwilm.v3 import KiwiLM3Config
from kiwilm.v3.diagnostics import _audit_payload, context_probe, gradient_probe
from kiwilm.v3.masking import MaskingConfig, corrupt_tokens
from kiwilm.v3.readout_diagnostic import centered_energy
from kiwilm.v3.tokenizer import MaskBPETokenizer
from kiwilm.v3.training_diagnostics import CollapseMonitor

POLICIES = ("variable", "fixed-015")
LEVELS = (0.15, 0.5, 0.9, 1.0)


def tensor_sha256(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@contextmanager
def paired_mask_policy(policy, records):
    """Temporarily intercept native train-step corruption, not evaluation corruption.

    Both policies consume exactly the native variable-policy RNG stream. For fixed
    noise, replay the same Bernoulli uniforms after the two native level draws.
    No clean IDs, eligible slots, padding or loss normalization are changed.
    Single-owner diagnostic only: nested or prepatched contexts are rejected.
    """
    if policy not in POLICIES:
        raise ValueError("unknown diagnostic mask policy")
    if engine.corrupt_tokens is not corrupt_tokens:
        raise RuntimeError("mask-policy diagnostic requires exclusive unpatched trainer")

    def paired(clean, config, *, generator, **kwargs):
        if "noise_level" in kwargs:
            raise ValueError("paired training policy requires native sampled noise")
        before = generator.get_state()
        native = corrupt_tokens(clean, config, generator=generator, **kwargs)
        result = native
        if policy == "fixed-015":
            replay = torch.Generator().set_state(before)
            torch.rand(clean.shape[0], generator=replay)
            torch.rand(clean.shape[0], generator=replay)
            result = corrupt_tokens(clean, config, generator=replay, noise_level=0.15, **kwargs)
            if not torch.equal(replay.get_state(), generator.get_state()):
                raise RuntimeError(
                    "native corruption RNG consumption changed; refuse unpaired test"
                )
        records.append(
            {
                "clean_sha256": tensor_sha256(clean),
                "mask_sha256": tensor_sha256(result.masked_positions),
                "noise_rng_end_sha256": tensor_sha256(generator.get_state()),
                "native_drawn_levels": native.noise_level.tolist(),
                "effective_levels": result.noise_level.tolist(),
                "selected_per_window": result.masked_positions.sum(-1).tolist(),
                "eligible_per_window": result.eligible_positions.sum(-1).tolist(),
            }
        )
        return result

    engine.corrupt_tokens = paired
    try:
        yield
    finally:
        engine.corrupt_tokens = corrupt_tokens


def aggregate_probes(rows):
    """Target-weighted losses/accuracies at each noise, never average unequal windows."""
    aggregate = []
    for level in LEVELS:
        subset = [row for row in rows if row["noise_level"] == level and row["masked_tokens"]]
        count = sum(row["masked_tokens"] for row in subset)
        result = {"noise_level": level, "masked_tokens": count, "windows": len(subset)}
        if count:
            for key in (
                "loss",
                "unigram_loss",
                "masked_accuracy",
                "unigram_accuracy",
                "reversed_context_loss_minus_original",
                "argmax_change_fraction",
                "target_margin_mean",
            ):
                result[key] = sum(row[key] * row["masked_tokens"] for row in subset) / count
            result["loss_minus_unigram"] = result["loss"] - result["unigram_loss"]
            result["maximum_probability_change"] = max(
                row["maximum_probability_change"] for row in subset
            )
            result["visible_content_tokens"] = sum(row["visible_content_tokens"] for row in subset)
            result["changed_visible_tokens"] = sum(row["changed_visible_tokens"] for row in subset)
            histogram = {}
            for row in subset:
                for token, number in row["argmax_counts"].items():
                    histogram[token] = histogram.get(token, 0) + number
            result["argmax_counts"] = histogram
            result["all_finite"] = all(row["finite_logits_and_hidden"] for row in subset)
        aggregate.append(result)
    return aggregate


def evaluate_policy_model(model, windows, masking, log_probabilities, *, seed=242):
    """Aligned held-out masks/context reversal; eval never consumes training RNGs."""
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("noise diagnostic is CPU-only")
    probes = []
    majority = log_probabilities.argmax().item()
    for index, clean in enumerate(windows):
        for level in LEVELS:
            corrupted = corrupt_tokens(
                clean,
                masking,
                generator=torch.Generator().manual_seed(seed + 1 + index),
                noise_level=level,
            )
            selected = corrupted.masked_positions
            if not selected.any():
                probes.append({"window": index, "noise_level": level, "masked_tokens": 0})
                continue
            geometry, margins = {}, {}

            def before(
                module,
                args,
                *,
                geometry=geometry,
                valid=corrupted.attention_mask,
                selected=selected,
            ):
                if not geometry:
                    geometry.update(
                        pre_norm_all_tokens=centered_energy(args[0][valid]),
                        pre_norm_masked_tokens=centered_energy(args[0][selected]),
                        finite_hidden=bool(torch.isfinite(args[0]).all()),
                    )

            def after(module, args, output, *, margins=margins, selected=selected, clean=clean):
                if not margins:
                    values = output[selected][:, : masking.mask_id].float()
                    labels = clean[selected]
                    top, ids = values.topk(2, dim=-1)
                    other = torch.where(ids[:, 0] == labels, top[:, 1], top[:, 0])
                    margin = values.gather(1, labels[:, None]).squeeze(1) - other
                    margins.update(
                        target_margin_mean=margin.mean().item(),
                        target_margin_p10=margin.quantile(0.1).item(),
                        finite_logits=bool(torch.isfinite(output).all()),
                    )

            handles = [
                model.final_norm.register_forward_pre_hook(before),
                model.reconstruction_head.register_forward_hook(after),
            ]
            try:
                row = context_probe(model, clean, masking, noise_level=level, seed=seed + 1 + index)
            finally:
                for handle in handles:
                    handle.remove()
            labels = clean[selected]
            row.update(
                window=index,
                unigram_loss=-log_probabilities[labels].mean().item(),
                unigram_accuracy=(labels == majority).float().mean().item(),
                **geometry,
                **margins,
            )
            row["loss_minus_unigram"] = row["loss"] - row["unigram_loss"]
            row["finite_logits_and_hidden"] = row["finite_logits"] and row["finite_hidden"]
            probes.append(row)
    gradient = gradient_probe(model, windows[0], masking)
    return {
        "rows": probes,
        "by_noise": aggregate_probes(probes),
        "gradient": gradient,
        "mlp_last_first_gradient_ratio": gradient["blocks"][-1]["mlp_gradient_norm"]
        / gradient["blocks"][0]["mlp_gradient_norm"]
        if gradient["blocks"][0]["mlp_gradient_norm"]
        else None,
    }


def run_policy_pair(model_config, training, tokenizer, data, masking, *, steps, progress=None):
    if type(steps) is not int or not 1 <= steps <= 128:
        raise ValueError("paired noise replay must be 1-128 updates")
    if training.device != "cpu" or training.precision != "fp32" or training.grad_accum_steps != 1:
        raise ValueError("paired replay requires CPU FP32 and accumulation 1")
    if steps * training.batch_size * model_config.context_length >= training.max_tokens:
        raise ValueError("replay must stop before the diagnostic schedule horizon")
    cases = []
    with torch.random.fork_rng(devices=[]):
        for policy in POLICIES:
            if progress:
                progress(f"Fresh CPU 512-context replay: {policy}, {steps} updates")
            trainer = engine.AcceleratorTrainer(
                model_config, training, tokenizer, data, masking=masking
            )
            monitor = CollapseMonitor(
                trainer
            )  # training-prefix unigram, independent validation RNG
            generator = torch.Generator().manual_seed(training.validation_seed)
            windows = [
                data.get_batch(
                    "validation",
                    batch_size=1,
                    context_length=model_config.context_length,
                    generator=generator,
                )[0]
                for _ in range(4)
            ]
            evaluation = partial(
                evaluate_policy_model,
                trainer.model,
                windows,
                masking,
                monitor.log_probabilities,
                seed=training.validation_seed,
            )
            trajectory = [{"step": 0, **evaluation()}]
            rows = []
            for step in range(1, steps + 1):
                records = []
                with paired_mask_policy(policy, records):
                    row = trainer.train_step()
                if len(records) != 1:
                    raise RuntimeError("diagnostic expected exactly one accumulated microbatch")
                row.update(
                    pairing=records[0], torch_rng_end_sha256=tensor_sha256(torch.get_rng_state())
                )
                if row["loss"] is not None and not math.isfinite(row["loss"]):
                    raise FloatingPointError("nonfinite mask-policy replay")
                rows.append(row)
                if step % 16 == 0 or step == steps:
                    trajectory.append({"step": step, **evaluation()})
                    if progress:
                        progress(
                            f"{policy}: update {step}, loss {row['loss']}, "
                            f"masks {trainer.masked_tokens_seen}"
                        )
            cases.append(
                {
                    "policy": policy,
                    "steps": trainer.step,
                    "input_tokens": trainer.tokens_seen,
                    "selected_targets": trainer.masked_tokens_seen,
                    "eligible_tokens": sum(row["eligible_tokens"] for row in rows),
                    "empty_mask_updates": sum(row["skipped_empty_mask"] for row in rows),
                    "validation_window_sha256": [tensor_sha256(window) for window in windows],
                    "unigram_prefix_tokens": monitor.prefix_tokens,
                    "unigram_log_probabilities_sha256": tensor_sha256(monitor.log_probabilities),
                    "training_rows": rows,
                    "trajectory": trajectory,
                }
            )
            del trainer, monitor, evaluation
    left, right = cases
    paired = all(
        a["pairing"][key] == b["pairing"][key]
        for a, b in zip(left["training_rows"], right["training_rows"], strict=True)
        for key in (
            "clean_sha256",
            "noise_rng_end_sha256",
            "native_drawn_levels",
            "eligible_per_window",
        )
    )
    if (
        not paired
        or left["trajectory"][0] != right["trajectory"][0]
        or left["validation_window_sha256"] != right["validation_window_sha256"]
        or left["unigram_log_probabilities_sha256"] != right["unigram_log_probabilities_sha256"]
        or any(
            a["learning_rate"] != b["learning_rate"]
            for a, b in zip(left["training_rows"], right["training_rows"], strict=True)
        )
    ):
        raise RuntimeError(
            "data, initial metrics, mask RNG, validation, unigram or LR pairing failed"
        )
    return cases


def noise_diagnostic(run_dir: Path, data_dir: Path, *, steps=64, seed=42, progress=None):
    if type(steps) is not int or not 1 <= steps <= 128:
        raise ValueError("paired noise replay must be 1-128 updates")
    if type(seed) is not int or not 0 <= seed < 2**32 - 1:
        raise ValueError("invalid model seed")
    payload, document, manifest = _audit_payload(run_dir)
    contract = document["job"]["contract"]
    del payload  # No saved model, optimizer or RNG is loaded into a trainer.
    config = KiwiLM3Config.from_dict(contract["model"])
    if config.context_length != 512 or config.d_model > 512 or config.num_blocks > 16:
        raise ValueError("diagnostic requires context 512 and bounded width/depth")
    original = engine.AcceleratorTrainConfig(**contract["training"])
    if (
        original.max_tokens != 5_000_000
        or original.batch_size != 8
        or original.grad_accum_steps != 4
    ):
        raise ValueError("paired replay requires original 5M 8x4 diagnostic controls")
    tokenizer = MaskBPETokenizer(Tokenizer.from_str(document["tokenizer_json"]))
    if tokenizer.fingerprint != contract["tokenizer_sha256"]:
        raise ValueError("tokenizer fingerprint mismatch")
    data = PreparedTokenData(data_dir, expected_fingerprint=contract["data_fingerprint"])
    tokenizer.assert_base_compatible(data.tokenizer)
    masking = MaskingConfig(**contract["masking"])
    training = replace(
        original,
        device="cpu",
        precision="fp32",
        batch_size=2,
        grad_accum_steps=1,
        max_tokens=original.max_tokens // 16,
        warmup_tokens=original.warmup_tokens // 16,
        seed=seed,
    )
    cases = run_policy_pair(
        config, training, tokenizer, data, masking, steps=steps, progress=progress
    )
    return {
        "schema_version": 1,
        "scope": "Fresh paired CPU test; not saved-checkpoint continuation or promotion",
        "source": {
            "checkpoint_sha256": manifest["files"]["latest.pt"]["sha256"],
            "job_identity": document["identity"],
            "original_contract": contract,
        },
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "threads": torch.get_num_threads(),
        },
        "model": config.to_dict(),
        "training": training.to_dict(),
        "masking": masking.to_dict(),
        "pairing": (
            "Identical data, initialization, LR, validation and selection uniforms; "
            "different noise thresholds"
        ),
        "validation_scope": "Four fixed held-out batch-1 windows at each of 15/50/90/100% noise",
        "cases": cases,
        "limitations": [
            "One model seed and 64 default updates; no long-run or CUDA BF16 conclusion",
            "2 vs 32 windows/update; 16x shorter horizon/warmup matches update-level LR only",
            "Matched inputs/updates, NOT selected-target counts or corruption difficulty",
            "Unigram: first 1M train tokens, add-one smoothing, no validation fitting",
            "Context reversal is an OOD sensitivity test, not semantic or causal proof",
            "Fully masked probes have no visible context; zero response is a negative control",
        ],
        "cloud_allocated": False,
        "checkpoint_modified": False,
        "canonical_fix": None,
    }
