"""Bounded CPU collapse diagnostics; never provision hardware or modify a checkpoint.

The synthetic overfit probe exercises the native optimizer/accumulation path,
but is deliberately not a corpus experiment or evidence of accelerator parity.
"""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from kiwilm.colab_artifacts import file_sha256
from kiwilm.data import PreparedTokenData
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.accelerator import (
    AcceleratorTrainConfig,
    AcceleratorTrainer,
    dense_reconstruction,
)
from kiwilm.v3.accelerator_checkpoint import FORMAT
from kiwilm.v3.accelerator_workflow import prepare_demo
from kiwilm.v3.experiments import canonical_digest
from kiwilm.v3.masking import MaskingConfig, corrupt_tokens
from kiwilm.v3.tokenizer import MaskBPETokenizer


class _FixedWindows:
    """Synthetic balanced cue/target pairs; every row has the same target position.

    IDs 6/7 are protected cues for targets 4/5, with either a left or right cue.
    At evaluation all targets are MASK, so position or visible-label copying
    cannot solve the task. IDs denote synthetic symbols, not natural words.
    """

    def __init__(self, base):
        self.tokenizer = base.tokenizer
        self.windows = torch.full((4, 16), 8, dtype=torch.long)
        self.windows[:, 0], self.windows[:, -1] = 2, 3
        self.windows[:, 8] = torch.tensor([4, 5, 4, 5])
        self.windows[:2, 1] = torch.tensor([6, 7])
        self.windows[2:, 14] = torch.tensor([6, 7])
        self.fingerprint = canonical_digest(
            {"synthetic": "context-cue-overfit-v1", "windows": self.windows.tolist()}
        )

    def tokens(self, split):
        return self.windows.flatten()

    def get_batch(self, split, *, batch_size, context_length, generator):
        if batch_size != 4 or context_length != 16:
            raise ValueError("fixed overfit windows require batch 4 and context 16")
        return self.windows.clone(), self.windows.clone()


@torch.no_grad()
def _cue_score(model, tokenizer, clean, masking, *, intervention="original"):
    corrupted = corrupt_tokens(
        clean, masking, generator=torch.Generator().manual_seed(243), noise_level=1.0
    )
    ids = corrupted.input_ids.clone()
    targets = clean.clone()
    if intervention == "swap":
        ids = torch.where(ids == 6, 7, torch.where(ids == 7, 6, ids))
        targets[:, 8] = torch.where(targets[:, 8] == 4, 5, 4)
    elif intervention == "erase":
        ids[(ids == 6) | (ids == 7)] = 8
    elif intervention != "original":
        raise ValueError("unknown cue intervention")
    logits = model(ids, attention_mask=corrupted.attention_mask, noise_level=corrupted.noise_level)
    total, hits = dense_reconstruction(
        logits, targets, corrupted.masked_positions, tokenizer.mask_id, accuracy=True
    )
    return {
        "loss": total.item() / 4,
        "accuracy": hits.item() / 4,
        "predictions": logits[:, 8, : tokenizer.mask_id].argmax(-1).tolist(),
        "targets": targets[:, 8].tolist(),
    }


def overfit_probe(
    *,
    steps: int = 500,
    learning_rate: float = 0.001,
    seed: int = 42,
    width: int = 32,
    tokenizer_data_dir: Path | None = None,
) -> dict:
    """Train only four synthetic windows on CPU, with the real native train_step.

    Uses variable-noise corruption, dropout 0.1, AdamW and depth-scaled residual
    initialization. Default small width/vocabulary isolate learnability; an
    explicit full-width/real-tokenizer probe keeps context short and synthetic.
    Restore the caller's global torch RNG on success or failure.
    """
    if type(steps) is not int or not 1 <= steps <= 2000:
        raise ValueError("overfit steps must be 1-2000")
    if type(width) is not int or width not in {32, 512}:
        raise ValueError("overfit width must be 32 or 512")
    if width == 512 and steps > 300:
        raise ValueError("full-width local overfit is bounded to 300 steps")
    if (
        isinstance(learning_rate, bool)
        or not math.isfinite(learning_rate)
        or not (0 < learning_rate <= 0.01)
    ):
        raise ValueError("overfit learning rate must be in (0, 0.01]")
    with torch.random.fork_rng(devices=[]), TemporaryDirectory(prefix="kiwilm3-overfit-") as tmp:
        base_path, tokenizer_path = Path(tmp) / "data", Path(tmp) / "tokenizer.json"
        if tokenizer_data_dir is None:
            prepare_demo(base_path, tokenizer_path)
            base = PreparedTokenData(base_path)
            tokenizer = MaskBPETokenizer.load(tokenizer_path)
        else:
            base = PreparedTokenData(tokenizer_data_dir)
            tokenizer = MaskBPETokenizer.from_base(base.tokenizer)
        data = _FixedWindows(base)
        model_config = KiwiLM3Config(
            vocab_size=tokenizer.vocab_size,
            context_length=16,
            d_model=width,
            num_blocks=12,
            num_heads=4 if width == 32 else 8,
            swiglu_dim=4 * width,
            noise_embedding_dim=16 if width == 32 else 64,
            dropout=0.1,
        )
        config = AcceleratorTrainConfig(
            device="cpu",
            precision="fp32",
            max_tokens=steps * 64,
            warmup_tokens=min(640, steps * 64 // 10),
            batch_size=4,
            grad_accum_steps=1,
            learning_rate=learning_rate,
            min_learning_rate=learning_rate / 10,
            seed=seed,
        )
        masking = MaskingConfig(
            vocab_size=tokenizer.vocab_size,
            mask_id=tokenizer.mask_id,
            protected_token_ids=(0, 1, 2, 3, 6, 7, 8),
        )
        trainer = AcceleratorTrainer(model_config, config, tokenizer, data, masking=masking)
        trainer.model.eval()
        initial = _cue_score(trainer.model, tokenizer, data.windows, masking)
        trajectory = [{"step": 0, **initial}]
        skipped = 0
        for step in range(1, steps + 1):
            row = trainer.train_step()
            skipped += row["skipped_empty_mask"]
            if step % 50 == 0 or step == steps:
                trainer.model.eval()
                trajectory.append(
                    {"step": step, **_cue_score(trainer.model, tokenizer, data.windows, masking)}
                )
        swapped = _cue_score(trainer.model, tokenizer, data.windows, masking, intervention="swap")
        erased = _cue_score(trainer.model, tokenizer, data.windows, masking, intervention="erase")
        final = trajectory[-1]
        return {
            "scope": (
                f"Synthetic CPU FP32, width {width}, 12 hybrid blocks, "
                f"vocabulary {tokenizer.vocab_size}, context 16; not corpus or CUDA BF16 proof"
            ),
            "tokenizer_sha256": tokenizer.fingerprint,
            "config": model_config.to_dict(),
            "training": config.to_dict(),
            "steps": trainer.step,
            "optimizer_steps": trainer.optimizer_steps,
            "empty_mask_skips": skipped,
            "tokens_seen": trainer.tokens_seen,
            "trajectory": trajectory,
            "swapped_cues": swapped,
            "erased_cues": erased,
            "passed": final["accuracy"] == swapped["accuracy"] == 1.0
            and final["loss"] < 0.1
            and erased["accuracy"] == 0.5,
        }


def _rms(values, valid):
    return values.detach()[valid].float().square().mean().sqrt().item()


def _token_cosine(values, valid):
    vectors = F.normalize(values.detach()[valid].float(), dim=-1)
    count = len(vectors)
    if count < 2:
        return None
    # Mean pairwise cosine without allocating an N x N matrix.
    return ((vectors.sum(0).square().sum() - count) / (count * (count - 1))).clamp(-1, 1).item()


def gradient_probe(model, clean, masking, *, noise_level=0.15) -> dict:
    """Read-only eval-mode gradients and residual statistics; .grad buffers untouched."""
    if model.token_embedding.weight.device.type != "cpu":
        raise ValueError("collapse gradient probe is explicitly CPU-only")
    corrupted = corrupt_tokens(
        clean, masking, generator=torch.Generator().manual_seed(243), noise_level=noise_level
    )
    valid = corrupted.attention_mask
    count = corrupted.masked_positions.sum().item()
    if not count:
        raise ValueError("gradient probe requires selected targets")
    records = [{"block": i, "mixer": kind} for i, kind in enumerate(model.config.mixer_schedule)]
    handles = []

    def capture(index, key):
        def hook(module, args, output):
            records[index][key] = _rms(output, valid)
            if key == "output_rms":
                records[index]["output_mean_token_cosine"] = _token_cosine(output, valid)

        return hook

    def before(index, key):
        def hook(module, args):
            records[index][key] = _rms(args[0], valid)
            if key == "input_rms":
                records[index]["input_mean_token_cosine"] = _token_cosine(args[0], valid)

        return hook

    previous = model.training
    model.eval()
    try:
        for i, block in enumerate(model.blocks):
            handles.extend(
                [
                    block.register_forward_pre_hook(before(i, "input_rms")),
                    block.mixer.register_forward_hook(capture(i, "mixer_update_rms")),
                    block.mlp_norm.register_forward_pre_hook(before(i, "post_mixer_rms")),
                    block.mlp.register_forward_hook(capture(i, "mlp_update_rms")),
                    block.register_forward_hook(capture(i, "output_rms")),
                ]
            )
        with torch.enable_grad():
            logits = model(
                corrupted.input_ids, attention_mask=valid, noise_level=corrupted.noise_level
            )
            total, _ = dense_reconstruction(
                logits, clean, corrupted.masked_positions, masking.mask_id
            )
            loss = total / count
            parameters = list(model.named_parameters())
            gradients = torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True)
        norms = {
            name: gradient.detach().float().norm().item() if gradient is not None else 0.0
            for (name, _), gradient in zip(parameters, gradients, strict=True)
        }
        finite = math.isfinite(loss.item()) and all(
            gradient is None or torch.isfinite(gradient).all().item() for gradient in gradients
        )
        for i, row in enumerate(records):
            for family in ("mixer", "mlp"):
                row[f"{family}_gradient_norm"] = math.sqrt(
                    sum(
                        norm**2
                        for name, norm in norms.items()
                        if name.startswith(f"blocks.{i}.{family}.")
                    )
                )
            denominator = row["post_mixer_rms"]
            row["mlp_contribution"] = row["mlp_update_rms"] / denominator if denominator else None
            row["residual_amplification"] = row["output_rms"] / denominator if denominator else None
            input_rms = row["input_rms"]
            row["mixer_contribution"] = row["mixer_update_rms"] / input_rms if input_rms else None
            row["mixer_residual_amplification"] = denominator / input_rms if input_rms else None
            row["whole_block_amplification"] = row["output_rms"] / input_rms if input_rms else None
        with torch.no_grad():
            condition_rms = (
                model.noise_conditioning(corrupted.noise_level).square().mean().sqrt().item()
            )
            embedding_rms = _rms(model.token_embedding(corrupted.input_ids), valid)
        return {
            "noise_level": noise_level,
            "loss": loss.item(),
            "all_finite": bool(finite),
            "all_parameter_gradients_nonzero": all(norm > 0 for norm in norms.values()),
            "parameter_gradient_norms": norms,
            "conditioning_rms": condition_rms,
            "token_embedding_rms": embedding_rms,
            "conditioning_embedding_rms_ratio": condition_rms / embedding_rms
            if embedding_rms
            else None,
            "blocks": records,
        }
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous)


@torch.no_grad()
def context_probe(model, clean, masking, *, noise_level=0.15, seed=243, runtime=None) -> dict:
    """Compare identical masked targets with original versus reversed visible content.

    Controls, MASK positions, noise and targets stay fixed. Report differences,
    not a promotion threshold or proof that a particular component caused collapse.
    """
    previous = model.training
    model.eval()
    try:
        batch = corrupt_tokens(
            clean, masking, generator=torch.Generator().manual_seed(seed), noise_level=noise_level
        )
        shuffled = batch.input_ids.clone()
        visible = batch.attention_mask & ~batch.masked_positions
        for token_id in masking.protected_token_ids:
            visible &= shuffled != token_id
        for i in range(len(clean)):
            shuffled[i, visible[i]] = shuffled[i, visible[i]].flip(0)
        count = batch.masked_positions.sum().item()
        if not count:
            raise ValueError("context probe requires selected targets")
        device = runtime.device if runtime is not None else torch.device("cpu")
        valid = batch.attention_mask.to(device)
        selected_positions = batch.masked_positions.to(device)
        labels = clean.to(device)
        outputs = []
        for ids in (batch.input_ids, shuffled):
            with runtime.autocast() if runtime is not None else nullcontext():
                logits = model(ids.to(device), attention_mask=valid, noise_level=batch.noise_level)
                total, hits = dense_reconstruction(
                    logits, labels, selected_positions, masking.mask_id, accuracy=True
                )
            selected = logits[..., : masking.mask_id][selected_positions].float()
            outputs.append((total.item() / count, hits.item() / count, selected.softmax(-1)))
        left, right = outputs
        predictions = left[2].argmax(-1)
        ids, counts = predictions.unique(return_counts=True)
        return {
            "noise_level": noise_level,
            "masked_tokens": count,
            "visible_content_tokens": visible.sum().item(),
            "changed_visible_tokens": ((shuffled != batch.input_ids) & visible).sum().item(),
            "loss": left[0],
            "masked_accuracy": left[1],
            "reversed_context_loss": right[0],
            "reversed_context_loss_minus_original": right[0] - left[0],
            "maximum_probability_change": (left[2] - right[2]).abs().max().item(),
            "argmax_change_fraction": (predictions != right[2].argmax(-1)).float().mean().item(),
            "argmax_counts": {
                str(i): n for i, n in zip(ids.tolist(), counts.tolist(), strict=True)
            },
        }
    finally:
        model.train(previous)


def collapse_flags(gradient: dict, probes: list[dict]) -> dict:
    """Descriptive investigation flags, not calibrated promotion criteria."""
    partial = [row for row in probes if row["changed_visible_tokens"] > 0]
    common_predictions = {token for row in partial for token in row["argmax_counts"]}
    return {
        "threshold_scope": "Investigation heuristics, not model-selection acceptance thresholds",
        "mixer_update_over_10x_input_blocks": [
            row["block"]
            for row in gradient["blocks"]
            if row["mixer_contribution"] is not None and row["mixer_contribution"] > 10
        ],
        "mean_output_token_cosine_over_0_9999_blocks": [
            row["block"]
            for row in gradient["blocks"]
            if row["output_mean_token_cosine"] is not None
            and row["output_mean_token_cosine"] > 0.9999
        ],
        "same_single_argmax_across_partial_context_probes": bool(partial)
        and len(common_predictions) == 1,
        "no_argmax_response_to_changed_context": bool(partial)
        and all(row["argmax_change_fraction"] == 0 for row in partial),
    }


def _audit_payload(run_dir: Path):
    """Accept real native generations or verified collected bundles, never fabricate a manifest."""
    native = (run_dir / "manifest.json").is_file()
    manifest = json.loads(
        (run_dir / ("manifest.json" if native else "artifact-manifest.json")).read_text()
    )
    if not native and manifest.get("schema_version") != 1:
        raise ValueError("unsupported collected artifact manifest")
    for name in ("latest.pt", "job.json", "metrics.jsonl"):
        path = run_dir / name
        if manifest["files"][name] != {"bytes": path.stat().st_size, "sha256": file_sha256(path)}:
            raise ValueError(f"native generation checksum mismatch: {name}")
    if not native:
        for name in ("summary.json", "spec.json"):
            path = run_dir / name
            if manifest["files"][name] != {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }:
                raise ValueError(f"collected provenance checksum mismatch: {name}")
        summary = json.loads((run_dir / "summary.json").read_text())
        spec = json.loads((run_dir / "spec.json").read_text())
        marker = json.loads((run_dir / "download-complete.json").read_text())
        if (
            marker != {"spec_sha256": canonical_digest(spec)}
            or summary["last_commit"]["checkpoint_sha256"]
            != manifest["files"]["latest.pt"]["sha256"]
        ):
            raise ValueError("collected checkpoint completion/provenance mismatch")
    payload = torch.load(run_dir / "latest.pt", weights_only=True, map_location="cpu")
    document = json.loads((run_dir / "job.json").read_text())
    job = document["job"]
    if not native:
        if (
            summary["job"] != job
            or summary["identity"] != canonical_digest(job)
            or spec["request"]["config"] != job["contract"]["training"]
        ):
            raise ValueError("collected job provenance mismatch")
        manifest = {
            "files": manifest["files"],
            "contract_digest": summary["identity"],
            "step": summary["step"],
            "tokens_seen": summary["tokens_seen"],
        }
    if (
        payload.get("format") != FORMAT
        or payload.get("completed_boundary") is not True
        or payload.get("job") != job
        or payload.get("contract") != job["contract"]
        or payload.get("identity") != canonical_digest(job["contract"])
        or payload.get("job_identity") != canonical_digest(job)
        or document["identity"] != canonical_digest(job)
        or manifest["contract_digest"] != canonical_digest(job)
        or manifest["step"] != payload.get("step")
        or manifest["tokens_seen"] != payload.get("tokens_seen")
        or payload.get("metrics") != manifest["files"]["metrics.jsonl"]
    ):
        raise ValueError("native checkpoint/job/progress provenance mismatch")
    return payload, document, manifest


def audit_checkpoint(run_dir: Path, data_dir: Path, *, batches: int = 2) -> dict:
    """Verify a native generation/collected bundle without resuming any training.

    CPU FP32 diagnostics do not reinterpret the saved CUDA runtime contract and
    cannot establish CUDA/BF16 parity. Checkpoint bytes remain untouched.
    """
    if type(batches) is not int or not 1 <= batches <= 4:
        raise ValueError("checkpoint audit batches must be 1-4")
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    payload, document, manifest = _audit_payload(run_dir)
    job = document["job"]
    contract = job["contract"]
    tokenizer = MaskBPETokenizer(Tokenizer.from_str(document["tokenizer_json"]))
    if tokenizer.fingerprint != contract["tokenizer_sha256"]:
        raise ValueError("native tokenizer fingerprint mismatch")
    data = PreparedTokenData(data_dir, expected_fingerprint=contract["data_fingerprint"])
    tokenizer.assert_base_compatible(data.tokenizer)
    state = payload["model"]
    if not all(torch.isfinite(t).all().item() and t.dtype == torch.float32 for t in state.values()):
        raise ValueError("native model tensors must be finite FP32")
    if not torch.equal(state["token_embedding.weight"], state["reconstruction_head.weight"]):
        raise ValueError("native tied weights differ")
    # Optimizer/RNG state is intentionally never loaded into a live trainer.
    del payload["optimizer"]
    with torch.random.fork_rng(devices=[]):
        model = build_encoder(KiwiLM3Config.from_dict(contract["model"]))
        model.load_state_dict(state, strict=True)
        generator = torch.Generator().manual_seed(contract["training"]["validation_seed"])
        windows = [
            data.get_batch(
                "validation",
                batch_size=1,
                context_length=model.config.context_length,
                generator=generator,
            )[0]
            for _ in range(batches)
        ]
        masking = MaskingConfig(**contract["masking"])
        probes = [
            {"batch": i, **context_probe(model, clean, masking, noise_level=level)}
            for i, clean in enumerate(windows)
            for level in (0.15, 0.5, 0.9, 1.0)
        ]
        gradient = gradient_probe(model, windows[0], masking)
        # Counterfactual only: remove conditioning for one forward comparison,
        # without changing any parameters, saved config or original checkpoint.
        handle = model.noise_conditioning.register_forward_hook(
            lambda module, args, output: torch.zeros_like(output)
        )
        try:
            no_condition = context_probe(model, windows[0], masking)
        finally:
            handle.remove()
        torch.manual_seed(contract["training"]["seed"])
        initial = build_encoder(model.config)
        initial_gradient = gradient_probe(initial, windows[0], masking)
    return {
        "scope": (
            "Read-only CPU FP32 diagnostic, batch 1; "
            "not full validation, accelerator parity or a training restart"
        ),
        "checkpoint_sha256": manifest["files"]["latest.pt"]["sha256"],
        "job_identity": canonical_digest(job),
        "original_runtime": contract["runtime"],
        "diagnostic_runtime": {
            "device": "cpu",
            "precision": "fp32",
            "torch": str(torch.__version__),
        },
        "step": payload["step"],
        "tokens_seen": payload["tokens_seen"],
        "context_probes": probes,
        "gradient_probe": gradient,
        "investigation_flags": collapse_flags(gradient, probes),
        "zero_conditioning_counterfactual": no_condition,
        "fresh_initialization_gradient_probe": initial_gradient,
        "initialization_reference_scope": (
            "Same config and saved model seed, initialized on the current CPU runtime; "
            "not a recovered original step-zero state or CUDA parity test"
        ),
    }
