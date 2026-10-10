"""Bounded fixed-corpus overfit/readout tests, not production training or a model fix."""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import functional as F

from kiwilm.data import PreparedTokenData
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.accelerator import AcceleratorTrainConfig, dense_reconstruction, learning_rate
from kiwilm.v3.diagnostics import _audit_payload
from kiwilm.v3.experiments import canonical_digest
from kiwilm.v3.tokenizer import MaskBPETokenizer

CASES = (
    ("tied-full", False, False),
    ("untied-full", True, False),
    ("tied-two-target", False, True),
    ("untied-two-target", True, True),
)


@dataclass(frozen=True)
class FixedCorpusTask:
    clean: torch.Tensor
    target_ids: tuple[int, int]
    starts: tuple[int, ...]
    position: int

    def __post_init__(self):
        if (
            self.clean.dtype != torch.long
            or self.clean.ndim != 2
            or self.clean.shape[0] != 4
            or self.clean.device.type != "cpu"
            or not 0 <= self.position < self.clean.shape[1]
            or len(set(self.target_ids)) != 2
            or len(self.starts) != 4
        ):
            raise ValueError("task requires four CPU windows, one shared target slot and two IDs")
        expected = torch.tensor([*self.target_ids, *self.target_ids])
        if not torch.equal(self.clean[:, self.position], expected):
            raise ValueError("task targets must be balanced in alternating class order")
        visible = self.clean.clone()
        visible[:, self.position] = -1
        if torch.isin(visible, torch.tensor(self.target_ids)).any():
            raise ValueError("target IDs cannot occur in visible context")
        if torch.unique(visible, dim=0).shape[0] != 4:
            raise ValueError("task contexts must be distinct")

    def inputs(self, mask_id, *, mode="original"):
        ids = self.clean.clone()
        ids[:, self.position] = mask_id
        targets = self.clean.clone()
        if mode == "swap":
            # Whole-context exchange, NOT a semantic edit or unseen-context generalization.
            ids = ids[[1, 0, 3, 2]]
            targets[:, self.position] = targets[[1, 0, 3, 2], self.position]
        elif mode == "erase":
            ids.fill_(mask_id)  # identical inputs/position/noise; out-of-distribution control
        elif mode != "original":
            raise ValueError("unknown task intervention")
        selected = torch.zeros_like(ids, dtype=torch.bool)
        selected[:, self.position] = True
        return ids, targets, selected

    def to_dict(self):
        values = {
            "clean_windows": self.clean.tolist(),
            "target_ids": list(self.target_ids),
            "starts": list(self.starts),
            "target_position": self.position,
            "context_length": self.clean.shape[1],
            "noise_level": 1 / self.clean.shape[1],
            "selected_targets_per_update": 4,
        }
        return {**values, "sha256": canonical_digest(values)}


def corpus_task(data, tokenizer, *, context_length=64, prefix_tokens=1_000_000):
    """First two nonoverlapping, label-copy-free windows per class in a fixed prefix."""
    if type(context_length) is not int or not 16 <= context_length <= 64:
        raise ValueError("fixed corpus context must be 16-64")
    if type(prefix_tokens) is not int or not 64 <= prefix_tokens <= 1_000_000:
        raise ValueError("search prefix must be 64-1000000 tokens")
    prefix = torch.as_tensor(data.tokens("train")[:prefix_tokens].copy(), dtype=torch.long)
    counts = torch.bincount(prefix, minlength=tokenizer.vocab_size)
    comma = [i for i in range(tokenizer.mask_id) if tokenizer.decode([i]) == ","]
    words = [i for i in counts.argsort(descending=True).tolist() if tokenizer.decode([i]) == " the"]
    if len(comma) != 1 or not words:
        raise ValueError("task requires single-token ',' and ' the' in the corpus tokenizer")
    targets = (comma[0], words[0])
    position = context_length // 2
    found = {i: [] for i in targets}
    used = []
    controls = torch.tensor(
        [tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id]
    )
    locations = torch.nonzero(torch.isin(prefix, torch.tensor(targets))).flatten().tolist()
    for center in locations:
        token = prefix[center].item()
        start, end = center - position, center - position + context_length
        if len(found[token]) == 2 or start < 0 or end > len(prefix):
            continue
        if any(start < other_end and other_start < end for other_start, other_end in used):
            continue
        window = prefix[start:end]
        if torch.isin(window, controls).any():
            continue
        visible = window.clone()
        visible[position] = -1
        if torch.isin(visible, torch.tensor(targets)).any():
            continue
        found[token].append((start, window.clone()))
        used.append((start, end))
        if all(len(value) == 2 for value in found.values()):
            break
    if any(len(value) != 2 for value in found.values()):
        raise ValueError("not enough nonoverlapping label-copy-free windows in fixed prefix")
    pairs = [found[token][i] for i in range(2) for token in targets]
    return FixedCorpusTask(
        torch.stack([x[1] for x in pairs]), targets, tuple(x[0] for x in pairs), position
    )


def task_loss(logits, clean, selected, mask_id, target_ids, *, two_target=False):
    if not two_target:
        total, _ = dense_reconstruction(logits, clean, selected, mask_id)
        return total / selected.sum()
    targets = clean[selected]
    classes = torch.tensor(target_ids, device=logits.device)
    if not torch.isin(targets, classes).all():
        raise ValueError("two-target loss received an unrelated target")
    labels = (targets == classes[1]).long()
    return F.cross_entropy(logits[selected][:, classes].float(), labels)


def centered_energy(values):
    """Within-set variation / total mean-square energy; no unstable cosine threshold."""
    values = values.detach().double().reshape(-1, values.shape[-1])
    total = values.square().mean()
    centered = (values - values.mean(dim=0)).square().mean()
    return {
        "rms": total.sqrt().item(),
        "centered_rms": centered.sqrt().item(),
        "centered_energy_fraction": (centered / total).item() if total > 0 else 0.0,
    }


@torch.no_grad()
def score_task(model, task, mask_id, *, mode="original"):
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("readout diagnostic is CPU-only")
    ids, targets, selected = task.inputs(mask_id, mode=mode)
    captured = []
    handle = model.final_norm.register_forward_pre_hook(lambda m, args: captured.append(args[0]))
    previous = model.training
    model.eval()
    try:
        logits = model(
            ids,
            attention_mask=torch.ones_like(ids, dtype=torch.bool),
            noise_level=1 / ids.shape[1],
        )
        scores = logits[selected][:, :mask_id].float()
        truth = targets[selected]
        alternatives = torch.where(
            truth == task.target_ids[0], task.target_ids[1], task.target_ids[0]
        )
        rows = torch.arange(len(truth))
        correct = scores[rows, truth]
        others = scores.clone()
        others[rows, truth] = -torch.inf
        probabilities = scores.softmax(-1)
        full_loss = task_loss(logits, targets, selected, mask_id, task.target_ids).item()
        return {
            "full_vocabulary_ce": full_loss,
            "two_target_ce": task_loss(
                logits, targets, selected, mask_id, task.target_ids, two_target=True
            ).item(),
            "balanced_unigram_ce": math.log(2),
            "full_ce_minus_balanced_unigram": full_loss - math.log(2),
            "full_accuracy": (scores.argmax(-1) == truth).float().mean().item(),
            "two_target_accuracy": (
                torch.tensor(task.target_ids)[scores[:, task.target_ids].argmax(-1)] == truth
            )
            .float()
            .mean()
            .item(),
            "predictions": scores.argmax(-1).tolist(),
            "targets": truth.tolist(),
            "target_margin_vs_best_other": (correct - others.max(-1).values).tolist(),
            "target_margin_vs_paired_class": (correct - scores[rows, alternatives]).tolist(),
            "entropy": (-(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)).tolist(),
            "target_pair_probability_mass": probabilities[:, task.target_ids].sum(-1).tolist(),
            "pre_norm_all_tokens": centered_energy(captured[0]),
            "pre_norm_target_contexts": centered_energy(captured[0][selected]),
            "finite_logits_and_hidden": bool(
                torch.isfinite(logits).all() and torch.isfinite(captured[0]).all()
            ),
        }
    finally:
        handle.remove()
        model.train(previous)


def run_case(config, task, mask_id, *, name, untied, two_target, steps, seed, progress=None):
    """Fresh isolated CPU model; never save it or use AcceleratorTrainer resume contracts."""
    if type(steps) is not int or not 1 <= steps <= 200:
        raise ValueError("fixed corpus replay must be 1-200 steps")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = build_encoder(config)
        if untied:
            # Clone, rather than randomly reinitialize: step-zero logits and RNG match exactly.
            # Do not call .to() after this diagnostic-only surgery: production _apply re-ties.
            model.reconstruction_head.weight = nn.Parameter(
                model.token_embedding.weight.detach().clone()
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.0003, weight_decay=0.01, foreach=False, fused=False
        )
        tokens_per_step = task.clean.numel()
        schedule = AcceleratorTrainConfig(
            device="cpu",
            precision="fp32",
            max_tokens=steps * tokens_per_step,
            warmup_tokens=min(10, steps - 1) * tokens_per_step,
            learning_rate=0.0003,
            min_learning_rate=0.00003,
            batch_size=4,
            grad_accum_steps=1,
            seed=seed,
        )
        ids, clean, selected = task.inputs(mask_id)
        valid = torch.ones_like(ids, dtype=torch.bool)
        trajectory = [{"step": 0, **score_task(model, task, mask_id)}]
        training = []
        for step in range(1, steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, attention_mask=valid, noise_level=1 / ids.shape[1])
            loss = task_loss(
                logits, clean, selected, mask_id, task.target_ids, two_target=two_target
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, foreach=False)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError("nonfinite fixed corpus replay")
            lr = learning_rate(schedule, step * tokens_per_step)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.step()
            training.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "gradient_norm": norm.item(),
                    "learning_rate": lr,
                }
            )
            if step % 20 == 0 or step == steps:
                trajectory.append({"step": step, **score_task(model, task, mask_id)})
                if progress:
                    progress(f"{name}: update {step}, loss {loss.item():.4f}")
        swap = score_task(model, task, mask_id, mode="swap")
        erase = score_task(model, task, mask_id, mode="erase")
        final = trajectory[-1]
        return {
            "case": name,
            "untied": untied,
            "two_target_loss": two_target,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "training": schedule.to_dict(),
            "steps": steps,
            "input_tokens": steps * tokens_per_step,
            "masked_targets_seen": steps * 4,
            "trajectory": trajectory,
            "training_rows": training,
            "swapped_context": swap,
            "erased_context": erase,
            "full_task_passed": final["full_accuracy"] == swap["full_accuracy"] == 1.0
            and final["full_vocabulary_ce"] < 0.1
            and erase["full_accuracy"] == 0.5,
            "two_target_task_passed": final["two_target_accuracy"]
            == swap["two_target_accuracy"]
            == 1.0
            and final["two_target_ce"] < 0.1
            and erase["two_target_accuracy"] == 0.5,
        }


def readout_diagnostic(run_dir: Path, data_dir: Path, *, steps=100, seed=42, progress=None):
    if type(steps) is not int or not 1 <= steps <= 200:
        raise ValueError("fixed corpus replay must be 1-200 steps")
    if type(seed) is not int or not 0 <= seed < 2**32 - 1:
        raise ValueError("invalid model seed")
    payload, document, manifest = _audit_payload(run_dir)
    contract = document["job"]["contract"]
    tokenizer = MaskBPETokenizer(Tokenizer.from_str(document["tokenizer_json"]))
    if tokenizer.fingerprint != contract["tokenizer_sha256"]:
        raise ValueError("tokenizer fingerprint mismatch")
    data = PreparedTokenData(data_dir, expected_fingerprint=contract["data_fingerprint"])
    tokenizer.assert_base_compatible(data.tokenizer)
    config = KiwiLM3Config.from_dict(contract["model"])
    if config.d_model > 512 or config.num_blocks > 16:
        raise ValueError("diagnostic model exceeds local bounds")
    task = corpus_task(data, tokenizer)
    if task.clean.shape[1] > config.context_length:
        raise ValueError("checkpoint context is shorter than the diagnostic task")
    del payload["optimizer"]
    state = payload["model"]
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in state.values()):
        raise ValueError("checkpoint model must be finite FP32")
    if not torch.equal(state["token_embedding.weight"], state["reconstruction_head.weight"]):
        raise ValueError("checkpoint tied weights disagree")
    with torch.random.fork_rng(devices=[]):
        frozen = build_encoder(config)
        frozen.load_state_dict(state, strict=True)
        frozen_score = score_task(frozen, task, tokenizer.mask_id)
        del frozen, state, payload
        cases = []
        for name, untied, two_target in CASES:
            if progress:
                progress(f"Fresh CPU fixed-corpus replay: {name}, {steps} updates")
            cases.append(
                run_case(
                    config,
                    task,
                    tokenizer.mask_id,
                    name=name,
                    untied=untied,
                    two_target=two_target,
                    steps=steps,
                    seed=seed,
                    progress=progress,
                )
            )
    return {
        "schema_version": 1,
        "scope": (
            "Four fixed balanced corpus windows; diagnostic memorization, "
            "not generalization or GPU parity"
        ),
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "threads": torch.get_num_threads(),
        },
        "source": {
            "checkpoint_sha256": manifest["files"]["latest.pt"]["sha256"],
            "job_identity": document["identity"],
            "data_fingerprint": data.fingerprint,
            "tokenizer_sha256": tokenizer.fingerprint,
        },
        "model": config.to_dict(),
        "task": task.to_dict(),
        "target_strings": [tokenizer.decode([i]) for i in task.target_ids],
        "frozen_checkpoint_score": frozen_score,
        "cases": cases,
        "limitations": [
            "One model seed and four training windows, no held-out generalization",
            "Context 64 rather than 512, fixed one-target masks rather than variable noise",
            "Two-target loss changes prediction support, not a proposed diffusion objective",
            "Whole-context swap checks pairing, not semantic edits or unseen contexts",
            "Erased context is an out-of-distribution identical-input negative control",
            "Balanced unigram is the target-only prior, not a smoothed full-corpus baseline",
            "Untied heads clone initial weights and add parameters; production defaults unchanged",
        ],
        "cloud_allocated": False,
        "checkpoint_modified": False,
        "canonical_fix": None,
    }
