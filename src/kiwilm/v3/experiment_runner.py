"""Opt-in, bounded CPU B/C execution with exclusive artifacts and locked resume."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch

from kiwilm.data import PreparedTokenData
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.checkpoints import load_training_state, save_training_state
from kiwilm.v3.evaluation import evaluate_candidate
from kiwilm.v3.experiments import COMPARISONS, GENERATION_CASES, validate_comparison, validate_suite
from kiwilm.v3.masking import MaskingConfig
from kiwilm.v3.objectives import OBJECTIVE
from kiwilm.v3.sampling import SamplingConfig, generate_slots
from kiwilm.v3.tokenizer import MaskBPETokenizer
from kiwilm.v3.trainer import DenoisingTrainConfig, DenoisingTrainer
from kiwilm.v3.weights import load_encoder_weights, save_encoder_weights


def checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: dict) -> Path:
    """Exclusive creation, no implicit overwrite. Never resolve caller paths into reports."""
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def preflight(
    suite: dict, candidate: str, data: PreparedTokenData, tokenizer: MaskBPETokenizer
) -> dict:
    validate_suite(suite)
    if candidate not in suite["candidates"]:
        raise ValueError("only the four B/C dense candidates are available")
    tokenizer.assert_base_compatible(data.tokenizer)
    if (
        data.fingerprint != suite["data_fingerprint"]
        or tokenizer.fingerprint != suite["tokenizer_sha256"]
    ):
        raise ValueError("suite data/tokenizer provenance mismatch")
    context = suite["controls"]["context_length"]
    if any(len(data.tokens(split)) <= context for split in ("train", "validation")):
        raise ValueError("prepared splits must exceed experiment context")
    for case in GENERATION_CASES:
        length = len(tokenizer.encode(case["prompt"])) + len(tokenizer.encode(case["suffix"]))
        if length + suite["controls"]["generation_slots"] + 2 > context:
            raise ValueError("frozen generation suite exceeds context; enlarge the planned context")
    return {
        "candidate": candidate,
        "suite_identity": suite["identity"],
        "profile": suite["candidates"][candidate]["profile"],
        "training_started": False,
        "runtime": suite["runtime"],
    }


def _provenance(suite, candidate):
    return {
        "suite_identity": suite["identity"],
        "candidate": candidate,
        "data_fingerprint": suite["data_fingerprint"],
        "tokenizer_sha256": suite["tokenizer_sha256"],
        "objective": OBJECTIVE,
        "model": suite["candidates"][candidate]["model"],
        "controls": suite["controls"],
    }


def run_candidate(
    suite: dict,
    candidate: str,
    data: PreparedTokenData,
    tokenizer: MaskBPETokenizer,
    output_dir: str | Path,
    *,
    start_training: bool = False,
    stop_after_step: int | None = None,
    resume_receipt: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict:
    """Execute only when explicitly requested; resume never falls back to fresh.

    Periodic local checkpoints preserve exact optimizer/data/noise/dropout state.
    This is not the production latest/previous Drive protocol or a TPU adapter.
    A metrics tail beyond the checkpoint is refused, never silently truncated.
    """
    preview = preflight(suite, candidate, data, tokenizer)
    if type(start_training) is not bool:
        raise ValueError("start_training must be an explicit boolean")
    if not start_training:
        return preview
    controls = validate_suite(suite)
    stop = controls.max_steps if stop_after_step is None else stop_after_step
    if type(stop) is not int or not 1 <= stop <= controls.max_steps:
        raise ValueError("stop_after_step must lie inside the frozen budget")
    if bool(resume_receipt) != bool(expected_checkpoint_sha256):
        raise ValueError("resume requires both a receipt and the original checkpoint SHA256")
    root = Path(output_dir)
    receipt = None
    rows = []
    if resume_receipt:
        receipt = json.loads(Path(resume_receipt).read_text(encoding="utf-8"))
        if receipt.get("provenance") != _provenance(suite, candidate):
            raise ValueError("resume receipt has incompatible B/C controls")
        name = receipt.get("checkpoint", "")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pt"):
            raise ValueError("receipt checkpoint must be a local basename")
        if receipt.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise ValueError("resume receipt/checkpoint checksum mismatch")
        if not root.is_dir() or Path(resume_receipt).resolve().parent != root.resolve():
            raise ValueError("resume receipt must belong to this existing candidate directory")
        rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        if [row.get("step") for row in rows] != list(range(1, receipt["step"] + 1)):
            raise ValueError(
                "metrics are not the verified checkpoint prefix; no implicit truncation"
            )
        if checksum(root / "metrics.jsonl") != receipt.get("metrics_sha256"):
            raise ValueError("metrics checksum differs from the verified checkpoint prefix")
        if stop <= receipt["step"]:
            raise ValueError("continuation must advance the checkpoint within its original budget")
    elif expected_checkpoint_sha256 is not None:
        raise ValueError("checkpoint checksum requires an explicit resume receipt")
    else:
        root.mkdir(parents=True, exist_ok=False)
    lock = root / ".training.lock"
    # Never remove another process's stale lock; this owner releases its own.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        torch.manual_seed(controls.initialization_seed)
        model = build_encoder(KiwiLM3Config.from_dict(suite["candidates"][candidate]["model"]))
        trainer = DenoisingTrainer(
            model,
            tokenizer,
            data,
            DenoisingTrainConfig(**controls.training_values()),
            MaskingConfig(**suite["masking"]),
        )
        torch.manual_seed(controls.initialization_seed)
        if receipt:
            load_training_state(
                trainer, root / receipt["checkpoint"], expected_sha256=expected_checkpoint_sha256
            )
            if trainer.step != receipt["step"] or trainer.identity != receipt["trainer_identity"]:
                raise ValueError("receipt progress/identity differs from checkpoint")
        checkpoint = None
        with (root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            while trainer.step < stop:
                start = time.perf_counter()
                before = trainer.tokens_seen
                row = trainer.train_step()
                row["step_seconds"] = time.perf_counter() - start
                row["tokens_per_second"] = (trainer.tokens_seen - before) / row["step_seconds"]
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if trainer.step % controls.checkpoint_interval == 0 or trainer.step == stop:
                    os.fsync(stream.fileno())
                    checkpoint = save_training_state(trainer, root / f"step-{trainer.step:06d}.pt")
                    write_json(
                        checkpoint.with_suffix(".json"),
                        {
                            "provenance": _provenance(suite, candidate),
                            "step": trainer.step,
                            "trainer_identity": trainer.identity,
                            "checkpoint": checkpoint.name,
                            "checkpoint_sha256": checksum(checkpoint),
                            "metrics_sha256": checksum(root / "metrics.jsonl"),
                        },
                    )
        report = {
            "provenance": _provenance(suite, candidate),
            "step": trainer.step,
            "optimizer_steps": trainer.optimizer_steps,
            "tokens_seen": trainer.tokens_seen,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checksum(checkpoint),
            "data_rng_sha256": hashlib.sha256(
                trainer.data_generator.get_state().numpy().tobytes()
            ).hexdigest(),
            "noise_rng_sha256": hashlib.sha256(
                trainer.noise_generator.get_state().numpy().tobytes()
            ).hexdigest(),
            "runtime": trainer.contract["runtime"],
            "code_sha256": trainer.contract["code_sha256"],
            "status": "complete" if trainer.step == controls.max_steps else "paused",
            "profile": preview["profile"],
            "peak_accelerator_memory_bytes": None,
            "throughput": {
                "median_tokens_per_second": statistics.median(
                    row["tokens_per_second"] for row in (rows[5:] or rows)
                ),
                "excluded_initial_steps": 5 if len(rows) > 5 else 0,
                "note": (
                    "Measured CPU step time only, excluding validation/checkpoint I/O; "
                    "not TPU performance."
                ),
            },
        }
        if trainer.step == controls.max_steps:
            report.update(evaluate_candidate(trainer, controls))
            weights = save_encoder_weights(
                model, root / "encoder.pt", tokenizer_sha256=tokenizer.fingerprint
            )
            loaded, _ = load_encoder_weights(
                weights,
                expected_sha256=checksum(weights),
                expected_tokenizer_sha256=tokenizer.fingerprint,
            )
            probe, _ = data.get_batch(
                "validation",
                batch_size=2,
                context_length=controls.context_length,
                generator=torch.Generator().manual_seed(controls.validation_seed),
            )
            model.eval()
            with torch.no_grad():
                logit_parity = torch.equal(
                    model(probe, noise_level=1.0), loaded(probe, noise_level=1.0)
                )
            generation = dict(
                prompt="",
                output_slots=controls.generation_slots,
                config=SamplingConfig(controls.sampler_steps, 0.0, 0, controls.generation_seeds[0]),
            )
            sampling_parity = (
                generate_slots(model, tokenizer, **generation)["token_ids"]
                == generate_slots(loaded, tokenizer, **generation)["token_ids"]
            )
            report["inference_parity"] = {
                "fp32_logits_exact": logit_parity,
                "greedy_slots_exact": sampling_parity,
            }
            safe = save_encoder_weights(
                model, root / "encoder.safetensors", tokenizer_sha256=tokenizer.fingerprint
            )
            report["inference_weights"] = {
                "fp32": {"path": str(weights), "sha256": checksum(weights)},
                "bf16": {"path": str(safe), "sha256": checksum(safe)},
                "bf16_parity": "Not bitwise-equivalent; exact qualification uses FP32 weights.",
            }
        else:
            report["validation"] = None
        path = root / f"report-{trainer.step:06d}.json"
        write_json(path, report)
        return report
    finally:
        lock.unlink()


def write_comparisons(suite: dict, reports: dict[str, dict], output_dir: str | Path) -> dict:
    """Reviewable portable B/C results. Refuse incomplete/mismatched runs before writing."""
    decisions = {key: validate_comparison(reports, suite, key) for key in COMPARISONS}
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"comparison already exists: {root}")
    root.mkdir(parents=True)
    write_json(root / "suite.json", suite)
    write_json(root / "results.json", {"reports": reports, "comparisons": decisions})
    lines = [
        "# KiwiLM 3 — B/C dense denoising comparison",
        "",
        "No automatic winner. Loss is masked reconstruction CE, not autoregressive perplexity.",
        "",
        "| Candidate | Parameters | Forward FLOPs/token | CE @ 15% | CE @ 100% | CPU tok/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, report in reports.items():
        losses = {row["noise_level"]: row["loss"] for row in report["validation"]["rows"]}
        lines.append(
            f"| {name} | {report['profile']['parameters']} | "
            f"{report['profile']['forward_flops_per_token']:.0f} | "
            f"{losses[0.15]} | {losses[1.0]} | "
            f"{report['throughput']['median_tokens_per_second']:.1f} |"
        )
    lines += [
        "",
        "Health distributions, inference parity and every generation sample are in results.json.",
        "CPU step times exclude checkpoint/evaluation I/O; accelerator memory/cost,",
        "transfer/retrieval remain unmeasured. Review semantics and health before selection.",
    ]
    with (root / "analysis.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return {"output_dir": str(root), "comparisons": decisions, "canonical_winner": None}
