"""Notebook-first V3 preflight/train/resume. No VM provisioning or implicit training."""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.v3.accelerator import (
    ENGINE,
    AcceleratorTrainConfig,
    AcceleratorTrainer,
    candidate_config,
)
from kiwilm.v3.accelerator_checkpoint import V3CheckpointStore
from kiwilm.v3.experiments import canonical_digest
from kiwilm.v3.profile import profile_encoder
from kiwilm.v3.tokenizer import MaskBPETokenizer


def vm_identity() -> str | None:
    source = Path("/proc/sys/kernel/random/boot_id")
    return source.read_text().strip() if source.is_file() else None


def prepare_demo(data_dir: Path, tokenizer_path: Path) -> dict:
    if data_dir.exists() or tokenizer_path.exists():
        raise FileExistsError("demo paths exist; reuse explicitly instead of overwriting")
    prepare_from_stories(
        data_dir,
        ["The kiwi found seeds and shared them with a friend. " * 40] * 8,
        ["The friend shared the seeds with the kiwi. " * 40] * 2,
        dataset_name="kiwilm3-synthetic-recovery-probe",
        vocab_size=300,
        min_frequency=1,
        show_progress=False,
    )
    data = PreparedTokenData(data_dir)
    tokenizer = MaskBPETokenizer.from_base(data.tokenizer)
    tokenizer.save(tokenizer_path)
    return {
        "training_started": False,
        "synthetic": True,
        "data_fingerprint": data.fingerprint,
        "tokenizer_sha256": tokenizer.fingerprint,
        "vocab_size": tokenizer.vocab_size,
    }


def construct(
    config: AcceleratorTrainConfig,
    *,
    data_dir: Path,
    tokenizer_path: Path,
    candidate: str,
    qualification: bool,
    run_name: str,
    collapse_diagnostics: bool = False,
):
    if type(collapse_diagnostics) is not bool or (
        collapse_diagnostics and config.device not in {"cpu", "cuda"}
    ):
        raise ValueError("collapse_diagnostics must be boolean and requires CPU/CUDA")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_name):
        raise ValueError("run_name must be a short lowercase hyphenated identifier")
    data, tokenizer = PreparedTokenData(data_dir), MaskBPETokenizer.load(tokenizer_path)
    if config.device == "cpu":
        torch.set_num_threads(1)
    random.seed(config.seed)
    np.random.seed(config.seed)
    model = candidate_config(candidate, tokenizer.vocab_size, qualification=qualification)
    trainer = AcceleratorTrainer(model, config, tokenizer, data)
    job = {
        "schema": 1,
        "engine": ENGINE,
        "candidate": candidate,
        "qualification": qualification,
        "run_name": run_name,
        "contract": trainer.contract,
    }
    if collapse_diagnostics:
        from kiwilm.v3.training_diagnostics import POLICY

        job["collapse_diagnostics"] = dict(POLICY)
    return trainer, job


def namespace(backup_root: Path, job: dict) -> Path:
    return (
        backup_root
        / "checkpoints"
        / f"v3-{job['run_name']}-{job['candidate']}-{canonical_digest(job)[:16]}"
    )


def preflight(config: AcceleratorTrainConfig, **kwargs) -> dict:
    trainer, job = construct(config, **kwargs)
    runtime = trainer.runtime
    ids = torch.tensor(
        [[trainer.tokenizer.bos_id, trainer.tokenizer.mask_id, trainer.tokenizer.eos_id]],
        device=runtime.device,
    )
    trainer.model.eval()
    with torch.no_grad(), runtime.autocast():
        output = trainer.model(ids, noise_level=torch.tensor([1.0]))
    runtime.sync()
    if not torch.isfinite(output).all().item():
        raise FloatingPointError("V3 backend/model preflight produced nonfinite logits")
    return {
        "engine": ENGINE,
        "job": job,
        "identity": canonical_digest(job),
        "training_started": False,
        "profile": profile_encoder(trainer.model.config),
        "counters": runtime.counters(),
        "memory": runtime.memory(),
        "live_continuation_qualified": False,
    }


def _append(path: Path, row: dict):
    text = json.dumps(row, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(text, flush=True)


def train(
    config: AcceleratorTrainConfig,
    *,
    data_dir: Path,
    tokenizer_path: Path,
    candidate: str,
    qualification: bool,
    run_name: str,
    run_dir: Path,
    backup_root: Path,
    mode: str,
    start_training: bool = False,
    storage_root: Path | None = None,
    stop_after_step: int | None = None,
    require_new_vm: bool = False,
    vm_id: str | None = None,
    retry_delay: float = 5,
    collapse_diagnostics: bool = False,
) -> dict:
    if start_training is not True:
        raise ValueError("training disabled; explicitly enable start_training=True")
    if mode not in {"fresh", "resume"}:
        raise ValueError("explicit fresh/resume mode required; never fallback to fresh")
    if stop_after_step is not None and (type(stop_after_step) is not int or stop_after_step < 1):
        raise ValueError("stop_after_step must be a positive completed step")
    if config.device != "cpu" and storage_root is None:
        raise ValueError("accelerator training requires a checked mounted Drive storage_root")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("use a new empty VM-local run directory; never overwrite progress")
    trainer, job = construct(
        config,
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
        candidate=candidate,
        qualification=qualification,
        run_name=run_name,
        collapse_diagnostics=collapse_diagnostics,
    )
    store = V3CheckpointStore(
        namespace(backup_root, job), job=job, storage_root=storage_root, retry_delay=retry_delay
    )
    vm_id = vm_id or vm_identity()
    origin = None
    if mode == "fresh":
        store.transport.check()
        if (
            (store.root / "latest.json").exists()
            or (store.root / "previous.json").exists()
            or any(store.root.glob("step-*"))
        ):
            raise FileExistsError(
                "backup already has progress or staged files; explicitly resume or choose a new run"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.jsonl").touch(exist_ok=False)
        (run_dir / "job.json").write_text(
            json.dumps(
                {
                    "job": job,
                    "identity": store.identity,
                    "tokenizer_json": trainer.tokenizer.to_json(),
                },
                sort_keys=True,
            )
            + "\n"
        )
    else:
        origin = store.restore(trainer, run_dir, vm_id=vm_id, require_new_vm=require_new_vm)
    metrics = run_dir / "metrics.jsonl"
    monitor = None
    if collapse_diagnostics:
        from kiwilm.v3.training_diagnostics import CollapseMonitor

        monitor = CollapseMonitor(trainer)
    committed = None
    if mode == "fresh":
        if monitor is not None:
            _append(metrics, trainer.evaluate())
            _append(metrics, monitor.evaluate())
        committed = store.publish(trainer, run_dir, vm_id=vm_id)
        print(json.dumps(committed), flush=True)  # step zero before any training
    while trainer.tokens_seen < config.max_tokens and (
        stop_after_step is None or trainer.step < stop_after_step
    ):
        started = time.perf_counter()
        row = trainer.train_step()
        row["step_seconds"] = time.perf_counter() - started
        row["tokens_per_second"] = row["input_tokens"] / row["step_seconds"]
        _append(metrics, row)
        complete = trainer.tokens_seen == config.max_tokens
        stopping = stop_after_step is not None and trainer.step == stop_after_step
        if trainer.step % config.eval_interval == 0 or complete or stopping:
            _append(metrics, trainer.evaluate())
            if monitor is not None:
                _append(metrics, monitor.evaluate())
        if trainer.step % config.checkpoint_interval == 0 or complete or stopping:
            # No catch-and-continue: failed publication halts the run with local
            # latest.pt retained and the previous Drive commit unchanged.
            committed = store.publish(trainer, run_dir, vm_id=vm_id)
            print(json.dumps(committed), flush=True)
    return {
        "engine": ENGINE,
        "identity": store.identity,
        "job": job,
        "step": trainer.step,
        "optimizer_steps": trainer.optimizer_steps,
        "tokens_seen": trainer.tokens_seen,
        "masked_tokens_seen": trainer.masked_tokens_seen,
        "status": "complete" if trainer.tokens_seen == config.max_tokens else "paused",
        "backup_dir": str(store.root),
        "last_commit": committed,
        "resume_origin": origin,
        "vm_id": vm_id,
        "counters": trainer.runtime.counters(),
        "memory": trainer.runtime.memory(),
        "live_continuation_qualified": False,
    }
