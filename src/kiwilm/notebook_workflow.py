"""M2 notebook qualification, deliberately separate from the frozen V2 trainer.

The tiny causal model is a recovery probe, NOT the future V3 encoder/objective.
No training starts on import. Paths are operational, never part of run identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kiwilm.checkpoint import load_checkpoint, save_checkpoint
from kiwilm.config import KiwiLM2Config
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.diagnostics import cached_generation_parity_report, model_health_report
from kiwilm.generation import generate
from kiwilm.safetensors_io import export_safetensors_bundle
from kiwilm.tpu_checkpoint import DriveCheckpointStore
from kiwilm.tpu_smoke import Runtime, prepare_tied_model
from kiwilm.training import TrainConfig, learning_rate_at_tokens, next_token_loss

SCHEMA = 1
ENGINE = "m2-v2-causal-recovery-probe"


@dataclass(frozen=True)
class NotebookConfig:
    experiment: str = "m2-recovery-smoke"
    engine: str = ENGINE
    device: str = "cpu"
    precision: str = "fp32"
    max_tokens: int = 256
    warmup_tokens: int = 64
    batch_size: int = 2
    grad_accum_steps: int = 1
    context_length: int = 16
    seed: int = 42
    noise_seed: int = 142
    checkpoint_interval: int = 2
    eval_interval: int = 2
    eval_batches: int = 2
    lr: float = 0.001

    def __post_init__(self) -> None:
        if self.engine != ENGINE:
            raise ValueError("M2 supports only the labelled V2 recovery probe; V3 is not built yet")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", self.experiment):
            raise ValueError("experiment must be a short lowercase, hyphenated identifier")
        if self.device not in {"cpu", "cuda", "xla"}:
            raise ValueError("device must explicitly select cpu, cuda or xla (no fallback)")
        allowed = {"cpu": {"fp32"}, "cuda": {"fp32", "bf16", "fp16"}, "xla": {"bf16"}}
        if self.precision not in allowed[self.device]:
            raise ValueError("precision is incompatible with the selected backend")
        for name in (
            "max_tokens",
            "batch_size",
            "grad_accum_steps",
            "context_length",
            "checkpoint_interval",
            "eval_interval",
            "eval_batches",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "noise_seed", "warmup_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.seed >= 2**32 or self.noise_seed >= 2**32:
            raise ValueError("seed and noise_seed must be below 2**32")
        if (
            self.max_tokens > 1_000_000
            or self.context_length > 64
            or self.batch_size > 32
            or self.grad_accum_steps > 32
        ):
            raise ValueError("M2 is a bounded qualification probe, not a production training run")
        if self.warmup_tokens > self.max_tokens:
            raise ValueError("warmup_tokens cannot exceed the full token budget")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")

    def model_config(self, vocab_size: int) -> KiwiLM2Config:
        return KiwiLM2Config(
            vocab_size=vocab_size,
            context_length=self.context_length,
            d_model=16,
            num_query_heads=2,
            num_kv_heads=1,
            swiglu_dim=32,
            dropout=0.1,
            bigram_buckets=17,
            trigram_buckets=19,
            conv_kernel_sizes=(3, 5, 3, 5, 3, 5),
        )

    def train_config(self) -> TrainConfig:
        return TrainConfig(
            max_steps=math.ceil(self.max_tokens / self.tokens_per_step),
            max_tokens=self.max_tokens,
            warmup_tokens=self.warmup_tokens,
            batch_size=self.batch_size,
            grad_accum_steps=self.grad_accum_steps,
            lr=self.lr,
            min_lr=self.lr / 10,
            precision=self.precision,
            seed=self.seed,
            checkpoint_interval=self.checkpoint_interval,
            eval_interval=self.eval_interval,
            eval_batches=self.eval_batches,
        )

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.context_length * self.grad_accum_steps


def implementation_digest() -> str:
    """Lock installed Python math/workflow, independently of checkout/VM paths."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def experiment_contract(config: NotebookConfig, data: PreparedTokenData) -> dict[str, Any]:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    if config.device == "cpu":
        versions["cpu_threads"] = torch.get_num_threads()
    if config.device == "xla":
        from importlib.metadata import version

        versions["torch_xla"] = version("torch_xla")
        if versions["torch"].split("+")[0] != versions["torch_xla"].split("+")[0]:
            raise ValueError("TPU requires matched torch and torch_xla releases")
    contract = {
        "schema": SCHEMA,
        "config": asdict(config),
        "model_config": config.model_config(data.tokenizer.vocab_size).to_dict(),
        "train_config": config.train_config().to_dict(),
        "data_fingerprint": data.fingerprint,
        "tokenizer_sha256": data.metadata["tokenizer"]["sha256"],
        "implementation_sha256": implementation_digest(),
        "runtime_versions": versions,
        "optimizer": "torch-adamw-single-tensor",
        "noise_role": "reserved-rng-probe-only",
    }
    # Canonical primitives also normalize tuple-valued model configuration.
    return json.loads(json.dumps(contract, sort_keys=True))


def contract_identity(contract: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def backup_namespace(root: Path, config: NotebookConfig, identity: str) -> Path:
    return root / "checkpoints" / f"{config.experiment}-{identity[:16]}"


def prepare_probe_data(destination: Path) -> dict[str, Any]:
    """Local synthetic corpus; never downloads pretraining data or overwrites files."""
    if destination.exists():
        return PreparedTokenData(destination).metadata
    return prepare_from_stories(
        destination,
        [
            f"The kiwi found {index} seeds. It shared the seeds with a friend."
            for index in range(128)
        ],
        [f"A friend found {index} seeds and shared them with the kiwi." for index in range(32)],
        dataset_name="m2-synthetic-recovery-probe",
        vocab_size=300,
        min_frequency=1,
    )


def preflight(config: NotebookConfig, data_dir: Path) -> dict[str, Any]:
    data = PreparedTokenData(data_dir, seed=config.seed)
    for split in ("train", "validation"):
        if len(data.tokens(split)) <= config.context_length:
            raise ValueError(f"{split} is too short for the selected context")
    runtime = Runtime(config.device, config.precision)
    if (
        config.device == "cuda"
        and config.precision == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("this GPU does not support BF16; explicitly choose FP16 or FP32")
    with runtime.autocast():
        probe = torch.ones((8, 8), device=runtime.device)
        result = probe @ probe
    runtime.sync()
    if not torch.isfinite(result).all().item():
        raise RuntimeError("backend preflight produced non-finite values")
    contract = experiment_contract(config, data)
    return {
        "engine": ENGINE,
        "training_started": False,
        "contract": contract,
        "identity": contract_identity(contract),
        "backend": str(runtime.device),
        "counters": runtime.counters(),
    }


class TokenSchedule:
    def __init__(self, config: TrainConfig) -> None:
        self.config, self.tokens_seen = config, 0

    def state_dict(self) -> dict[str, int]:
        return {"tokens_seen": self.tokens_seen}

    def load_state_dict(self, state: dict) -> None:
        tokens = state.get("tokens_seen")
        if type(tokens) is not int or not 0 <= tokens <= self.config.max_tokens:
            raise ValueError("invalid token schedule progress")
        self.tokens_seen = tokens


def _numpy_state() -> dict[str, Any]:
    algorithm, keys, position, gaussian, cached = np.random.get_state()
    return {
        "algorithm": algorithm,
        "keys": keys.tolist(),
        "position": position,
        "gaussian": gaussian,
        "cached": cached,
    }


def _restore_numpy(state: dict) -> None:
    np.random.set_state(
        (
            state["algorithm"],
            np.asarray(state["keys"], dtype=np.uint32),
            state["position"],
            state["gaussian"],
            state["cached"],
        )
    )


def _append_metric(path: Path, metric: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metric, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(metric, sort_keys=True), flush=True)


def _validate_payload(payload: dict, contract: dict, identity: str) -> None:
    state = payload.get("training_state", {})
    if (
        state.get("notebook_schema") != SCHEMA
        or state.get("identity") != identity
        or (state.get("contract") != contract)
    ):
        raise ValueError("resume experiment fingerprint differs; refusing to restart or warm-start")
    if (
        payload.get("model_config") != contract["model_config"]
        or payload.get("train_config") != contract["train_config"]
        or payload.get("data_fingerprint") != contract["data_fingerprint"]
    ):
        raise ValueError("checkpoint model/training/data schema differs from its contract")
    config = NotebookConfig(**contract["config"])
    if (
        type(payload.get("step")) is not int
        or not 0 <= payload["step"] <= config.train_config().max_steps
    ):
        raise ValueError("checkpoint step is outside the locked budget")
    required = ("optimizer_state_dict", "scheduler_state_dict", "rng_state", "batcher_state")
    if any(not isinstance(payload.get(name), dict) for name in required):
        raise ValueError("checkpoint is missing required recovery state")
    streams = payload["batcher_state"].get("generators", {})
    if (
        set(streams) != {"validation", "noise"}
        or "owner" not in payload["batcher_state"]
        or (not {"python", "torch"}.issubset(payload["rng_state"]))
    ):
        raise ValueError("checkpoint is missing data/global/noise RNG recovery state")
    if (
        "numpy_rng" not in state
        or "scaler" not in state
        or (contract["config"]["device"] == "xla" and "xla_rng" not in state)
        or (contract["config"]["device"] == "cuda" and "cuda" not in payload["rng_state"])
    ):
        raise ValueError("checkpoint is missing backend/NumPy/scaler recovery state")
    tokens = payload["scheduler_state_dict"].get("tokens_seen")
    expected = min(
        payload["step"] * NotebookConfig(**contract["config"]).tokens_per_step,
        contract["config"]["max_tokens"],
    )
    if tokens != state.get("tokens_seen") or tokens != expected:
        raise ValueError("checkpoint step/token/scheduler progress is inconsistent")


def _validate_metrics(path: Path, step: int) -> None:
    if not path.is_file():
        raise ValueError("committed checkpoint is missing its metrics log")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    train_steps = [row["step"] for row in rows if row.get("event") == "train"]
    if train_steps != list(range(1, step + 1)) or any(row["step"] > step for row in rows):
        raise ValueError("committed metrics do not match checkpoint progress")


def train_probe(
    config: NotebookConfig,
    *,
    data_dir: Path,
    run_dir: Path,
    backup_root: Path,
    mode: str,
    start_training: bool = False,
    stop_after_step: int | None = None,
    storage_root: Path | None = None,
) -> dict[str, Any]:
    """Explicit train/resume. Interruptions retain the last complete committed boundary."""
    if start_training is not True:
        raise ValueError("training is disabled; explicitly set start_training=True")
    if mode not in {"fresh", "resume"}:
        raise ValueError("choose fresh or resume explicitly; there is no automatic fallback")
    total_steps = config.train_config().max_steps
    if stop_after_step is not None and (
        type(stop_after_step) is not int or not 1 <= stop_after_step <= total_steps
    ):
        raise ValueError("stop_after_step must be an optimizer boundary within the full budget")
    checked = preflight(config, data_dir)
    identity, contract = checked["identity"], checked["contract"]
    store = DriveCheckpointStore(
        backup_namespace(backup_root, config, identity),
        identity=identity,
        storage_root=storage_root,
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "use a new empty VM-local run directory (never overwrite local progress)"
        )
    if mode == "fresh":
        store.check()
    if mode == "fresh" and (
        any(store.root.glob("step-*")) or (store.root / "latest.json").exists()
    ):
        raise FileExistsError("backup already contains progress; explicitly resume instead")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, metrics_path = run_dir / "latest.pt", run_dir / "metrics.jsonl"
    if mode == "resume":
        restored = store.restore(run_dir)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _validate_payload(payload, contract, identity)
        if restored["step"] != payload["step"]:
            raise ValueError("manifest/checkpoint step mismatch")
        _validate_metrics(metrics_path, payload["step"])
        store.validate_progress(payload["step"])
    else:
        metrics_path.touch()
        payload = None
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    data = PreparedTokenData(data_dir, seed=config.seed)
    runtime = Runtime(config.device, config.precision)
    if runtime.xm is not None:
        runtime.xm.set_rng_state(config.seed, runtime.device)
    model_config = config.model_config(data.tokenizer.vocab_size)
    model = prepare_tied_model(model_config, runtime.device)
    settings = config.train_config()
    lr = torch.tensor(config.lr, device=runtime.device if runtime.xla else "cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, settings.beta2),
        weight_decay=settings.weight_decay,
        foreach=False,
        fused=False,
        capturable=runtime.xla is not None,
    )
    schedule = TokenSchedule(settings)
    streams = {
        "validation": torch.Generator().manual_seed(config.seed + 1),
        "noise": torch.Generator().manual_seed(config.noise_seed),
    }
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.device == "cuda" and config.precision == "fp16"
    )
    step = 0
    if payload is not None:
        load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=schedule,
            expected_model_config=model_config,
            expected_data_fingerprint=data.fingerprint,
            generators=streams,
            batcher=data,
        )
        state = payload["training_state"]
        _restore_numpy(state["numpy_rng"])
        scaler.load_state_dict(state["scaler"])
        if runtime.xm is not None:
            runtime.xm.set_rng_state(state["xla_rng"], runtime.device)
        # load_state_dict preserves capturable step placement but LR starts on CPU.
        optimizer.param_groups[0]["lr"] = lr
        step = payload["step"]
    (run_dir / "job.json").write_text(json.dumps(contract, indent=2) + "\n")

    def commit() -> None:
        runtime.sync()
        state = {
            "notebook_schema": SCHEMA,
            "identity": identity,
            "contract": contract,
            "tokens_seen": schedule.tokens_seen,
            "numpy_rng": _numpy_state(),
            "scaler": scaler.state_dict(),
        }
        if runtime.xm is not None:
            state["xla_rng"] = runtime.xm.get_rng_state(runtime.device)
        save_checkpoint(
            checkpoint,
            model=model,
            step=step,
            model_config=model_config,
            train_config=settings,
            data_fingerprint=data.fingerprint,
            optimizer=optimizer,
            scheduler=schedule,
            generators=streams,
            batcher=data,
            training_state=state,
        )
        published = store.publish(
            run_dir, step=step, tokens=schedule.tokens_seen, job=run_dir / "job.json"
        )
        print(json.dumps(published, sort_keys=True), flush=True)

    if payload is None:
        commit()  # step zero is recoverable before the first update
    limit = total_steps if stop_after_step is None else stop_after_step
    if step >= limit:
        return {
            "engine": ENGINE,
            "identity": identity,
            "step": step,
            "tokens_seen": schedule.tokens_seen,
            "status": "already-at-boundary",
        }
    model.train()
    while step < limit:
        started = time.perf_counter()
        remaining = min(config.tokens_per_step, config.max_tokens - schedule.tokens_seen)
        optimizer.zero_grad(set_to_none=True)
        lr_value = learning_rate_at_tokens(schedule.tokens_seen + remaining, settings)
        lr.fill_(lr_value)
        loss_sum = torch.zeros((), device=runtime.device)
        consumed = 0
        for _ in range(config.grad_accum_steps):
            inputs, targets = data.get_batch(
                "train",
                batch_size=config.batch_size,
                context_length=config.context_length,
                device=runtime.device,
            )
            active = min(targets.numel(), remaining - consumed)
            # Static shapes on XLA, exact final token budget without dropping a microbatch.
            mask = torch.arange(targets.numel(), device=runtime.device).reshape_as(targets) < active
            targets = torch.where(mask, targets, -100)
            with runtime.autocast():
                logits = model(inputs)
                loss = (
                    torch.nn.functional.cross_entropy(
                        logits.flatten(0, 1).float(), targets.flatten(), reduction="sum"
                    )
                    / remaining
                )
            scaler.scale(loss).backward()
            loss_sum = loss_sum + loss.detach()
            consumed += active
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
        runtime.sync()
        if not torch.isfinite(norm).item() or not torch.isfinite(loss_sum).item():
            raise RuntimeError("non-finite training state; resume only the last committed boundary")
        scaler.step(optimizer)
        scaler.update()
        runtime.sync()
        step += 1
        schedule.tokens_seen += remaining
        # Reserved stream is exercised to verify recovery; no diffusion corruption exists yet.
        noise_probe = torch.rand((), generator=streams["noise"]).item()
        metric = {
            "event": "train",
            "engine": ENGINE,
            "step": step,
            "tokens_seen": schedule.tokens_seen,
            "loss": loss_sum.item(),
            "lr": lr_value,
            "gradient_norm": norm.item(),
            "noise_rng_probe": noise_probe,
            "python_rng_probe": random.random(),
            "numpy_rng_probe": float(np.random.random()),
            "seconds": time.perf_counter() - started,
        }
        metric["tokens_per_second"] = remaining / metric["seconds"]
        _append_metric(metrics_path, metric)
        if step % config.eval_interval == 0 or step == total_steps:
            model.eval()
            losses = []
            with torch.no_grad(), runtime.autocast():
                for _ in range(config.eval_batches):
                    inputs, targets = data.get_batch(
                        "validation",
                        batch_size=config.batch_size,
                        context_length=config.context_length,
                        generator=streams["validation"],
                        device=runtime.device,
                    )
                    losses.append(next_token_loss(model(inputs).float(), targets))
                validation_loss = torch.stack(losses).mean()
            runtime.sync()
            _append_metric(
                metrics_path,
                {
                    "event": "validation",
                    "step": step,
                    "tokens_seen": schedule.tokens_seen,
                    "causal_validation_loss": validation_loss.item(),
                },
            )
            model.train()
        if step % config.checkpoint_interval == 0 or step == limit:
            commit()  # synchronous publication; stop rather than run ahead of failed Drive writes
    return {
        "engine": ENGINE,
        "identity": identity,
        "step": step,
        "tokens_seen": schedule.tokens_seen,
        "status": "complete" if step == total_steps else "paused",
        "checkpoint": str(checkpoint),
        "backup": str(store.root),
        "memory": runtime.memory(),
        "counters": runtime.counters(),
    }


def evaluate_probe(
    config: NotebookConfig,
    *,
    data_dir: Path,
    checkpoint: Path,
    batches: int = 2,
    export_dir: Path | None = None,
) -> dict[str, Any]:
    """Portable CPU qualification report, explicitly causal rather than diffusion metrics."""
    if type(batches) is not int or not 1 <= batches <= 100:
        raise ValueError("evaluation batches must be between 1 and 100")
    data = PreparedTokenData(data_dir)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = payload.get("training_state", {}).get("contract", {})
    # Evaluation is portable: use recorded runtime/code identity, not a new optimizer contract.
    if (
        contract.get("config") != asdict(config)
        or contract.get("data_fingerprint") != data.fingerprint
        or (contract.get("tokenizer_sha256") != data.metadata["tokenizer"]["sha256"])
    ):
        raise ValueError(
            "evaluation configuration/data/tokenizer differs from the recorded experiment"
        )
    _validate_payload(payload, contract, contract_identity(contract))
    model = prepare_tied_model(config.model_config(data.tokenizer.vocab_size), torch.device("cpu"))
    load_checkpoint(
        checkpoint,
        model=model,
        expected_model_config=model.config,
        expected_data_fingerprint=data.fingerprint,
        restore_rng=False,
    )
    model.eval()
    generator = torch.Generator().manual_seed(141)
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = data.get_batch(
                "validation",
                batch_size=config.batch_size,
                context_length=config.context_length,
                generator=generator,
            )
            logits = model(inputs)
            losses.append(next_token_loss(logits, targets).item())
    sample = generate(
        model,
        data.tokenizer,
        "The kiwi",
        max_new_tokens=8,
        temperature=0.8,
        top_k=40,
        seed=42,
        cache="auto",
    )
    health = model_health_report(model, inputs, targets)
    parity = cached_generation_parity_report(model, inputs, rtol=1e-5, atol=1e-5)
    report = {
        "engine": ENGINE,
        "metric_kind": "tiny-V2-causal-qualification-only",
        "step": payload["step"],
        "tokens_seen": payload["training_state"]["tokens_seen"],
        "fixed_validation_loss": sum(losses) / len(losses),
        "parameters": sum(value.numel() for value in model.parameters()),
        "weights_finite": all(torch.isfinite(value).all().item() for value in model.parameters()),
        "sample": sample,
        "evaluation_backend": "cpu-fp32",
        "health": health,
        "cached_generation_parity": parity,
        "retrieval": "not meaningful for the synthetic M2 recovery probe; qualify in later phases",
    }
    if export_dir is not None:
        tokenizer = data_dir / data.metadata["tokenizer"]["file"]
        export_safetensors_bundle(
            checkpoint,
            export_dir,
            tokenizer_path=tokenizer,
            expected_data_fingerprint=data.fingerprint,
            expected_tokenizer_sha256=data.metadata["tokenizer"]["sha256"],
            variant=ENGINE,
            dtype="bf16",
        )
        report["export_dir"] = str(export_dir)
    return report
