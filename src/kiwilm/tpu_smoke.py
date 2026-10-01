"""Experimental single-chip XLA throughput/convergence probe, not the final trainer.

The standard GPU trainer and its checkpoint namespaces remain unchanged. CPU and
CUDA backends are provided for tests and matched hardware controls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from kiwilm.checkpoint import load_checkpoint, save_checkpoint
from kiwilm.colab_artifacts import create_colab_artifacts
from kiwilm.colab_kiwilm2 import build_colab_job
from kiwilm.config import KiwiLM2Config
from kiwilm.data import PreparedTokenData
from kiwilm.models import KiwiLM2LM
from kiwilm.optim import MuonWithAuxAdamW, split_muon_parameters, zeroth_power_via_newton_schulz
from kiwilm.tpu_checkpoint import DriveCheckpointStore, compare_continuation, contract_digest
from kiwilm.training import TrainConfig, learning_rate_at_tokens


class TensorMuon(MuonWithAuxAdamW):
    """Same Muon/auxiliary AdamW math, with changing scalars kept as device inputs.

    Python learning rates and Adam bias corrections otherwise become new XLA
    graph constants every step. Never use this class for an existing GPU run.
    """

    def set_learning_rate(self, learning_rate: float) -> None:
        for group in self.param_groups:
            group["lr"] = torch.tensor(
                learning_rate * group["lr_multiplier"], dtype=torch.float32,
            ).to(group["params"][0].device)

    def _step_muon(self, group: dict[str, Any]) -> None:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(gradient)
            buffer = state["momentum_buffer"]
            buffer.mul_(group["momentum"]).add_(gradient)
            update = gradient.add(buffer, alpha=group["momentum"]) if group["nesterov"] else buffer
            update = zeroth_power_via_newton_schulz(update, steps=group["ns_steps"])
            scale = math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
            parameter.add_(update * (-group["lr"] * scale))

    def _step_adamw(self, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            state = self.state[parameter]
            if not state:
                state.update(
                    step=torch.zeros((), device=parameter.device),
                    exp_avg=torch.zeros_like(gradient), exp_avg_sq=torch.zeros_like(gradient),
                )
            state["step"].add_(1)
            avg, square = state["exp_avg"], state["exp_avg_sq"]
            avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            square.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            denominator = square.sqrt() / (1.0 - beta2 ** state["step"]).sqrt() + group["eps"]
            step_size = group["lr"] / (1.0 - beta1 ** state["step"])
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
            parameter.add_(-step_size * avg / denominator)


class Runtime:
    def __init__(self, device: str, precision: str) -> None:
        self.xla = self.xm = self.metrics = None
        if device == "xla":
            if precision != "bf16":
                raise ValueError("TPU smoke requires bf16")
            os.environ.setdefault("PJRT_DEVICE", "TPU")
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.debug.metrics as metrics
            import torch_xla.runtime as xr

            if xr.device_type() != "TPU":
                raise RuntimeError("XLA must target a real TPU, not CPU fallback")
            self.device = torch_xla.device()
            if xr.global_runtime_device_count() != 1:
                raise RuntimeError("this smoke supports exactly one TPU chip")
            self.xla, self.xm, self.metrics = torch_xla, xm, metrics
        else:
            self.device = torch.device(device)
            if device == "cpu" and precision != "fp32":
                raise ValueError("CPU tests require fp32")
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
        self.precision = precision

    def sync(self) -> None:
        if self.xla is not None:
            self.xla.sync(wait=True)
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def autocast(self):
        return torch.autocast(
            self.device.type,
            dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16,
            enabled=self.precision != "fp32",
        )

    def counters(self) -> dict[str, Any]:
        if self.metrics is None:
            return {}
        compilation = self.metrics.metric_data("CompileTime")
        return {
            "compile_count": compilation[0] if compilation else 0,
            "cpu_fallbacks": {
                name: self.metrics.counter_value(name)
                for name in self.metrics.counter_names() or [] if name.startswith("aten::")
            },
        }

    def memory(self) -> dict[str, Any]:
        if self.xm is not None:
            return dict(self.xm.get_memory_info(self.device))
        if self.device.type == "cuda":
            return {"peak_bytes_used": torch.cuda.max_memory_allocated(self.device)}
        return {}


def prepare_tied_model(config: KiwiLM2Config, device: torch.device) -> KiwiLM2LM:
    """XLA transfer replaces shared Parameters: re-tie before optimizer creation."""
    if not config.tie_embeddings:
        raise ValueError("hardware smoke requires tied embeddings")
    model = KiwiLM2LM(config)
    expected_parameters = sum(parameter.numel() for parameter in model.parameters())
    model.to(device)
    model.lm_head.weight = model.token_embedding.weight
    if model.lm_head.weight is not model.token_embedding.weight or sum(
        parameter.numel() for parameter in model.parameters()
    ) != expected_parameters:
        raise RuntimeError("device transfer changed the tied model parameterization")
    return model


def validate_tied_checkpoint(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    if not payload["model_config"].get("tie_embeddings") or not torch.equal(
        state["token_embedding.weight"], state["lm_head.weight"],
    ):
        raise ValueError("TPU checkpoint has unequal tied embedding/head weights; start fresh")


def portable_loss_report(
    model: KiwiLM2LM, data: PreparedTokenData, path: Path, runtime: Runtime,
    *, batch_size: int, batches: int,
) -> dict[str, Any]:
    """Compare the saved ordinary CPU reconstruction with fixed runtime batches."""
    validate_tied_checkpoint(path)
    cpu_model = KiwiLM2LM(model.config)
    load_checkpoint(
        path, model=cpu_model, expected_model_config=model.config,
        expected_data_fingerprint=data.fingerprint, restore_rng=False,
    )
    cpu_model.eval()
    generator = torch.Generator().manual_seed(43)
    losses: dict[str, list[float]] = {"runtime": [], "cpu": []}
    was_training = model.training
    model.eval()
    logit_relative_rms = None
    try:
        for index in range(batches):
            inputs, targets = data.get_batch(
                "validation", batch_size=batch_size,
                context_length=model.config.context_length, generator=generator,
            )
            with torch.no_grad(), runtime.autocast():
                logits = model(inputs.to(runtime.device))
                loss = F.cross_entropy(
                    logits.float().reshape(-1, model.config.vocab_size),
                    targets.to(runtime.device).reshape(-1),
                )
            runtime.sync()
            losses["runtime"].append(float(loss.cpu()))
            with torch.no_grad():
                cpu_logits = cpu_model(inputs)
                cpu_loss = F.cross_entropy(
                    cpu_logits.float().reshape(-1, model.config.vocab_size), targets.reshape(-1),
                )
            losses["cpu"].append(float(cpu_loss))
            if index == 0:
                runtime_logits = logits.cpu().float()
                logit_relative_rms = float(
                    (runtime_logits - cpu_logits).square().mean().sqrt()
                    / cpu_logits.square().mean().sqrt().clamp_min(1e-8)
                )
    finally:
        model.train(was_training)
    runtime_loss = sum(losses["runtime"]) / batches
    cpu_loss = sum(losses["cpu"]) / batches
    difference = abs(runtime_loss - cpu_loss)
    report = {
        "passed": math.isfinite(difference) and difference <= 0.02
        and math.isfinite(logit_relative_rms) and logit_relative_rms <= 0.02,
        "runtime_device": str(runtime.device), "runtime_precision": runtime.precision,
        "cpu_precision": "fp32", "seed": 43, "batches": batches,
        "batch_size": batch_size, "context_length": model.config.context_length,
        "runtime_loss": runtime_loss, "cpu_loss": cpu_loss,
        "absolute_loss_difference": difference, "loss_tolerance": 0.02,
        "first_batch_logit_relative_rms": logit_relative_rms, "logit_rms_tolerance": 0.02,
        "checkpoint_weights_equal": True,
    }
    if not report["passed"]:
        raise RuntimeError(f"TPU checkpoint portability failed: {report}")
    return report


def probe_settings(
    config: KiwiLM2Config, *, precision: str, max_tokens: int = 50_000_000,
    batch_size: int = 8, accumulation: int = 4, eval_batches: int = 50,
    eval_interval: int = 500, checkpoint_interval: int = 500,
    warmup_tokens: int | None = None,
) -> TrainConfig:
    return TrainConfig(
        max_steps=math.ceil(max_tokens / (batch_size * accumulation * config.context_length)) + 100,
        max_tokens=max_tokens, warmup_tokens=(min(1_000_000, max_tokens - 1)
                                           if warmup_tokens is None else warmup_tokens),
        batch_size=batch_size, grad_accum_steps=accumulation, precision=precision,
        optimizer="muon", muon_lr=0.01, seed=42, eval_batches=eval_batches,
        eval_interval=eval_interval, checkpoint_interval=checkpoint_interval,
    )


def probe_contract(settings: TrainConfig, device: str) -> dict:
    engine = ("single-device-final-v1-tied" if settings.max_tokens == 1_000_000_000
              else "single-device-smoke-v2-tied")
    return {"engine": engine, "device": device,
            "train_config": settings.to_dict()}


def probe(
    data: PreparedTokenData, output: Path, *, config: KiwiLM2Config,
    runtime: Runtime, steps: int = 200, warmup_steps: int = 20,
    batch_size: int = 8, accumulation: int = 4, eval_batches: int = 5,
    resume: Path | None = None,
    max_tokens: int = 50_000_000, eval_interval: int = 500,
    checkpoint_interval: int = 500, artifact_dir: Path | None = None,
    final_diagnostics: bool = False, verify_checkpoint_reload: bool = False,
    drive_store: DriveCheckpointStore | None = None, job_path: Path | None = None,
    vm_id: str | None = None, require_new_vm: bool = False,
    warmup_tokens: int | None = None, periodic_artifacts: bool = True,
) -> dict[str, Any]:
    """Train a bounded prefix of an immutable token schedule using real packed data."""
    if not 0 < warmup_steps < steps or min(
        batch_size, accumulation, eval_batches, max_tokens, eval_interval, checkpoint_interval
    ) < 1:
        raise ValueError("require steps > warmup_steps > 0 and positive batch/evaluation sizes")
    if config.architecture != "kiwilm2" or config.dropout != 0:
        raise ValueError("hardware smoke supports Dense with zero dropout only")
    settings = probe_settings(
        config, precision=runtime.precision, max_tokens=max_tokens,
        batch_size=batch_size, accumulation=accumulation, eval_batches=eval_batches,
        eval_interval=eval_interval, checkpoint_interval=checkpoint_interval,
        warmup_tokens=warmup_tokens,
    )
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.pt").exists() and resume is None:
        raise ValueError("output already has a checkpoint; use --resume or a new directory")
    torch.manual_seed(settings.seed)
    model = prepare_tied_model(config, runtime.device)
    muon, auxiliary = split_muon_parameters(model)
    if any(parameter is model.lm_head.weight for parameter in muon):
        raise RuntimeError("tied embedding/head must use auxiliary AdamW, not Muon")
    optimizer = TensorMuon(
        muon, auxiliary, muon_lr=settings.muon_lr, adamw_lr=settings.lr,
        weight_decay=settings.weight_decay, beta2=settings.beta2,
    )
    generator = torch.Generator(device="cpu").manual_seed(settings.seed)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=runtime.device.type == "cuda" and runtime.precision == "fp16",
    )
    completed, tokens = 0, 0
    contract = probe_contract(settings, runtime.device.type)
    resume_origin = None
    if require_new_vm and resume is None:
        raise ValueError("fresh-VM continuation requires a checkpoint; refusing fresh training")
    if resume is not None:
        validate_tied_checkpoint(resume)
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("training_state", {}).get("smoke_contract") != contract:
            raise ValueError(
                "resume requires a matching hardware smoke/final budget and schedule, not a GPU run"
            )
        if not saved.get("optimizer_state_dict") or saved.get("batcher_state", {}).get(
            "generators", {}
        ).get("train") is None:
            raise ValueError("resume checkpoint lacks optimizer or training data-generator state")
        source_vm = saved["training_state"].get("vm_id")
        if require_new_vm and (not vm_id or not source_vm or vm_id == source_vm):
            raise ValueError("continuation must run in a different freshly allocated VM")
        saved = load_checkpoint(
            resume, model=model, optimizer=optimizer, expected_model_config=config,
            expected_data_fingerprint=data.fingerprint, generators={"train": generator},
        )
        for parameter, state in optimizer.state.items():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(parameter.device)
        completed = saved["step"]
        tokens = saved["training_state"]["tokens_seen"]
        if saved["training_state"].get("scaler_state"):
            scaler.load_state_dict(saved["training_state"]["scaler_state"])
        resume_origin = {"step": completed, "tokens_seen": tokens, "source_vm_id": source_vm,
                         "optimizer_restored": True, "data_generator_restored": True}
    if runtime.xm is not None:
        runtime.xm.set_rng_state(
            saved["training_state"].get("xla_rng_state", settings.seed)
            if resume is not None else settings.seed
        )
    if drive_store is not None:
        expected_identity = contract_digest(contract, data.fingerprint, config.to_dict())
        if drive_store.identity != expected_identity:
            raise ValueError("Drive checkpoint store does not match the training contract")
        drive_store.check()  # Refuse to train without a reachable writable backup.
        drive_store.validate_progress(completed)
    metrics_path = output / "metrics.jsonl"
    if resume is not None and metrics_path.exists():
        retained = [
            line for line in metrics_path.read_text().splitlines()
            if json.loads(line).get("step", 0) <= completed
        ]
        metrics_path.write_text("".join(line + "\n" for line in retained))

    def validation() -> float:
        model.eval()
        evaluation_generator = torch.Generator(device="cpu").manual_seed(43)
        eval_losses = []
        for _ in range(eval_batches):
            inputs, targets = data.get_batch(
                "validation", batch_size=batch_size, context_length=config.context_length,
                generator=evaluation_generator, device=runtime.device,
            )
            with torch.no_grad(), runtime.autocast():
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, config.vocab_size), targets.reshape(-1)
                )
            runtime.sync()
            eval_losses.append(float(loss.cpu()))
        result = sum(eval_losses) / len(eval_losses)
        if not math.isfinite(result):
            raise FloatingPointError("non-finite validation loss")
        event = {"event": "validation", "step": completed, "tokens_seen": tokens,
                 "validation_loss": result, "perplexity": math.exp(min(result, 80))}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
        model.train()
        return result

    validation_loss = None
    checkpoint_reload = None
    first_portability = None
    last_drive_backup = None

    def checkpoint() -> None:
        nonlocal checkpoint_reload, first_portability, last_drive_backup
        if model.lm_head.weight is not model.token_embedding.weight:
            raise RuntimeError("embedding/head identity lost during training")
        save_checkpoint(
            output / "latest.pt", model=model, optimizer=optimizer, step=completed,
            model_config=config, train_config=settings, data_fingerprint=data.fingerprint,
            generators={"train": generator},
            metrics={"validation_loss": validation_loss},
            training_state={"tokens_seen": tokens, "smoke_contract": contract,
                            "scaler_state": scaler.state_dict(), "vm_id": vm_id,
                            "xla_rng_state": runtime.xm.get_rng_state()
                            if runtime.xm is not None else None},
        )
        # Publish before artifact packaging or diagnostics: loss of the VM after
        # this acknowledgement can recover this exact optimizer boundary.
        if drive_store is not None:
            last_drive_backup = drive_store.publish(
                output, step=completed, tokens=tokens, job=job_path,
            )
            print(json.dumps(last_drive_backup), flush=True)
        # Preserve downloadable evidence even if a portability gate fails.
        if artifact_dir is not None and periodic_artifacts:
            create_colab_artifacts(
                {f.name: f for f in output.iterdir() if f.is_file() and f.name != "worker.log"},
                artifact_dir, chunk_size=4 * 1024 * 1024,
            )
        if verify_checkpoint_reload and checkpoint_reload is None:
            first_portability = portable_loss_report(
                model, data, output / "latest.pt", runtime,
                batch_size=batch_size, batches=min(5, eval_batches),
            )
            inputs, _ = data.get_batch(
                "validation", batch_size=batch_size, context_length=config.context_length,
                generator=torch.Generator().manual_seed(43), device=runtime.device,
            )
            with torch.no_grad(), runtime.autocast():
                before = model(inputs).cpu()
            restored = load_checkpoint(
                output / "latest.pt", model=model, optimizer=optimizer,
                expected_model_config=config, expected_data_fingerprint=data.fingerprint,
                generators={"train": generator}, restore_rng=False,
            )
            for parameter, state in optimizer.state.items():
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[name] = value.to(parameter.device)
            runtime.sync()
            with torch.no_grad(), runtime.autocast():
                after = model(inputs).cpu()
            if model.lm_head.weight is not model.token_embedding.weight or not torch.equal(
                before, after,
            ):
                raise RuntimeError("same-process reload changed tied identity or logits")
            assert restored["step"] == completed
            assert restored["training_state"]["tokens_seen"] == tokens
            checkpoint_reload = {
                "passed": True, "device": str(runtime.device), "step": completed,
                "logits_equal": True, "weight_identity_preserved": True,
                "scope": "same-process model/optimizer/data-generator/logit reload; not VM restart",
            }
            print(json.dumps({"event": "checkpoint_portability", "step": completed,
                              **first_portability}), flush=True)

    model.train()
    runtime.sync()
    if runtime.metrics is not None:
        runtime.metrics.clear_all()
    rows, durations = [], []
    started = time.perf_counter()
    steady_tokens, steady_seconds = 0, 0.0
    after_warmup = {}
    initial_step = completed
    initial_tokens = tokens
    for local_step in range(steps):
        if tokens >= settings.max_tokens:
            break
        tick = time.perf_counter()
        remaining = min(
            batch_size * accumulation * config.context_length, settings.max_tokens - tokens
        )
        valid_this_step = remaining
        optimizer.set_learning_rate(learning_rate_at_tokens(tokens + remaining, settings))
        optimizer.zero_grad(set_to_none=True)
        nll = torch.zeros((), device=runtime.device)
        # Fixed shapes, including the final partially masked step. CPU counting
        # avoids .item()/nonzero synchronizations inside the XLA microbatch loop.
        for _ in range(accumulation):
            inputs, targets = data.get_batch(
                "train", batch_size=batch_size, context_length=config.context_length,
                generator=generator,
            )
            count = min(remaining, targets.numel())
            targets = targets.clone().contiguous()
            targets.view(-1)[count:] = -100
            remaining -= count
            inputs, targets = inputs.to(runtime.device), targets.to(runtime.device)
            with runtime.autocast():
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, config.vocab_size), targets.reshape(-1),
                    ignore_index=-100, reduction="sum",
                )
            scaler.scale(loss / valid_this_step).backward()
            nll = nll + loss.detach()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip, foreach=False)
        scaler.step(optimizer)
        scaler.update()
        runtime.sync()  # Timing must include actual TPU execution, not just graph tracing.
        loss_value, norm_value = float(nll.cpu()) / valid_this_step, float(norm.cpu())
        if not math.isfinite(loss_value) or not math.isfinite(norm_value) or norm_value == 0:
            raise FloatingPointError("non-finite loss/gradient or zero gradient in hardware smoke")
        elapsed = time.perf_counter() - tick
        completed += 1
        tokens += valid_this_step
        durations.append(elapsed)
        if local_step >= warmup_steps and valid_this_step == (
            batch_size * accumulation * config.context_length
        ):
            steady_tokens += valid_this_step
            steady_seconds += elapsed
        if local_step + 1 == warmup_steps:
            after_warmup = runtime.counters()
        row = {"event": "train", "step": completed, "tokens_seen": tokens,
               "train_loss": loss_value, "gradient_norm": norm_value,
               "step_seconds": elapsed, "tokens_per_second": valid_this_step / elapsed,
               "warmup": local_step < warmup_steps}
        rows.append(row)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if local_step == 0 or completed % 10 == 0:
            print(json.dumps(row), flush=True)
        if completed % eval_interval == 0:
            validation_loss = validation()
        if completed % checkpoint_interval == 0:
            checkpoint()
    training_seconds = time.perf_counter() - started
    training_counters = runtime.counters()
    validation_loss = validation()
    memory = runtime.memory()
    checkpoint()
    report = {
        "status": (("final-complete" if settings.max_tokens == 1_000_000_000
                    else "smoke-complete") if tokens == settings.max_tokens else "probe-complete"),
        "not_a_1b_promotion": settings.max_tokens != 1_000_000_000,
        "device": str(runtime.device), "precision": runtime.precision,
        "torch": torch.__version__,
        "torch_xla": runtime.xla.__version__ if runtime.xla is not None else None,
        "data_fingerprint": data.fingerprint,
        "tokenizer_sha256": data.metadata["tokenizer"]["sha256"],
        "model_config": config.to_dict(), "train_config": settings.to_dict(),
        "initial_step": initial_step, "step": completed, "tokens_seen": tokens,
        "resume_origin": resume_origin, "vm_id": vm_id, "drive_backup": last_drive_backup,
        "first_step_seconds": durations[0] if durations else None,
        "warmup_steps": warmup_steps, "training_seconds": training_seconds,
        "steady_tokens_per_second": steady_tokens / steady_seconds if steady_seconds else None,
        "steady_tokens": steady_tokens, "steady_seconds": steady_seconds,
        "session_tokens": tokens - initial_tokens,
        "training_and_periodic_io_seconds": training_seconds,
        "validation_loss": validation_loss, "perplexity": math.exp(min(validation_loss, 80)),
        "memory": memory, "xla_after_warmup": after_warmup, "xla_after_training": training_counters,
        "checkpoint_reload": checkpoint_reload,
        "weight_tying": {
            "identity_preserved": model.lm_head.weight is model.token_embedding.weight,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "head_in_muon": False,
        },
        "first_checkpoint_portability": first_portability,
        "cached_generation_parity": "not measured; required before production promotion",
    }
    if runtime.metrics is not None:
        (output / "xla-metrics.txt").write_text(runtime.metrics.metrics_report())
    # Hook-based health and variable-length decoding compile many additional XLA
    # graphs. Audit the portable saved weights on CPU and label that evidence.
    if final_diagnostics and tokens == settings.max_tokens:
        from kiwilm.diagnostics import (
            aggregate_health_reports,
            cached_generation_parity_report,
            model_health_report,
        )
        from kiwilm.generation import generate

        torch.set_num_threads(4)
        report["final_checkpoint_portability"] = portable_loss_report(
            model, data, output / "latest.pt", runtime,
            batch_size=batch_size, batches=eval_batches,
        )
        cpu_model = KiwiLM2LM(config)
        load_checkpoint(
            output / "latest.pt", model=cpu_model, expected_model_config=config,
            expected_data_fingerprint=data.fingerprint, restore_rng=False,
        )
        cpu_model.eval()
        reports = []
        for seed in (141, 142):
            health_generator = torch.Generator(device="cpu").manual_seed(seed)
            for _ in range(25):
                inputs, targets = data.get_batch(
                    "validation", batch_size=2, context_length=config.context_length,
                    generator=health_generator, device="cpu",
                )
                reports.append(model_health_report(cpu_model, inputs, targets))
        report["health"] = aggregate_health_reports(reports)
        report["cached_generation_parity"] = cached_generation_parity_report(cpu_model, inputs)
        report["diagnostic_device"] = "cpu"
        report["diagnostic_precision"] = "fp32"
        report["xla_cached_generation_parity"] = "not measured"
        report["generation"] = generate(
            cpu_model, data.tokenizer, "Once upon a time", max_new_tokens=64,
            temperature=0.8, top_k=40, seed=42, device="cpu", cache="auto",
        )
        (output / "health-batches.json").write_text(json.dumps(reports, indent=2) + "\n")
    report["session_seconds_before_final_packaging"] = time.perf_counter() - started
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if artifact_dir is not None:
        create_colab_artifacts(
            {f.name: f for f in output.iterdir() if f.is_file() and f.name != "worker.log"},
            artifact_dir, chunk_size=4 * 1024 * 1024,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("xla", "cuda", "cpu"), default="xla")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--steps", type=int, default=None,
                        help="optional bounded prefix; default runs to the frozen token budget")
    parser.add_argument("--phase", choices=("smoke", "final-1b"), default="smoke")
    parser.add_argument("--warmup-tokens", type=int)
    parser.add_argument("--final-artifacts-only", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--drive-backup-dir", type=Path)
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive"))
    parser.add_argument("--job", type=Path)
    parser.add_argument("--vm-id")
    parser.add_argument("--require-new-vm", action="store_true")
    parser.add_argument("--compare-reference", type=Path)
    args = parser.parse_args()
    frozen = build_colab_job(args.data_dir, phase=args.phase, architecture="kiwilm2")
    max_tokens = frozen["max_tokens"]
    warmup_tokens = args.warmup_tokens
    if args.phase == "final-1b":
        if args.device != "xla" or args.precision != "bf16" or args.drive_backup_dir is None:
            raise ValueError("1B TPU training requires XLA BF16 and verified Drive backups")
        if (args.batch_size, args.grad_accum_steps, args.eval_batches,
            args.eval_interval, args.checkpoint_interval) != (8, 4, 200, 500, 500):
            raise ValueError("1B TPU training requires the frozen batch/evaluation/save settings")
        if warmup_tokens not in {None, 20_000_000}:
            raise ValueError("1B TPU training requires 20M warmup tokens")
        warmup_tokens = 20_000_000
    data = PreparedTokenData(args.data_dir)
    if data.metadata["config"].get("seed") != 42:
        raise ValueError("hardware probe requires the frozen seed-42 smoke data")
    runtime = Runtime(args.device, args.precision)
    config = KiwiLM2Config(vocab_size=data.tokenizer.vocab_size)
    store = None
    if args.drive_backup_dir is not None:
        settings = probe_settings(
            config, precision=args.precision, batch_size=args.batch_size,
            accumulation=args.grad_accum_steps, eval_batches=args.eval_batches,
            eval_interval=args.eval_interval, checkpoint_interval=args.checkpoint_interval,
            max_tokens=max_tokens, warmup_tokens=warmup_tokens,
        )
        store = DriveCheckpointStore(
            args.drive_backup_dir, storage_root=args.drive_root,
            identity=contract_digest(probe_contract(settings, runtime.device.type),
                                     data.fingerprint, config.to_dict()),
        )
    report = probe(
        data, args.output_dir, config=config,
        runtime=runtime,
        steps=(math.ceil(max_tokens / (
            args.batch_size * args.grad_accum_steps * 512
        )) + 100) if args.steps is None else args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size, accumulation=args.grad_accum_steps,
        eval_batches=args.eval_batches, resume=args.resume,
        eval_interval=args.eval_interval, checkpoint_interval=args.checkpoint_interval,
        artifact_dir=args.artifact_dir, final_diagnostics=args.steps is None,
        verify_checkpoint_reload=args.steps is None,
        drive_store=store, job_path=args.job, vm_id=args.vm_id,
        require_new_vm=args.require_new_vm,
        max_tokens=max_tokens, warmup_tokens=warmup_tokens,
        periodic_artifacts=not args.final_artifacts_only,
    )
    if args.compare_reference is not None:
        result = compare_continuation(args.compare_reference, args.output_dir / "latest.pt")
        report["continuation_test"] = result
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": "continuation_test", **result}), flush=True)
        if args.artifact_dir is not None:
            create_colab_artifacts(
                {f.name: f for f in args.output_dir.iterdir()
                 if f.is_file() and f.name != "worker.log"},
                args.artifact_dir, chunk_size=4 * 1024 * 1024,
            )
        if not result["passed"]:
            raise RuntimeError("fresh-VM continuation differs from uninterrupted reference")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
