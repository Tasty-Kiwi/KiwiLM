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


def probe(
    data: PreparedTokenData, output: Path, *, config: KiwiLM2Config,
    runtime: Runtime, steps: int = 200, warmup_steps: int = 20,
    batch_size: int = 8, accumulation: int = 4, eval_batches: int = 5,
    resume: Path | None = None,
    max_tokens: int = 50_000_000, eval_interval: int = 500,
    checkpoint_interval: int = 500, artifact_dir: Path | None = None,
    final_diagnostics: bool = False, verify_checkpoint_reload: bool = False,
) -> dict[str, Any]:
    """Train a bounded prefix of a frozen 50M schedule using real packed data."""
    if not 0 < warmup_steps < steps or min(
        batch_size, accumulation, eval_batches, max_tokens, eval_interval, checkpoint_interval
    ) < 1:
        raise ValueError("require steps > warmup_steps > 0 and positive batch/evaluation sizes")
    if config.architecture != "kiwilm2" or config.dropout != 0:
        raise ValueError("hardware smoke supports Dense with zero dropout only")
    settings = TrainConfig(
        max_steps=math.ceil(max_tokens / (batch_size * accumulation * config.context_length)) + 100,
        max_tokens=max_tokens, warmup_tokens=min(1_000_000, max_tokens - 1),
        batch_size=batch_size, grad_accum_steps=accumulation, precision=runtime.precision,
        optimizer="muon", muon_lr=0.01, seed=42, eval_batches=eval_batches,
        eval_interval=eval_interval, checkpoint_interval=checkpoint_interval,
    )
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.pt").exists() and resume is None:
        raise ValueError("output already has a checkpoint; use --resume or a new directory")
    torch.manual_seed(settings.seed)
    model = KiwiLM2LM(config).to(runtime.device)
    muon, auxiliary = split_muon_parameters(model)
    optimizer = TensorMuon(
        muon, auxiliary, muon_lr=settings.muon_lr, adamw_lr=settings.lr,
        weight_decay=settings.weight_decay, beta2=settings.beta2,
    )
    generator = torch.Generator(device="cpu").manual_seed(settings.seed)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=runtime.device.type == "cuda" and runtime.precision == "fp16",
    )
    completed, tokens = 0, 0
    contract = {"engine": "single-device-smoke-v1", "device": runtime.device.type,
                "train_config": settings.to_dict()}
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("training_state", {}).get("smoke_contract") != contract:
            raise ValueError("resume must be a matching hardware smoke, not the ongoing GPU run")
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
    if runtime.xm is not None:
        runtime.xm.set_rng_state(settings.seed)
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

    def checkpoint() -> None:
        nonlocal checkpoint_reload
        save_checkpoint(
            output / "latest.pt", model=model, optimizer=optimizer, step=completed,
            model_config=config, train_config=settings, data_fingerprint=data.fingerprint,
            generators={"train": generator},
            metrics={"validation_loss": validation_loss},
            training_state={"tokens_seen": tokens, "smoke_contract": contract,
                            "scaler_state": scaler.state_dict()},
        )
        if verify_checkpoint_reload and checkpoint_reload is None:
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
            assert restored["step"] == completed
            assert restored["training_state"]["tokens_seen"] == tokens
            checkpoint_reload = {
                "passed": True, "device": str(runtime.device), "step": completed,
                "scope": "same-process model/optimizer/data-generator reload; not VM restart",
            }
        if artifact_dir is not None:
            create_colab_artifacts(
                {f.name: f for f in output.iterdir() if f.is_file() and f.name != "worker.log"},
                artifact_dir, chunk_size=4 * 1024 * 1024,
            )

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
        "status": "smoke-complete" if tokens == settings.max_tokens else "probe-complete",
        "not_a_1b_promotion": True,
        "device": str(runtime.device), "precision": runtime.precision,
        "torch": torch.__version__,
        "torch_xla": runtime.xla.__version__ if runtime.xla is not None else None,
        "data_fingerprint": data.fingerprint,
        "tokenizer_sha256": data.metadata["tokenizer"]["sha256"],
        "model_config": config.to_dict(), "train_config": settings.to_dict(),
        "initial_step": initial_step, "step": completed, "tokens_seen": tokens,
        "first_step_seconds": durations[0] if durations else None,
        "warmup_steps": warmup_steps, "training_seconds": training_seconds,
        "steady_tokens_per_second": steady_tokens / steady_seconds if steady_seconds else None,
        "steady_tokens": steady_tokens, "steady_seconds": steady_seconds,
        "session_tokens": tokens - initial_tokens,
        "training_and_periodic_io_seconds": training_seconds,
        "validation_loss": validation_loss, "perplexity": math.exp(min(validation_loss, 80)),
        "memory": memory, "xla_after_warmup": after_warmup, "xla_after_training": training_counters,
        "checkpoint_reload": checkpoint_reload,
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
                        help="optional bounded probe; default runs to exactly 50M tokens")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    # Reuse the real frozen 50M dataset; no synthetic benchmark or new tokenizer.
    build_colab_job(args.data_dir, phase="smoke", architecture="kiwilm2")
    data = PreparedTokenData(args.data_dir)
    if data.metadata["config"].get("seed") != 42:
        raise ValueError("hardware probe requires the frozen seed-42 smoke data")
    runtime = Runtime(args.device, args.precision)
    report = probe(
        data, args.output_dir, config=KiwiLM2Config(vocab_size=data.tokenizer.vocab_size),
        runtime=runtime,
        steps=(math.ceil(50_000_000 / (
            args.batch_size * args.grad_accum_steps * 512
        )) + 100) if args.steps is None else args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size, accumulation=args.grad_accum_steps,
        eval_batches=args.eval_batches, resume=args.resume,
        eval_interval=args.eval_interval, checkpoint_interval=args.checkpoint_interval,
        artifact_dir=args.artifact_dir, final_diagnostics=args.steps is None,
        verify_checkpoint_reload=args.steps is None,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
