#!/usr/bin/env python3
"""Retrain Model B and the Transformer baseline under one smoke profile."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from kiwilm.comparison import compare_checkpoints
from kiwilm.config import CNNAttentionConfig, TransformerConfig
from kiwilm.data import PreparedTokenData
from kiwilm.generation import generate_tokens
from kiwilm.inference import load_trained_model
from kiwilm.training import TrainConfig, choose_device, evaluate, train

DEFAULT_DATA_FINGERPRINT = "a01d7037441e4cc0f1fe48615d384761c47cea506101708bbe42a0cee8ec7418"
DEFAULT_MODEL_B_PARAMETERS = 5_261_056
DEFAULT_TRANSFORMER_PARAMETERS = 5_264_896


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/transformer-smoke"),
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("eval/story-consistency-prompts.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--expected-data-fingerprint",
        default=DEFAULT_DATA_FINGERPRINT,
    )
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--post-eval-batches", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--sample-tokens", type=int, default=64)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--generation-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_empty_destination(args.output_dir)
    data = PreparedTokenData(args.data_dir, seed=args.seed)
    if data.fingerprint != args.expected_data_fingerprint:
        raise ValueError(
            "prepared dataset fingerprint differs from the smoke benchmark: "
            f"expected {args.expected_data_fingerprint}, got {data.fingerprint}"
        )
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shared_model = {
        "vocab_size": data.tokenizer.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "dropout": 0.1,
        "tie_embeddings": True,
        "num_heads": args.attention_heads,
        "feedforward_dim": args.feedforward_dim,
    }
    model_configs = {
        "model_b": CNNAttentionConfig(**shared_model),
        "transformer": TransformerConfig(
            **shared_model,
            num_layers=4,
        ),
    }
    train_config = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=1,
        lr=3e-4,
        min_lr=3e-5,
        warmup_steps=args.warmup_steps,
        batch_mode="packed",
        eval_mode="packed",
        precision="fp32",
        weight_decay=0.1,
        beta2=0.95,
        grad_clip=1.0,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        sample_prompt="Once upon a time",
        sample_tokens=args.sample_tokens,
        seed=args.seed,
    )

    model_results: dict[str, dict[str, Any]] = {}
    checkpoints: list[Path] = []
    for name, model_config in model_configs.items():
        run_dir = args.output_dir / name.replace("_", "-")
        started = time.perf_counter()
        training_summary = train(
            model_config,
            PreparedTokenData(args.data_dir, seed=args.seed),
            run_dir,
            train_config,
            device=device,
        )
        training_elapsed = time.perf_counter() - started
        checkpoint_value = training_summary["best_checkpoint"]
        if checkpoint_value is None:
            raise RuntimeError(f"{name} training did not produce a best checkpoint")
        checkpoint = Path(checkpoint_value)
        checkpoints.append(checkpoint)

        model, loaded_config = load_trained_model(
            checkpoint,
            data_fingerprint=data.fingerprint,
            device=device,
        )
        post_evaluation = {
            mode: evaluate(
                model,
                data,
                batch_size=args.batch_size,
                context_length=loaded_config.context_length,
                num_batches=args.post_eval_batches,
                device=device,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
                batch_mode=mode,
                precision="fp32",
                seed=args.seed,
            )
            for mode in ("packed", "story")
        }
        generation = {
            cache: _benchmark_generation(
                model,
                data.tokenizer,
                device=device,
                context_length=loaded_config.context_length,
                max_new_tokens=args.generation_tokens,
                repeats=args.generation_repeats,
                cache=cache,
            )
            for cache in ("auto", "off")
        }
        tokens_seen = int(training_summary["tokens_seen"])
        model_results[name] = {
            "architecture": loaded_config.architecture,
            "parameter_count": training_summary["parameter_count"],
            "training_elapsed_seconds": training_elapsed,
            "end_to_end_valid_tokens_per_second": tokens_seen / training_elapsed,
            "final_logged_throughput": _last_training_throughput(
                Path(training_summary["metrics_path"])
            ),
            "training": training_summary,
            "post_evaluation": post_evaluation,
            "generation": generation,
        }

    _assert_default_parameter_counts(args, model_results)
    comparison = compare_checkpoints(
        checkpoints,
        data=data,
        suite_path=args.suite,
        output_dir=args.output_dir / "comparison",
        device=device,
        labels=["Model B", "GPT-style Transformer"],
    )
    model_b_parameters = int(model_results["model_b"]["parameter_count"])
    transformer_parameters = int(model_results["transformer"]["parameter_count"])
    summary = {
        "benchmark": "model-b-vs-transformer-smoke",
        "data_dir": str(args.data_dir.resolve()),
        "data_fingerprint": data.fingerprint,
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "settings": {
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "training_targets_per_model": args.max_steps * args.batch_size * args.context_length,
            "seed": args.seed,
            "precision": "fp32",
            "batch_mode": "packed",
        },
        "parameter_delta": {
            "absolute": transformer_parameters - model_b_parameters,
            "percent_of_model_b": (
                100.0 * (transformer_parameters - model_b_parameters) / model_b_parameters
            ),
        },
        "models": model_results,
        "comparison": comparison,
    }
    _atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _require_empty_destination(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"benchmark output directory is not empty: {path}; "
            "choose a new directory or move the existing artifacts"
        )


def _benchmark_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    device: torch.device,
    context_length: int,
    max_new_tokens: int,
    repeats: int,
    cache: str,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("generation-tokens must be positive")
    if repeats <= 0:
        raise ValueError("generation-repeats must be positive")
    prompt_ids = tokenizer.encode(
        "Once upon a time",
        add_bos=True,
        add_eos=False,
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generate_tokens(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        temperature=0,
        eos_id=None,
        cache=cache,
    )
    _synchronize(device)
    elapsed_values: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        generate_tokens(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            context_length=context_length,
            temperature=0,
            eos_id=None,
            cache=cache,
        )
        _synchronize(device)
        elapsed_values.append(time.perf_counter() - started)
    median_elapsed = statistics.median(elapsed_values)
    return {
        "cache": cache,
        "forced_new_tokens": max_new_tokens,
        "repeats": repeats,
        "median_elapsed_seconds": median_elapsed,
        "median_tokens_per_second": max_new_tokens / median_elapsed,
        "elapsed_seconds": elapsed_values,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _last_training_throughput(metrics_path: Path) -> dict[str, float] | None:
    latest: dict[str, float] | None = None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") != "train":
            continue
        latest = {
            key: float(row[key])
            for key in (
                "valid_tokens_per_second",
                "model_tokens_per_second",
                "padding_fraction",
            )
            if key in row
        }
    return latest


def _assert_default_parameter_counts(
    args: argparse.Namespace,
    results: dict[str, dict[str, Any]],
) -> None:
    if args.expected_data_fingerprint != DEFAULT_DATA_FINGERPRINT or (
        args.d_model,
        args.attention_heads,
        args.feedforward_dim,
    ) != (256, 8, 1024):
        return
    actual = {
        "model_b": results["model_b"]["parameter_count"],
        "transformer": results["transformer"]["parameter_count"],
    }
    expected = {
        "model_b": DEFAULT_MODEL_B_PARAMETERS,
        "transformer": DEFAULT_TRANSFORMER_PARAMETERS,
    }
    if actual != expected:
        raise RuntimeError(
            f"default benchmark parameter counts changed: expected {expected}, got {actual}"
        )


def _atomic_write_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
