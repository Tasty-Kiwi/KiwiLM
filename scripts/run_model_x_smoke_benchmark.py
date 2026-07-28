#!/usr/bin/env python3
"""Retrain Model B, the Transformer, and Model X under one smoke profile."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from run_transformer_smoke_benchmark import (
    DEFAULT_DATA_FINGERPRINT,
    DEFAULT_MODEL_B_PARAMETERS,
    DEFAULT_TRANSFORMER_PARAMETERS,
    _atomic_write_json,
    _benchmark_generation,
    _last_training_throughput,
    _require_empty_destination,
)

from kiwilm.comparison import compare_checkpoints
from kiwilm.config import CNNAttentionConfig, ModelXConfig, TransformerConfig
from kiwilm.data import PreparedTokenData
from kiwilm.inference import load_trained_model
from kiwilm.training import TrainConfig, choose_device, evaluate, train

DEFAULT_MODEL_X_PARAMETERS = 5_387_520


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/model-x-smoke"),
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
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16", "auto"),
        default="fp32",
    )
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--swiglu-dim", type=int, default=640)
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

    shared = {
        "vocab_size": data.tokenizer.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "dropout": 0.1,
        "tie_embeddings": True,
        "num_heads": args.attention_heads,
    }
    model_configs = {
        "model_b": CNNAttentionConfig(
            **shared,
            feedforward_dim=args.feedforward_dim,
        ),
        "transformer": TransformerConfig(
            **shared,
            num_layers=4,
            feedforward_dim=args.feedforward_dim,
        ),
        "model_x": ModelXConfig(
            **shared,
            swiglu_dim=args.swiglu_dim,
        ),
    }
    train_config = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=3e-4,
        min_lr=3e-5,
        warmup_steps=args.warmup_steps,
        batch_mode="packed",
        eval_mode="packed",
        precision=args.precision,
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
                precision=args.precision,
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
        labels=["Model B", "GPT-style Transformer", "Model X"],
    )
    model_b_parameters = int(model_results["model_b"]["parameter_count"])
    summary = {
        "benchmark": "model-b-vs-transformer-vs-model-x-smoke",
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
            "grad_accum_steps": args.grad_accum_steps,
            "context_length": args.context_length,
            "training_targets_per_model": (
                args.max_steps
                * args.batch_size
                * args.grad_accum_steps
                * args.context_length
            ),
            "seed": args.seed,
            "precision": args.precision,
            "batch_mode": "packed",
        },
        "parameter_deltas_vs_model_b": {
            name: {
                "absolute": int(result["parameter_count"]) - model_b_parameters,
                "percent": (
                    100.0
                    * (int(result["parameter_count"]) - model_b_parameters)
                    / model_b_parameters
                ),
            }
            for name, result in model_results.items()
            if name != "model_b"
        },
        "models": model_results,
        "comparison": comparison,
    }
    _atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _assert_default_parameter_counts(
    args: argparse.Namespace,
    results: dict[str, dict[str, Any]],
) -> None:
    if args.expected_data_fingerprint != DEFAULT_DATA_FINGERPRINT or (
        args.d_model,
        args.attention_heads,
        args.feedforward_dim,
        args.swiglu_dim,
    ) != (256, 8, 1024, 640):
        return
    actual = {name: result["parameter_count"] for name, result in results.items()}
    expected = {
        "model_b": DEFAULT_MODEL_B_PARAMETERS,
        "transformer": DEFAULT_TRANSFORMER_PARAMETERS,
        "model_x": DEFAULT_MODEL_X_PARAMETERS,
    }
    if actual != expected:
        raise RuntimeError(
            f"default benchmark parameter counts changed: expected {expected}, got {actual}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
