#!/usr/bin/env python3
"""Train KiwiLM-SAN and compare it with saved Model X and Model Y smokes."""

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
    _atomic_write_json,
    _benchmark_generation,
    _last_training_throughput,
    _require_empty_destination,
)

from kiwilm.comparison import compare_checkpoints, load_prompt_suite
from kiwilm.config import KiwiLMSANConfig
from kiwilm.data import PreparedTokenData
from kiwilm.inference import load_trained_model
from kiwilm.retrieval import (
    build_retrieval_suite,
    evaluate_retrieval_model,
    write_retrieval_artifacts,
)
from kiwilm.training import TrainConfig, choose_device, evaluate, train

DEFAULT_SAN_PARAMETERS = 5_260_560
DEFAULT_MODEL_X_CHECKPOINT = Path("runs/benchmarks/model-xyz-smoke/model-x/best.pt")
DEFAULT_MODEL_Y_CHECKPOINT = Path("runs/benchmarks/model-xyz-smoke/model-y/best.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/kiwilm-san-smoke"),
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("eval/story-consistency-prompts.json"),
    )
    parser.add_argument(
        "--model-x-checkpoint",
        type=Path,
        default=DEFAULT_MODEL_X_CHECKPOINT,
    )
    parser.add_argument(
        "--model-y-checkpoint",
        type=Path,
        default=DEFAULT_MODEL_Y_CHECKPOINT,
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
    parser.add_argument("--san-layers", type=int, default=16)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--rope-base", type=float, default=10_000.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--post-eval-batches", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--sample-tokens", type=int, default=64)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--generation-repeats", type=int, default=3)
    parser.add_argument("--retrieval-batch-size", type=int, default=32)
    parser.add_argument("--retrieval-pairs-per-distance", type=int, default=32)
    parser.add_argument(
        "--retrieval-distances",
        type=_parse_distances,
        default=(32, 64, 128, 192),
        metavar="N[,N...]",
    )
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

    baseline_specs = (
        ("model_x", "Model X", args.model_x_checkpoint, "model_x"),
        ("model_y", "Model Y", args.model_y_checkpoint, "model_y"),
    )
    baseline_metadata = {
        key: _validate_baseline_checkpoint(
            checkpoint,
            expected_architecture=architecture,
            data=data,
            context_length=args.context_length,
        )
        for key, _, checkpoint, architecture in baseline_specs
    }
    load_prompt_suite(args.suite)
    retrieval_suite = build_retrieval_suite(
        data.tokenizer,
        context_length=args.context_length,
        distances=args.retrieval_distances,
        pairs_per_distance=args.retrieval_pairs_per_distance,
        seed=args.seed,
    )
    device = choose_device(args.device)

    san_config = KiwiLMSANConfig(
        vocab_size=data.tokenizer.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        dropout=args.dropout,
        tie_embeddings=True,
        num_layers=args.san_layers,
        num_query_heads=args.query_heads,
        num_kv_heads=args.kv_heads,
        rms_norm_eps=args.rms_norm_eps,
        rope_base=args.rope_base,
    )
    training_targets = args.max_steps * args.batch_size * args.context_length
    warmup_targets = min(
        args.warmup_steps * args.batch_size * args.context_length,
        training_targets,
    )
    train_config = TrainConfig(
        max_steps=args.max_steps,
        max_tokens=training_targets,
        batch_size=args.batch_size,
        grad_accum_steps=1,
        lr=3e-4,
        min_lr=3e-5,
        warmup_steps=args.warmup_steps,
        warmup_tokens=warmup_targets,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    training_summary = train(
        san_config,
        PreparedTokenData(args.data_dir, seed=args.seed),
        args.output_dir / "kiwilm-san",
        train_config,
        device=device,
    )
    training_elapsed = time.perf_counter() - started
    training_memory = _accelerator_memory(device)
    checkpoint_value = training_summary.get("best_checkpoint")
    if checkpoint_value is None:
        raise RuntimeError("KiwiLM-SAN training did not produce a best checkpoint")
    san_checkpoint = Path(checkpoint_value)
    san_checkpoint_metadata = _validate_baseline_checkpoint(
        san_checkpoint,
        expected_architecture="kiwilm_san",
        data=data,
        context_length=args.context_length,
    )
    if _is_default_model_shape(args):
        parameter_count = int(training_summary["parameter_count"])
        if parameter_count != DEFAULT_SAN_PARAMETERS:
            raise RuntimeError(
                "default KiwiLM-SAN parameter count changed: "
                f"expected {DEFAULT_SAN_PARAMETERS}, got {parameter_count}"
            )

    evaluation_specs = (
        (
            "kiwilm_san",
            "KiwiLM-SAN",
            san_checkpoint,
            "kiwilm_san",
            san_checkpoint_metadata,
        ),
        *(
            (key, label, checkpoint, architecture, baseline_metadata[key])
            for key, label, checkpoint, architecture in baseline_specs
        ),
    )
    model_results: dict[str, dict[str, Any]] = {}
    for key, label, checkpoint, architecture, checkpoint_metadata in evaluation_specs:
        result = _evaluate_checkpoint(
            checkpoint,
            expected_architecture=architecture,
            label=label,
            data=data,
            device=device,
            batch_size=args.batch_size,
            context_length=args.context_length,
            post_eval_batches=args.post_eval_batches,
            generation_tokens=args.generation_tokens,
            generation_repeats=args.generation_repeats,
            seed=args.seed,
        )
        result["checkpoint_training_metadata"] = checkpoint_metadata["checkpoint_training_metadata"]
        if key == "kiwilm_san":
            result["training"] = training_summary
            result["live_training_measurement"] = {
                "comparable_across_models": False,
                "elapsed_seconds": training_elapsed,
                "end_to_end_valid_tokens_per_second": (
                    int(training_summary["tokens_seen"]) / training_elapsed
                ),
                "final_logged_throughput": _last_training_throughput(
                    Path(training_summary["metrics_path"])
                ),
                **training_memory,
            }
        else:
            result["live_training_measurement"] = {
                "comparable_across_models": False,
                "status": "not_measured",
                "reason": (
                    "saved baseline checkpoint; its historical training runtime may use "
                    "different hardware and is not compared with this SAN smoke run"
                ),
            }
        model_results[key] = result

    checkpoints = [spec[2] for spec in evaluation_specs]
    labels = [spec[1] for spec in evaluation_specs]
    comparison = compare_checkpoints(
        checkpoints,
        data=data,
        suite_path=args.suite,
        output_dir=args.output_dir / "comparison",
        device=device,
        labels=labels,
    )

    retrieval_evaluations = []
    for _, label, checkpoint, architecture, _ in evaluation_specs:
        model, _config = load_trained_model(
            checkpoint,
            data_fingerprint=data.fingerprint,
            device=device,
        )
        retrieval_evaluations.append(
            evaluate_retrieval_model(
                model,
                retrieval_suite,
                label=label,
                device=device,
                batch_size=args.retrieval_batch_size,
                architecture=architecture,
                checkpoint=str(checkpoint.resolve()),
            )
        )
        del model
        _release_accelerator_cache(device)
    retrieval_artifacts = write_retrieval_artifacts(
        args.output_dir / "retrieval",
        suite=retrieval_suite,
        evaluations=retrieval_evaluations,
        title="KiwiLM-SAN Context Retrieval Benchmark",
    )

    san_parameters = int(model_results["kiwilm_san"]["parameter_count"])
    summary = {
        "benchmark": "kiwilm-san-vs-model-x-vs-model-y-smoke",
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
            "training_targets": training_targets,
            "warmup_steps": args.warmup_steps,
            "warmup_targets": warmup_targets,
            "seed": args.seed,
            "precision": "fp32",
            "batch_mode": "packed",
            "post_eval_batches_per_mode": args.post_eval_batches,
            "retrieval_distances": list(args.retrieval_distances),
            "retrieval_pairs_per_distance": args.retrieval_pairs_per_distance,
        },
        "training_comparison": {
            "valid": False,
            "reason": (
                "only KiwiLM-SAN was trained live; Model X and Model Y use saved "
                "checkpoints and may have been trained on different hardware"
            ),
        },
        "parameter_deltas_from_san": {
            key: {
                "absolute": int(result["parameter_count"]) - san_parameters,
                "percent_of_san": (
                    100.0 * (int(result["parameter_count"]) - san_parameters) / san_parameters
                ),
            }
            for key, result in model_results.items()
            if key != "kiwilm_san"
        },
        "models": model_results,
        "comparison": comparison,
        "retrieval": {
            **retrieval_artifacts,
            "models": {
                evaluation["summary"]["label"]: evaluation["summary"]
                for evaluation in retrieval_evaluations
            },
        },
    }
    _atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_distances(value: str) -> tuple[int, ...]:
    try:
        distances = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "retrieval distances must be comma-separated integers"
        ) from error
    if not distances or any(distance <= 0 for distance in distances):
        raise argparse.ArgumentTypeError("retrieval distances must be positive")
    if len(set(distances)) != len(distances):
        raise argparse.ArgumentTypeError("retrieval distances must be unique")
    return distances


def _validate_baseline_checkpoint(
    checkpoint: Path,
    *,
    expected_architecture: str,
    data: PreparedTokenData,
    context_length: int,
) -> dict[str, Any]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"benchmark checkpoint does not exist: {checkpoint}")
    model, config = load_trained_model(
        checkpoint,
        data_fingerprint=data.fingerprint,
        device=torch.device("cpu"),
    )
    try:
        if config.architecture != expected_architecture:
            raise ValueError(
                f"checkpoint {checkpoint} has architecture {config.architecture!r}; "
                f"expected {expected_architecture!r}"
            )
        if config.vocab_size != data.tokenizer.vocab_size:
            raise ValueError(
                f"checkpoint {checkpoint} vocabulary differs from the smoke dataset: "
                f"expected {data.tokenizer.vocab_size}, got {config.vocab_size}"
            )
        if config.context_length != context_length:
            raise ValueError(
                f"checkpoint {checkpoint} context length differs from the smoke run: "
                f"expected {context_length}, got {config.context_length}"
            )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
    finally:
        del model
    return {
        "checkpoint": str(checkpoint.resolve()),
        "architecture": config.architecture,
        "vocab_size": config.vocab_size,
        "context_length": config.context_length,
        "parameter_count": parameter_count,
        "checkpoint_training_metadata": _checkpoint_training_metadata(checkpoint),
    }


def _checkpoint_training_metadata(checkpoint: Path) -> dict[str, Any] | None:
    if checkpoint.is_dir() or checkpoint.suffix == ".safetensors":
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        return None
    training_state = payload.get("training_state")
    tokens_seen = training_state.get("tokens_seen") if isinstance(training_state, dict) else None
    return {
        "step": payload.get("step"),
        "tokens_seen": tokens_seen,
        "train_config": payload.get("train_config"),
    }


def _evaluate_checkpoint(
    checkpoint: Path,
    *,
    expected_architecture: str,
    label: str,
    data: PreparedTokenData,
    device: torch.device,
    batch_size: int,
    context_length: int,
    post_eval_batches: int,
    generation_tokens: int,
    generation_repeats: int,
    seed: int,
) -> dict[str, Any]:
    model, config = load_trained_model(
        checkpoint,
        data_fingerprint=data.fingerprint,
        device=device,
    )
    try:
        if config.architecture != expected_architecture:
            raise ValueError(
                f"checkpoint architecture changed after validation: {config.architecture!r}"
            )
        model.eval()
        post_evaluation = {
            mode: evaluate(
                model,
                data,
                batch_size=batch_size,
                context_length=context_length,
                num_batches=post_eval_batches,
                device=device,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                batch_mode=mode,
                precision="fp32",
                seed=seed,
            )
            for mode in ("packed", "story")
        }
        generation = {
            cache: _benchmark_generation(
                model,
                data.tokenizer,
                device=device,
                context_length=context_length,
                max_new_tokens=generation_tokens,
                repeats=generation_repeats,
                cache=cache,
            )
            for cache in ("auto", "off")
        }
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
    finally:
        del model
        _release_accelerator_cache(device)
    return {
        "label": label,
        "architecture": config.architecture,
        "checkpoint": str(checkpoint.resolve()),
        "parameter_count": parameter_count,
        "post_evaluation": post_evaluation,
        "generation": generation,
    }


def _accelerator_memory(device: torch.device) -> dict[str, int | None]:
    if device.type == "cuda":
        return {
            "peak_accelerator_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "accelerator_allocated_memory_after_training_bytes": int(
                torch.cuda.memory_allocated(device)
            ),
        }
    if device.type == "mps":
        return {
            "peak_accelerator_memory_bytes": None,
            "accelerator_allocated_memory_after_training_bytes": int(
                torch.mps.current_allocated_memory()
            ),
        }
    return {
        "peak_accelerator_memory_bytes": None,
        "accelerator_allocated_memory_after_training_bytes": None,
    }


def _release_accelerator_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _is_default_model_shape(args: argparse.Namespace) -> bool:
    return (
        args.expected_data_fingerprint == DEFAULT_DATA_FINGERPRINT
        and args.context_length == 256
        and args.d_model == 256
        and args.san_layers == 16
        and args.query_heads == 8
        and args.kv_heads == 4
    )


if __name__ == "__main__":
    raise SystemExit(main())
