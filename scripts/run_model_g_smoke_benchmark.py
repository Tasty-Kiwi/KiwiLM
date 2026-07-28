#!/usr/bin/env python3
"""Train Model G and compare it with an existing matched smoke benchmark."""

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

from kiwilm.comparison import compare_checkpoints
from kiwilm.config import CNNFFNAttentionConfig
from kiwilm.data import PreparedTokenData
from kiwilm.inference import load_trained_model
from kiwilm.training import TrainConfig, choose_device, evaluate, train

DEFAULT_MODEL_G_PARAMETERS = 8_417_536


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("runs/benchmarks/transformer-smoke"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/model-g-smoke"),
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
    parser.add_argument("--post-eval-batches", type=int, default=50)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--generation-repeats", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_empty_destination(args.output_dir)
    reference = _load_reference_summary(args.reference_dir)
    data = PreparedTokenData(args.data_dir)
    device = choose_device(args.device)
    _validate_reference(
        reference,
        data_fingerprint=data.fingerprint,
        expected_data_fingerprint=args.expected_data_fingerprint,
        device=device,
    )

    model_b_checkpoint = Path(reference["models"]["model_b"]["training"]["best_checkpoint"])
    transformer_checkpoint = Path(reference["models"]["transformer"]["training"]["best_checkpoint"])
    model_b_payload = _load_checkpoint_payload(model_b_checkpoint)
    transformer_payload = _load_checkpoint_payload(transformer_checkpoint)
    if model_b_payload["train_config"] != transformer_payload["train_config"]:
        raise ValueError("reference checkpoints do not use identical training settings")
    train_config = TrainConfig(**dict(model_b_payload["train_config"]))
    model_b_config = dict(model_b_payload["model_config"])
    model_g_config = CNNFFNAttentionConfig(
        vocab_size=int(model_b_config["vocab_size"]),
        context_length=int(model_b_config["context_length"]),
        d_model=int(model_b_config["d_model"]),
        dropout=float(model_b_config["dropout"]),
        tie_embeddings=bool(model_b_config["tie_embeddings"]),
        kernel_size=int(model_b_config["kernel_size"]),
        pre_attention_dilations=tuple(model_b_config["pre_attention_dilations"]),
        post_attention_dilations=tuple(model_b_config["post_attention_dilations"]),
        num_heads=int(model_b_config["num_heads"]),
        feedforward_dim=int(model_b_config["feedforward_dim"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / "model-g"
    started = time.perf_counter()
    training_summary = train(
        model_g_config,
        PreparedTokenData(args.data_dir, seed=train_config.seed),
        run_dir,
        train_config,
        device=device,
    )
    training_elapsed = time.perf_counter() - started
    checkpoint_value = training_summary["best_checkpoint"]
    if checkpoint_value is None:
        raise RuntimeError("Model G training did not produce a best checkpoint")
    model_g_checkpoint = Path(checkpoint_value)
    model, loaded_config = load_trained_model(
        model_g_checkpoint,
        data_fingerprint=data.fingerprint,
        device=device,
    )
    post_evaluation = {
        mode: evaluate(
            model,
            data,
            batch_size=train_config.batch_size,
            context_length=loaded_config.context_length,
            num_batches=args.post_eval_batches,
            device=device,
            generator=torch.Generator(device="cpu").manual_seed(train_config.seed),
            batch_mode=mode,
            precision=train_config.precision,
            seed=train_config.seed,
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
    model_g_result = {
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
    if (
        args.expected_data_fingerprint == DEFAULT_DATA_FINGERPRINT
        and model_g_result["parameter_count"] != DEFAULT_MODEL_G_PARAMETERS
    ):
        raise RuntimeError(
            "default Model G parameter count changed: expected "
            f"{DEFAULT_MODEL_G_PARAMETERS}, got {model_g_result['parameter_count']}"
        )

    comparison = compare_checkpoints(
        [model_b_checkpoint, transformer_checkpoint, model_g_checkpoint],
        data=data,
        suite_path=args.suite,
        output_dir=args.output_dir / "comparison",
        device=device,
        labels=["Model B", "GPT-style Transformer", "Model G"],
    )
    model_b_parameters = int(reference["models"]["model_b"]["parameter_count"])
    summary = {
        "benchmark": "model-b-vs-transformer-vs-model-g-smoke",
        "data_dir": str(args.data_dir.resolve()),
        "data_fingerprint": data.fingerprint,
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "settings": reference["settings"],
        "reference_summary": str((args.reference_dir / "summary.json").resolve()),
        "parameter_delta_model_g_vs_model_b": {
            "absolute": int(model_g_result["parameter_count"]) - model_b_parameters,
            "percent_of_model_b": (
                100.0
                * (int(model_g_result["parameter_count"]) - model_b_parameters)
                / model_b_parameters
            ),
        },
        "models": {
            "model_b": reference["models"]["model_b"],
            "transformer": reference["models"]["transformer"],
            "model_g": model_g_result,
        },
        "comparison": comparison,
    }
    _atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_reference_summary(reference_dir: Path) -> dict[str, Any]:
    path = reference_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"reference benchmark summary not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("reference benchmark summary must contain an object")
    try:
        if value["benchmark"] != "model-b-vs-transformer-smoke":
            raise ValueError("reference summary is not the matched smoke benchmark")
        value["models"]["model_b"]
        value["models"]["transformer"]
        value["settings"]
    except (KeyError, TypeError) as error:
        raise ValueError("reference benchmark summary is incomplete") from error
    return value


def _validate_reference(
    reference: dict[str, Any],
    *,
    data_fingerprint: str,
    expected_data_fingerprint: str,
    device: torch.device,
) -> None:
    if data_fingerprint != expected_data_fingerprint:
        raise ValueError(
            "prepared dataset fingerprint differs from the smoke benchmark: "
            f"expected {expected_data_fingerprint}, got {data_fingerprint}"
        )
    if reference.get("data_fingerprint") != data_fingerprint:
        raise ValueError("reference benchmark uses a different prepared dataset")
    if reference.get("device") != str(device):
        raise ValueError(
            f"reference benchmark used {reference.get('device')}, but Model G would use {device}"
        )
    environment = reference.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("reference benchmark is missing its environment")
    if environment.get("torch") != torch.__version__:
        raise ValueError("reference benchmark used a different PyTorch version")
    if environment.get("platform") != platform.platform():
        raise ValueError("reference benchmark ran on a different platform")


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"reference checkpoint not found: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"reference checkpoint must contain a mapping: {path}")
    for key in ("model_config", "train_config", "data_fingerprint"):
        if key not in value:
            raise ValueError(f"reference checkpoint is missing {key}: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
