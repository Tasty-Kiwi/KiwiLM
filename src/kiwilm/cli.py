"""Command-line interface for preparing, training, and sampling KiwiLM."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from kiwilm import __version__
from kiwilm.checkpoint import load_checkpoint
from kiwilm.config import GatedCNNConfig, ModelConfig
from kiwilm.data import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_REVISION,
    DEFAULT_TRAIN_LIMIT,
    DEFAULT_VALIDATION_LIMIT,
    DEFAULT_VOCAB_SIZE,
    PreparedTokenData,
    prepare_tinystories,
)
from kiwilm.generation import generate
from kiwilm.models import build_model
from kiwilm.training import TrainConfig, choose_device, evaluate, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kiwilm",
        description="Train modular toy causal language models on TinyStories.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="stream TinyStories, train a BPE tokenizer, and pack token streams",
    )
    prepare_parser.add_argument("--output-dir", type=Path, default=Path("data/tinystories"))
    prepare_parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    prepare_parser.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    prepare_parser.add_argument("--text-field", default="text")
    prepare_parser.add_argument("--train-limit", type=int, default=DEFAULT_TRAIN_LIMIT)
    prepare_parser.add_argument(
        "--validation-limit",
        type=int,
        default=DEFAULT_VALIDATION_LIMIT,
    )
    prepare_parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    prepare_parser.add_argument("--min-frequency", type=int, default=2)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--quiet", action="store_true")
    prepare_parser.set_defaults(handler=_prepare_command)

    train_parser = subparsers.add_parser(
        "train",
        help="train Model A against prepared token streams",
    )
    _add_data_argument(train_parser)
    train_parser.add_argument("--output-dir", type=Path, default=Path("runs/model-a"))
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--context-length", type=int, default=256)
    train_parser.add_argument("--d-model", type=int, default=256)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--untie-embeddings", action="store_true")
    train_parser.add_argument("--max-steps", type=int, default=2_000)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--grad-accum-steps", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    train_parser.add_argument("--warmup-steps", type=int, default=100)
    train_parser.add_argument("--weight-decay", type=float, default=0.1)
    train_parser.add_argument("--beta2", type=float, default=0.95)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--eval-interval", type=int, default=200)
    train_parser.add_argument("--eval-batches", type=int, default=20)
    train_parser.add_argument("--checkpoint-interval", type=int, default=500)
    train_parser.add_argument("--log-interval", type=int, default=10)
    train_parser.add_argument("--sample-prompt", default="Once upon a time")
    train_parser.add_argument(
        "--sample-tokens",
        type=int,
        default=64,
        help="greedy tokens generated after training; 0 disables the sample",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(handler=_train_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="measure sampled validation loss and perplexity",
    )
    _add_data_argument(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--batch-size", type=int, default=32)
    evaluate_parser.add_argument("--batches", type=int, default=20)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.set_defaults(handler=_evaluate_command)

    generate_parser = subparsers.add_parser(
        "generate",
        help="generate text from a trained checkpoint",
    )
    _add_data_argument(generate_parser)
    generate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--device", default="auto")
    generate_parser.add_argument("--max-new-tokens", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="sampling shortlist size; 0 disables top-k filtering",
    )
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.set_defaults(handler=_generate_command)
    return parser


def _add_data_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tinystories"),
        help="directory containing metadata.json and prepared token artifacts",
    )


def _prepare_command(args: argparse.Namespace) -> int:
    metadata = prepare_tinystories(
        args.output_dir,
        dataset_name=args.dataset_name,
        revision=args.revision,
        text_field=args.text_field,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=not args.quiet,
        force=args.force,
    )
    _print_json(
        {
            "output_dir": str(args.output_dir.resolve()),
            "fingerprint": metadata["fingerprint"],
            "dataset": metadata["dataset"],
            "vocab_size": metadata["tokenizer"]["vocab_size"],
            "splits": metadata["splits"],
        }
    )
    return 0


def _train_command(args: argparse.Namespace) -> int:
    data = PreparedTokenData(args.data_dir, seed=args.seed)
    model_config = GatedCNNConfig(
        vocab_size=data.tokenizer.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        dropout=args.dropout,
        tie_embeddings=not args.untie_embeddings,
    )
    train_config = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        sample_prompt=args.sample_prompt,
        sample_tokens=args.sample_tokens,
        seed=args.seed,
    )
    summary = train(
        model_config,
        data,
        args.output_dir,
        train_config,
        device=args.device,
        resume_from=args.resume,
    )
    _print_json(summary)
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    data = PreparedTokenData(args.data_dir, seed=args.seed)
    device = choose_device(args.device)
    model, config = _load_trained_model(
        args.checkpoint,
        data_fingerprint=data.fingerprint,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    metrics = evaluate(
        model,
        data,
        batch_size=args.batch_size,
        context_length=config.context_length,
        num_batches=args.batches,
        device=device,
        generator=generator,
    )
    _print_json(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "data_fingerprint": data.fingerprint,
            "device": str(device),
            **metrics,
        }
    )
    return 0


def _generate_command(args: argparse.Namespace) -> int:
    data = PreparedTokenData(args.data_dir, seed=args.seed)
    device = choose_device(args.device)
    model, config = _load_trained_model(
        args.checkpoint,
        data_fingerprint=data.fingerprint,
        device=device,
    )
    text = generate(
        model,
        data.tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=config.context_length,
        temperature=args.temperature,
        top_k=None if args.top_k == 0 else args.top_k,
        seed=args.seed,
        device=device,
    )
    print(text)
    return 0


def _load_trained_model(
    checkpoint_path: str | Path,
    *,
    data_fingerprint: str,
    device: torch.device,
) -> tuple[torch.nn.Module, ModelConfig]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    serialized_config = payload.get("model_config")
    if not isinstance(serialized_config, dict):
        raise ValueError("checkpoint does not contain a model configuration")
    config = ModelConfig.from_dict(serialized_config)
    model = build_model(config)
    load_checkpoint(
        checkpoint_path,
        model=model,
        expected_model_config=config,
        expected_data_fingerprint=data_fingerprint,
        map_location="cpu",
        restore_rng=False,
    )
    model.to(device)
    model.eval()
    return model, config


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


__all__ = ["build_parser", "main"]
