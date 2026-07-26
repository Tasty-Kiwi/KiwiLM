"""Command-line interface for preparing, training, and sampling KiwiLM."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from kiwilm import __version__
from kiwilm.comparison import compare_checkpoints
from kiwilm.config import (
    CNNAttentionConfig,
    CNNAttentionMambaConfig,
    CNNDeepInterleavedAttentionConfig,
    CNNDualAttentionConfig,
    CNNInterleavedAttentionConfig,
    GatedCNNConfig,
)
from kiwilm.data import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_REVISION,
    DEFAULT_TRAIN_LIMIT,
    DEFAULT_VALIDATION_LIMIT,
    DEFAULT_VOCAB_SIZE,
    PreparedTokenData,
    prepare_tinystories,
)
from kiwilm.generation import generate, generate_stream
from kiwilm.inference import load_trained_model
from kiwilm.training import TrainConfig, choose_device, evaluate, train

_load_trained_model = load_trained_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kiwilm",
        description="Train modular toy causal language models on TinyStories.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="stream TinyStories, select a BPE tokenizer, and pack token streams",
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
    prepare_parser.add_argument(
        "--tokenizer-from",
        type=Path,
        help="reuse the validated tokenizer from another prepared dataset",
    )
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--quiet", action="store_true")
    prepare_parser.set_defaults(handler=_prepare_command)

    train_parser = subparsers.add_parser(
        "train",
        help="train a model architecture against prepared token streams",
    )
    _add_data_argument(train_parser)
    train_parser.add_argument(
        "--architecture",
        choices=(
            "gated_cnn",
            "cnn_attention",
            "cnn_dual_attention",
            "cnn_attention_mamba",
            "cnn_interleaved_attention",
            "cnn_deep_interleaved_attention",
        ),
        default="gated_cnn",
    )
    train_parser.add_argument("--output-dir", type=Path)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--context-length", type=int, default=256)
    train_parser.add_argument("--d-model", type=int, default=256)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--attention-heads", type=int, default=8)
    train_parser.add_argument("--attention-feedforward-dim", type=int, default=1024)
    train_parser.add_argument("--mamba-inner-dim", type=int, default=896)
    train_parser.add_argument("--mamba-state-dim", type=int, default=16)
    train_parser.add_argument("--mamba-conv-kernel", type=int, default=4)
    train_parser.add_argument("--mamba-dt-rank", type=int, default=16)
    train_parser.add_argument("--untie-embeddings", action="store_true")
    train_parser.add_argument("--max-steps", type=int, default=2_000)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--grad-accum-steps", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    train_parser.add_argument("--warmup-steps", type=int, default=100)
    train_parser.add_argument("--max-tokens", type=int)
    train_parser.add_argument("--warmup-tokens", type=int)
    train_parser.add_argument(
        "--batch-mode", choices=("packed", "story"), default="packed"
    )
    train_parser.add_argument(
        "--eval-mode", choices=("packed", "story", "both"), default="packed"
    )
    train_parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16", "auto"),
        default="fp32",
    )
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
    evaluate_parser.add_argument(
        "--batch-mode", choices=("packed", "story"), default="packed"
    )
    evaluate_parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16", "auto"),
        default="fp32",
    )
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
    generate_parser.add_argument(
        "--cache",
        choices=("auto", "off"),
        default="auto",
        help="use incremental model caches when available",
    )
    generate_parser.add_argument(
        "--stream",
        action="store_true",
        help="print decoded text as tokens are generated",
    )
    generate_parser.set_defaults(handler=_generate_command)

    compare_parser = subparsers.add_parser(
        "compare",
        help="generate a reproducible side-by-side checkpoint report",
    )
    _add_data_argument(compare_parser)
    compare_parser.add_argument("--checkpoint-a", type=Path)
    compare_parser.add_argument("--checkpoint-b", type=Path)
    compare_parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        help="two or more checkpoints for an N-way report",
    )
    compare_parser.add_argument(
        "--suite",
        type=Path,
        default=Path("eval/story-consistency-prompts.json"),
    )
    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/comparisons/model-a-vs-model-b"),
    )
    compare_parser.add_argument("--label-a")
    compare_parser.add_argument("--label-b")
    compare_parser.add_argument(
        "--labels",
        nargs="+",
        help="labels corresponding positionally to --checkpoints",
    )
    compare_parser.add_argument("--device", default="auto")
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.set_defaults(handler=_compare_command)
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
        tokenizer_from=args.tokenizer_from,
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
    shared_config = {
        "vocab_size": data.tokenizer.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "dropout": args.dropout,
        "tie_embeddings": not args.untie_embeddings,
    }
    if args.architecture == "cnn_attention":
        model_config = CNNAttentionConfig(
            **shared_config,
            num_heads=args.attention_heads,
            feedforward_dim=args.attention_feedforward_dim,
        )
        default_output_dir = Path("runs/model-b")
    elif args.architecture == "cnn_dual_attention":
        model_config = CNNDualAttentionConfig(
            **shared_config,
            num_heads=args.attention_heads,
            feedforward_dim=args.attention_feedforward_dim,
        )
        default_output_dir = Path("runs/model-c")
    elif args.architecture == "cnn_attention_mamba":
        model_config = CNNAttentionMambaConfig(
            **shared_config,
            num_heads=args.attention_heads,
            feedforward_dim=args.attention_feedforward_dim,
            mamba_inner_dim=args.mamba_inner_dim,
            mamba_state_dim=args.mamba_state_dim,
            mamba_conv_kernel=args.mamba_conv_kernel,
            mamba_dt_rank=args.mamba_dt_rank,
        )
        default_output_dir = Path("runs/model-d")
    elif args.architecture == "cnn_interleaved_attention":
        model_config = CNNInterleavedAttentionConfig(
            **shared_config,
            num_heads=args.attention_heads,
            feedforward_dim=args.attention_feedforward_dim,
        )
        default_output_dir = Path("runs/model-e")
    elif args.architecture == "cnn_deep_interleaved_attention":
        model_config = CNNDeepInterleavedAttentionConfig(
            **shared_config,
            num_heads=args.attention_heads,
            feedforward_dim=args.attention_feedforward_dim,
        )
        default_output_dir = Path("runs/model-f")
    else:
        model_config = GatedCNNConfig(**shared_config)
        default_output_dir = Path("runs/model-a")
    output_dir = args.output_dir or default_output_dir
    train_config = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        max_tokens=args.max_tokens,
        warmup_tokens=args.warmup_tokens,
        batch_mode=args.batch_mode,
        eval_mode=args.eval_mode,
        precision=args.precision,
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
        output_dir,
        train_config,
        device=args.device,
        resume_from=args.resume,
    )
    _print_json(summary)
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    data = PreparedTokenData(args.data_dir, seed=args.seed)
    device = choose_device(args.device)
    model, config = load_trained_model(
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
        batch_mode=args.batch_mode,
        precision=args.precision,
        seed=args.seed,
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
    model, config = load_trained_model(
        args.checkpoint,
        data_fingerprint=data.fingerprint,
        device=device,
    )
    generation_options = {
        "max_new_tokens": args.max_new_tokens,
        "context_length": config.context_length,
        "temperature": args.temperature,
        "top_k": None if args.top_k == 0 else args.top_k,
        "seed": args.seed,
        "device": device,
        "cache": args.cache,
    }
    if args.stream:
        for chunk in generate_stream(
            model,
            data.tokenizer,
            args.prompt,
            **generation_options,
        ):
            print(chunk, end="", flush=True)
        print()
    else:
        text = generate(
            model,
            data.tokenizer,
            args.prompt,
            **generation_options,
        )
        print(text)
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    data = PreparedTokenData(args.data_dir, seed=args.seed)
    if args.checkpoints is not None:
        if (
            args.checkpoint_a is not None
            or args.checkpoint_b is not None
            or args.label_a is not None
            or args.label_b is not None
        ):
            raise ValueError(
                "use either --checkpoints/--labels or the "
                "--checkpoint-a/--checkpoint-b form"
            )
        checkpoints = args.checkpoints
        labels = args.labels
    else:
        if args.checkpoint_a is None or args.checkpoint_b is None:
            raise ValueError(
                "pass --checkpoints with two or more paths, or both "
                "--checkpoint-a and --checkpoint-b"
            )
        if args.labels is not None:
            raise ValueError("--labels requires --checkpoints")
        checkpoints = [args.checkpoint_a, args.checkpoint_b]
        labels = [args.label_a, args.label_b]
    summary = compare_checkpoints(
        checkpoints,
        data=data,
        suite_path=args.suite,
        output_dir=args.output_dir,
        device=choose_device(args.device),
        labels=labels,
    )
    _print_json(summary)
    return 0


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


__all__ = ["build_parser", "main"]
