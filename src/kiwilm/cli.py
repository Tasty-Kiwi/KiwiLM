"""Command-line interface for preparing, training, and sampling KiwiLM."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
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
    CNNFFNAttentionConfig,
    CNNInterleavedAttentionConfig,
    GatedCNNConfig,
    KiwiLMSANConfig,
    ModelConfig,
    ModelXConfig,
    ModelYConfig,
    ModelZParallelConfig,
    TransformerConfig,
)
from kiwilm.data import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_REVISION,
    DEFAULT_SIMPLESTORIES_DATASET_NAME,
    DEFAULT_SIMPLESTORIES_DATASET_REVISION,
    DEFAULT_SIMPLESTORIES_TRAIN_LIMIT,
    DEFAULT_SIMPLESTORIES_VALIDATION_LIMIT,
    DEFAULT_TRAIN_LIMIT,
    DEFAULT_VALIDATION_LIMIT,
    DEFAULT_VOCAB_SIZE,
    PreparedTokenData,
    export_tokenizer_bundle,
    prepare_simplestories,
    prepare_tinystories,
)
from kiwilm.generation import generate, generate_stream
from kiwilm.inference import load_trained_model
from kiwilm.safetensors_io import (
    export_safetensors_bundle,
    read_safetensors_metadata,
)
from kiwilm.sft import (
    DEFAULT_INSTRUCT_DATASET,
    DEFAULT_INSTRUCT_REVISION,
    DEFAULT_INSTRUCT_TRAIN_LIMIT,
    DEFAULT_INSTRUCT_VALIDATION_LIMIT,
    DEFAULT_REQUIRED_WORD_WEIGHT,
    SFT_FORMATS,
    PreparedSFTData,
    load_prepared_data,
    prepare_tinystories_instruct,
)
from kiwilm.sft_report import generate_sft_adherence_report
from kiwilm.tokenizer import ByteBPETokenizer
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

    prepare_simplestories_parser = subparsers.add_parser(
        "prepare-simplestories",
        help="prepare SimpleStories using a frozen KiwiLM tokenizer",
    )
    prepare_simplestories_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/simplestories-250k"),
    )
    prepare_simplestories_parser.add_argument(
        "--tokenizer-from",
        type=Path,
        required=True,
    )
    prepare_simplestories_parser.add_argument(
        "--dataset-name",
        default=DEFAULT_SIMPLESTORIES_DATASET_NAME,
    )
    prepare_simplestories_parser.add_argument(
        "--revision",
        default=DEFAULT_SIMPLESTORIES_DATASET_REVISION,
    )
    prepare_simplestories_parser.add_argument("--text-field", default="story")
    prepare_simplestories_parser.add_argument(
        "--train-limit",
        type=int,
        default=DEFAULT_SIMPLESTORIES_TRAIN_LIMIT,
    )
    prepare_simplestories_parser.add_argument(
        "--validation-limit",
        type=int,
        default=DEFAULT_SIMPLESTORIES_VALIDATION_LIMIT,
    )
    prepare_simplestories_parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
    )
    prepare_simplestories_parser.add_argument("--min-frequency", type=int, default=2)
    prepare_simplestories_parser.add_argument("--force", action="store_true")
    prepare_simplestories_parser.add_argument("--quiet", action="store_true")
    prepare_simplestories_parser.set_defaults(
        handler=_prepare_simplestories_command
    )

    prepare_instruct_parser = subparsers.add_parser(
        "prepare-instruct",
        help="prepare response-masked TinyStoriesInstruct data",
    )
    prepare_instruct_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tinystories-instruct-50k"),
    )
    prepare_instruct_parser.add_argument("--tokenizer-from", type=Path, required=True)
    prepare_instruct_parser.add_argument(
        "--dataset-name",
        default=DEFAULT_INSTRUCT_DATASET,
    )
    prepare_instruct_parser.add_argument(
        "--revision",
        default=DEFAULT_INSTRUCT_REVISION,
    )
    prepare_instruct_parser.add_argument(
        "--train-limit",
        type=int,
        default=DEFAULT_INSTRUCT_TRAIN_LIMIT,
    )
    prepare_instruct_parser.add_argument(
        "--validation-limit",
        type=int,
        default=DEFAULT_INSTRUCT_VALIDATION_LIMIT,
    )
    prepare_instruct_parser.add_argument("--train-file", type=Path)
    prepare_instruct_parser.add_argument("--validation-file", type=Path)
    prepare_instruct_parser.add_argument(
        "--sft-format",
        choices=SFT_FORMATS,
        default="v1",
    )
    prepare_instruct_parser.add_argument(
        "--required-word-weight",
        type=float,
        default=DEFAULT_REQUIRED_WORD_WEIGHT,
    )
    prepare_instruct_parser.add_argument("--force", action="store_true")
    prepare_instruct_parser.add_argument("--quiet", action="store_true")
    prepare_instruct_parser.set_defaults(handler=_prepare_instruct_command)

    export_tokenizer_parser = subparsers.add_parser(
        "export-tokenizer",
        help="export a prepared tokenizer as a small portable bundle",
    )
    export_tokenizer_parser.add_argument("--data-dir", type=Path, required=True)
    export_tokenizer_parser.add_argument("--output-dir", type=Path, required=True)
    export_tokenizer_parser.add_argument("--force", action="store_true")
    export_tokenizer_parser.set_defaults(handler=_export_tokenizer_command)

    export_safetensors_parser = subparsers.add_parser(
        "export-safetensors",
        help="export an inference-only KiwiLM Safetensors bundle",
    )
    export_safetensors_parser.add_argument(
        "--tokenizer-from",
        type=Path,
        required=True,
        help="prepared dataset containing metadata.json and the exact tokenizer",
    )
    export_safetensors_parser.add_argument("--checkpoint", type=Path, required=True)
    export_safetensors_parser.add_argument("--output-dir", type=Path, required=True)
    export_safetensors_parser.add_argument("--variant", required=True)
    export_safetensors_parser.set_defaults(handler=_export_safetensors_command)

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
            "cnn_attention_ffn",
            "cnn_dual_attention",
            "cnn_attention_mamba",
            "cnn_interleaved_attention",
            "cnn_deep_interleaved_attention",
            "transformer",
            "model_x",
            "model_y",
            "model_z_parallel",
            "kiwilm_san",
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
    train_parser.add_argument("--swiglu-dim", type=int, default=640)
    train_parser.add_argument(
        "--model-y-swiglu-dim",
        type=int,
        default=720,
    )
    train_parser.add_argument(
        "--model-z-swiglu-dim",
        type=int,
        default=1280,
    )
    train_parser.add_argument("--san-layers", type=int, default=16)
    train_parser.add_argument("--san-kv-heads", type=int, default=4)
    train_parser.add_argument("--san-rms-norm-eps", type=float, default=1e-6)
    train_parser.add_argument("--san-rope-base", type=float, default=10_000.0)
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

    cpt_parser = subparsers.add_parser(
        "cpt",
        help="continue pretraining a checkpoint on prepared story data",
    )
    _add_data_argument(
        cpt_parser,
        default=Path("data/simplestories-250k"),
    )
    cpt_initialization = cpt_parser.add_mutually_exclusive_group(required=True)
    cpt_initialization.add_argument("--init-from", type=Path)
    cpt_initialization.add_argument("--resume", type=Path)
    cpt_parser.add_argument("--output-dir", type=Path)
    cpt_parser.add_argument("--device", default="auto")
    cpt_parser.add_argument("--max-tokens", type=int, default=50_000_000)
    cpt_parser.add_argument("--warmup-tokens", type=int, default=1_000_000)
    cpt_parser.add_argument("--max-steps", type=int, default=6_000)
    cpt_parser.add_argument("--batch-size", type=int, default=64)
    cpt_parser.add_argument("--grad-accum-steps", type=int, default=1)
    cpt_parser.add_argument("--learning-rate", type=float, default=3e-5)
    cpt_parser.add_argument("--min-learning-rate", type=float, default=3e-6)
    cpt_parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16", "auto"),
        default="auto",
    )
    cpt_parser.add_argument("--weight-decay", type=float, default=0.1)
    cpt_parser.add_argument("--beta2", type=float, default=0.95)
    cpt_parser.add_argument("--grad-clip", type=float, default=1.0)
    cpt_parser.add_argument("--eval-interval", type=int, default=500)
    cpt_parser.add_argument("--eval-batches", type=int, default=50)
    cpt_parser.add_argument("--checkpoint-interval", type=int, default=500)
    cpt_parser.add_argument("--log-interval", type=int, default=10)
    cpt_parser.add_argument("--sample-prompt", default="Once upon a time")
    cpt_parser.add_argument("--sample-tokens", type=int, default=160)
    cpt_parser.add_argument("--seed", type=int, default=42)
    cpt_parser.set_defaults(handler=_cpt_command)

    sft_parser = subparsers.add_parser(
        "sft",
        help="supervised fine-tune a checkpoint on prepared instruction data",
    )
    _add_data_argument(
        sft_parser,
        default=Path("data/tinystories-instruct-50k"),
    )
    initialization = sft_parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--init-from", type=Path)
    initialization.add_argument("--resume", type=Path)
    sft_parser.add_argument("--output-dir", type=Path)
    sft_parser.add_argument("--device", default="auto")
    sft_parser.add_argument("--max-tokens", type=int, default=10_000_000)
    sft_parser.add_argument("--warmup-tokens", type=int, default=250_000)
    sft_parser.add_argument("--max-steps", type=int, default=10_000)
    sft_parser.add_argument("--batch-size", type=int, default=8)
    sft_parser.add_argument("--grad-accum-steps", type=int, default=4)
    sft_parser.add_argument("--learning-rate", type=float, default=1e-5)
    sft_parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    sft_parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16", "auto"),
        default="auto",
    )
    sft_parser.add_argument("--weight-decay", type=float, default=0.1)
    sft_parser.add_argument("--beta2", type=float, default=0.95)
    sft_parser.add_argument("--grad-clip", type=float, default=1.0)
    sft_parser.add_argument("--eval-interval", type=int, default=250)
    sft_parser.add_argument("--eval-batches", type=int, default=50)
    sft_parser.add_argument("--checkpoint-interval", type=int, default=500)
    sft_parser.add_argument("--log-interval", type=int, default=10)
    sft_parser.add_argument(
        "--sample-prompt",
        default=(
            "Features: Dialogue\n"
            "Words: oak, gloomy, kind\n"
            "Summary: Two friends help each other get home before dark.\n"
            "Story:\n"
        ),
    )
    sft_parser.add_argument("--sample-tokens", type=int, default=160)
    sft_parser.add_argument("--seed", type=int, default=42)
    sft_parser.set_defaults(handler=_sft_command)

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
        "--batch-mode", choices=("packed", "story", "sft"), default="packed"
    )
    evaluate_parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16", "auto"),
        default="fp32",
    )
    evaluate_parser.add_argument(
        "--allow-data-mismatch",
        action="store_true",
        help=(
            "explicitly evaluate on another prepared dataset; tokenizer vocabulary "
            "size must still match"
        ),
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
        default=Path("examples/comparisons/model-a-vs-model-b"),
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

    sft_report_parser = subparsers.add_parser(
        "sft-report",
        help="generate and score a fixed instruction-adherence suite",
    )
    _add_data_argument(
        sft_report_parser,
        default=Path("data/tinystories-instruct-50k"),
    )
    sft_report_parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
    )
    sft_report_parser.add_argument("--labels", nargs="+")
    sft_report_parser.add_argument(
        "--suite",
        type=Path,
        default=Path("eval/instruction-adherence-prompts.json"),
    )
    sft_report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/comparisons/sft-adherence"),
    )
    sft_report_parser.add_argument("--device", default="auto")
    sft_report_parser.add_argument(
        "--cache",
        choices=("auto", "off"),
        default="off",
    )
    sft_report_parser.set_defaults(handler=_sft_report_command)
    return parser


def _add_data_argument(
    parser: argparse.ArgumentParser,
    *,
    default: Path = Path("data/tinystories"),
) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default,
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


def _prepare_simplestories_command(args: argparse.Namespace) -> int:
    metadata = prepare_simplestories(
        args.output_dir,
        tokenizer_from=args.tokenizer_from,
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
            "tokenizer": metadata["tokenizer"],
            "splits": metadata["splits"],
        }
    )
    return 0


def _prepare_instruct_command(args: argparse.Namespace) -> int:
    metadata = prepare_tinystories_instruct(
        args.output_dir,
        tokenizer_from=args.tokenizer_from,
        dataset_name=args.dataset_name,
        revision=args.revision,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        train_file=args.train_file,
        validation_file=args.validation_file,
        sft_format=args.sft_format,
        required_word_weight=args.required_word_weight,
        show_progress=not args.quiet,
        force=args.force,
    )
    _print_json(
        {
            "output_dir": str(args.output_dir.resolve()),
            "fingerprint": metadata["fingerprint"],
            "dataset": metadata["dataset"],
            "task": metadata["task"],
            "config": metadata["config"],
            "splits": metadata["splits"],
        }
    )
    return 0


def _export_safetensors_command(args: argparse.Namespace) -> int:
    metadata_path = args.tokenizer_from / "metadata.json"
    try:
        prepared_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read prepared metadata at {metadata_path}") from error
    if not isinstance(prepared_metadata, Mapping):
        raise ValueError("prepared metadata must be an object")
    prepared_fingerprint = prepared_metadata.get("fingerprint")
    if not isinstance(prepared_fingerprint, str):
        raise ValueError("prepared data does not contain a fingerprint")
    tokenizer_metadata = prepared_metadata.get("tokenizer")
    if not isinstance(tokenizer_metadata, Mapping):
        raise ValueError("prepared data does not contain tokenizer metadata")
    tokenizer_file = tokenizer_metadata.get("file")
    tokenizer_sha256 = tokenizer_metadata.get("sha256")
    if not isinstance(tokenizer_file, str):
        raise ValueError("prepared data does not name its tokenizer artifact")
    if not isinstance(tokenizer_sha256, str):
        raise ValueError("prepared data does not contain a tokenizer checksum")
    manifest = export_safetensors_bundle(
        args.checkpoint,
        args.output_dir,
        tokenizer_path=args.tokenizer_from / tokenizer_file,
        expected_data_fingerprint=prepared_fingerprint,
        expected_tokenizer_sha256=tokenizer_sha256,
        variant=args.variant,
    )
    _print_json(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            **manifest,
        }
    )
    return 0


def _export_tokenizer_command(args: argparse.Namespace) -> int:
    bundle = export_tokenizer_bundle(
        args.data_dir,
        args.output_dir,
        force=args.force,
    )
    _print_json(
        {
            "output_dir": str(args.output_dir.resolve()),
            "fingerprint": bundle["fingerprint"],
            "source_dataset_fingerprint": bundle["source_dataset_fingerprint"],
            "tokenizer": bundle["tokenizer"],
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
    elif args.architecture == "cnn_attention_ffn":
        model_config = CNNFFNAttentionConfig(
            **shared_config,
            num_heads=args.attention_heads,
            feedforward_dim=args.attention_feedforward_dim,
        )
        default_output_dir = Path("runs/model-g")
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
    elif args.architecture == "transformer":
        model_config = TransformerConfig(
            **shared_config,
            num_heads=args.attention_heads,
            feedforward_dim=args.attention_feedforward_dim,
        )
        default_output_dir = Path("runs/transformer")
    elif args.architecture == "model_y":
        model_config = ModelYConfig(
            **shared_config,
            num_heads=args.attention_heads,
            swiglu_dim=args.model_y_swiglu_dim,
        )
        default_output_dir = Path("runs/model-y")
    elif args.architecture == "model_x":
        model_config = ModelXConfig(
            **shared_config,
            num_heads=args.attention_heads,
            swiglu_dim=args.swiglu_dim,
        )
        default_output_dir = Path("runs/model-x")
    elif args.architecture == "model_z_parallel":
        model_config = ModelZParallelConfig(
            **shared_config,
            num_heads=args.attention_heads,
            swiglu_dim=args.model_z_swiglu_dim,
        )
        default_output_dir = Path("runs/model-z-parallel")
    elif args.architecture == "kiwilm_san":
        model_config = KiwiLMSANConfig(
            **shared_config,
            num_layers=args.san_layers,
            num_query_heads=args.attention_heads,
            num_kv_heads=args.san_kv_heads,
            rms_norm_eps=args.san_rms_norm_eps,
            rope_base=args.san_rope_base,
        )
        default_output_dir = Path("runs/kiwilm-san")
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


def _sft_command(args: argparse.Namespace) -> int:
    data = PreparedSFTData(args.data_dir, seed=args.seed)
    source_checkpoint = args.init_from or args.resume
    model_config = _checkpoint_model_config(source_checkpoint)
    if model_config.vocab_size != data.tokenizer.vocab_size:
        raise ValueError(
            "SFT tokenizer vocabulary does not match the source checkpoint"
        )
    output_dir = args.output_dir or Path(
        f"runs/{model_config.architecture.replace('_', '-')}-sft"
    )
    settings = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_steps=0,
        max_tokens=args.max_tokens,
        warmup_tokens=args.warmup_tokens,
        batch_mode="sft",
        eval_mode="sft",
        precision=args.precision,
        weight_decay=args.weight_decay,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        sample_prompt=data.format_prompt(args.sample_prompt),
        sample_tokens=args.sample_tokens,
        seed=args.seed,
    )
    summary = train(
        model_config,
        data,
        output_dir,
        settings,
        device=args.device,
        resume_from=args.resume,
        init_from=args.init_from,
    )
    _print_json(summary)
    return 0


def _cpt_command(args: argparse.Namespace) -> int:
    data = PreparedTokenData(args.data_dir, seed=args.seed)
    source_checkpoint = args.init_from or args.resume
    model_config = _checkpoint_model_config(source_checkpoint)
    if model_config.vocab_size != data.tokenizer.vocab_size:
        raise ValueError(
            "CPT tokenizer vocabulary does not match the source checkpoint"
        )
    if args.init_from is not None:
        tokenizer_metadata = data.metadata.get("tokenizer")
        reused_from = (
            tokenizer_metadata.get("reused_from")
            if isinstance(tokenizer_metadata, Mapping)
            else None
        )
        source_fingerprint = _checkpoint_data_fingerprint(args.init_from)
        if (
            not isinstance(reused_from, Mapping)
            or reused_from.get("dataset_fingerprint") != source_fingerprint
        ):
            raise ValueError(
                "CPT data must reuse the tokenizer from the source checkpoint's "
                "prepared dataset"
            )
    output_dir = args.output_dir or Path(
        f"runs/{model_config.architecture.replace('_', '-')}-simplestories-cpt"
    )
    settings = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_steps=0,
        max_tokens=args.max_tokens,
        warmup_tokens=args.warmup_tokens,
        batch_mode="story",
        eval_mode="both",
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
        settings,
        device=args.device,
        resume_from=args.resume,
        init_from=args.init_from,
    )
    _print_json(summary)
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    data = load_prepared_data(args.data_dir, seed=args.seed)
    device = choose_device(args.device)
    checkpoint_data_fingerprint = _checkpoint_data_fingerprint(args.checkpoint)
    data_mismatch = checkpoint_data_fingerprint != data.fingerprint
    model, config = load_trained_model(
        args.checkpoint,
        data_fingerprint=None if args.allow_data_mismatch else data.fingerprint,
        device=device,
    )
    if config.vocab_size != data.tokenizer.vocab_size:
        raise ValueError(
            "evaluation tokenizer vocabulary does not match the checkpoint"
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
            "checkpoint_data_fingerprint": checkpoint_data_fingerprint,
            "data_fingerprint": data.fingerprint,
            "data_mismatch": data_mismatch,
            "device": str(device),
            **metrics,
        }
    )
    return 0


def _generate_command(args: argparse.Namespace) -> int:
    device = choose_device(args.device)
    bundled_tokenizer = _bundled_tokenizer_path(args.checkpoint)
    data = (
        None
        if bundled_tokenizer is not None
        else load_prepared_data(args.data_dir, seed=args.seed)
    )
    model, config = load_trained_model(
        args.checkpoint,
        data_fingerprint=None if data is None else data.fingerprint,
        device=device,
    )
    tokenizer = (
        ByteBPETokenizer.load(bundled_tokenizer)
        if bundled_tokenizer is not None
        else data.tokenizer
    )
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("generation tokenizer vocabulary does not match the model")
    generation_options = {
        "max_new_tokens": args.max_new_tokens,
        "context_length": config.context_length,
        "temperature": args.temperature,
        "top_k": None if args.top_k == 0 else args.top_k,
        "seed": args.seed,
        "device": device,
        "cache": args.cache,
    }
    prompt = (
        data.format_prompt(args.prompt)
        if data is not None and isinstance(data, PreparedSFTData)
        else args.prompt
    )
    if args.stream:
        for chunk in generate_stream(
            model,
            tokenizer,
            prompt,
            **generation_options,
        ):
            print(chunk, end="", flush=True)
        print()
    else:
        text = generate(
            model,
            tokenizer,
            prompt,
            **generation_options,
        )
        print(text)
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    data = load_prepared_data(args.data_dir, seed=args.seed)
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


def _sft_report_command(args: argparse.Namespace) -> int:
    data = PreparedSFTData(args.data_dir)
    device = choose_device(args.device)
    summary = generate_sft_adherence_report(
        args.checkpoints,
        data=data,
        suite_path=args.suite,
        output_dir=args.output_dir,
        device=device,
        labels=args.labels,
        cache=args.cache,
    )
    _print_json(summary)
    return 0


def _checkpoint_model_config(path: Path) -> ModelConfig:
    if path.is_dir() or path.suffix == ".safetensors":
        metadata = read_safetensors_metadata(path)
        try:
            serialized = json.loads(metadata["model_config"])
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(
                "Safetensors metadata has an invalid model configuration"
            ) from error
        if not isinstance(serialized, dict):
            raise ValueError("Safetensors model configuration must be an object")
        return ModelConfig.from_dict(serialized)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    serialized = payload.get("model_config")
    if not isinstance(serialized, dict):
        raise ValueError("checkpoint does not contain a model configuration")
    return ModelConfig.from_dict(serialized)


def _checkpoint_data_fingerprint(path: Path) -> str:
    if path.is_dir() or path.suffix == ".safetensors":
        metadata = read_safetensors_metadata(path)
        fingerprint = metadata.get("data_fingerprint")
        if not isinstance(fingerprint, str):
            raise ValueError("Safetensors metadata lacks a data fingerprint")
        return fingerprint
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    fingerprint = payload.get("data_fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("checkpoint does not contain a data fingerprint")
    return fingerprint


def _bundled_tokenizer_path(checkpoint: Path) -> Path | None:
    candidate = (
        checkpoint / "tokenizer.json"
        if checkpoint.is_dir()
        else checkpoint.parent / "tokenizer.json"
    )
    return candidate if candidate.is_file() else None


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


__all__ = ["build_parser", "main"]
