"""CLI parser and checkpoint-loading coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import CheckpointCompatibilityError, save_checkpoint
from kiwilm.cli import _load_trained_model, build_parser
from kiwilm.config import CNNAttentionConfig, GatedCNNConfig
from kiwilm.models import build_model


def test_cli_defaults_select_fast_smoke_profile() -> None:
    parser = build_parser()
    prepare = parser.parse_args(["prepare"])
    training = parser.parse_args(["train"])

    assert prepare.train_limit == 25_000
    assert prepare.validation_limit == 2_000
    assert prepare.vocab_size == 8_192
    assert training.max_steps == 2_000
    assert training.batch_size == 32
    assert training.context_length == 256
    assert training.architecture == "gated_cnn"
    assert training.output_dir is None
    generation = parser.parse_args(
        ["generate", "--checkpoint", "model.pt", "--prompt", "Once"]
    )
    assert not generation.stream
    assert generation.cache == "auto"
    assert training.batch_mode == "packed"
    assert training.precision == "fp32"


def test_cli_accepts_streaming_generation() -> None:
    args = build_parser().parse_args(
        [
            "generate",
            "--checkpoint",
            "model.pt",
            "--prompt",
            "Once",
            "--stream",
        ]
    )

    assert args.stream


def test_cli_selects_model_b_and_comparison_defaults() -> None:
    parser = build_parser()
    training = parser.parse_args(["train", "--architecture", "cnn_attention"])
    comparison = parser.parse_args(
        [
            "compare",
            "--checkpoint-a",
            "a.pt",
            "--checkpoint-b",
            "b.pt",
        ]
    )

    assert training.attention_heads == 8
    assert training.attention_feedforward_dim == 1024
    assert comparison.suite == Path("eval/story-consistency-prompts.json")


def test_cli_accepts_models_c_d_and_n_way_comparison() -> None:
    parser = build_parser()
    model_c = parser.parse_args(["train", "--architecture", "cnn_dual_attention"])
    model_d = parser.parse_args(["train", "--architecture", "cnn_attention_mamba"])
    comparison = parser.parse_args(
        [
            "compare",
            "--checkpoints",
            "b.pt",
            "c.pt",
            "d.pt",
            "--labels",
            "Model B",
            "Model C",
            "Model D",
        ]
    )

    assert model_c.architecture == "cnn_dual_attention"
    assert model_d.mamba_inner_dim == 896
    assert model_d.mamba_state_dim == 16
    assert comparison.checkpoints == [
        Path("b.pt"),
        Path("c.pt"),
        Path("d.pt"),
    ]
    assert comparison.labels == ["Model B", "Model C", "Model D"]


def test_cli_loads_checkpoint_and_rejects_other_data(tmp_path: Path) -> None:
    config = GatedCNNConfig(
        vocab_size=300,
        context_length=8,
        d_model=8,
        dropout=0.0,
        num_layers=1,
        dilations=(1,),
    )
    checkpoint = save_checkpoint(
        tmp_path / "model.pt",
        model=build_model(config),
        step=1,
        model_config=config,
        data_fingerprint="a" * 64,
    )

    model, loaded_config = _load_trained_model(
        checkpoint,
        data_fingerprint="a" * 64,
        device=torch.device("cpu"),
    )
    assert loaded_config == config
    assert model(torch.ones((1, 2), dtype=torch.long)).shape == (1, 2, 300)

    with pytest.raises(CheckpointCompatibilityError, match="fingerprint"):
        _load_trained_model(
            checkpoint,
            data_fingerprint="b" * 64,
            device=torch.device("cpu"),
        )


def test_cli_loads_model_b_checkpoint(tmp_path: Path) -> None:
    config = CNNAttentionConfig(
        vocab_size=64,
        context_length=8,
        d_model=16,
        dropout=0.0,
        num_heads=2,
        feedforward_dim=32,
    )
    checkpoint = save_checkpoint(
        tmp_path / "model-b.pt",
        model=build_model(config),
        step=1,
        model_config=config,
        data_fingerprint="a" * 64,
    )

    model, loaded_config = _load_trained_model(
        checkpoint,
        data_fingerprint="a" * 64,
        device=torch.device("cpu"),
    )

    assert loaded_config == config
    assert model(torch.ones((1, 2), dtype=torch.long)).shape == (1, 2, 64)
