"""CLI parser and checkpoint-loading coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import CheckpointCompatibilityError, save_checkpoint
from kiwilm.cli import _load_trained_model, build_parser
from kiwilm.config import GatedCNNConfig
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

