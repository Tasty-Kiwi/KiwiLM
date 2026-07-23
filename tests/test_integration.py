"""Offline prepare-to-generation integration coverage."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from kiwilm.cli import _load_trained_model
from kiwilm.config import CNNAttentionConfig, GatedCNNConfig
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.generation import generate
from kiwilm.training import TrainConfig, evaluate, train


def test_offline_prepare_train_reload_evaluate_and_generate(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    prepare_from_stories(
        data_dir,
        [
            "Once there was a small green bird.",
            "The bird found a warm and friendly home.",
            "A child planted a tree beside the river.",
        ]
        * 4,
        ["The little fox helped its friend.", "They went home before dark."] * 3,
        vocab_size=300,
        min_frequency=1,
    )
    data = PreparedTokenData(data_dir, seed=5)
    config = GatedCNNConfig(
        vocab_size=data.tokenizer.vocab_size,
        context_length=8,
        d_model=8,
        dropout=0.0,
        num_layers=1,
        dilations=(1,),
    )
    summary = train(
        config,
        data,
        run_dir,
        TrainConfig(
            max_steps=1,
            batch_size=2,
            lr=1e-3,
            min_lr=1e-3,
            warmup_steps=0,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            log_interval=0,
            sample_tokens=2,
            seed=5,
        ),
        device="cpu",
        log_fn=None,
    )

    checkpoint = summary["best_checkpoint"]
    assert checkpoint is not None
    model, loaded_config = _load_trained_model(
        checkpoint,
        data_fingerprint=data.fingerprint,
        device=torch.device("cpu"),
    )
    metrics = evaluate(
        model,
        data,
        batch_size=2,
        context_length=loaded_config.context_length,
        num_batches=1,
        device="cpu",
        generator=torch.Generator().manual_seed(9),
    )
    sample = generate(
        model,
        data.tokenizer,
        "Once",
        max_new_tokens=2,
        temperature=0,
        device="cpu",
    )

    assert math.isfinite(metrics["validation_loss"])
    assert math.isfinite(metrics["perplexity"])
    assert isinstance(summary["sample"], str)
    assert isinstance(sample, str)
    assert sample


def test_model_b_one_step_training_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    prepare_from_stories(
        data_dir,
        ["Once there was a small green bird.", "The bird found a friend."] * 4,
        ["They went home before dark."] * 3,
        vocab_size=300,
        min_frequency=1,
    )
    data = PreparedTokenData(data_dir, seed=11)
    config = CNNAttentionConfig(
        vocab_size=data.tokenizer.vocab_size,
        context_length=8,
        d_model=8,
        dropout=0.0,
        num_heads=1,
        feedforward_dim=16,
    )
    summary = train(
        config,
        data,
        tmp_path / "run",
        TrainConfig(
            max_steps=1,
            batch_size=2,
            lr=1e-3,
            min_lr=1e-3,
            warmup_steps=0,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            log_interval=0,
            sample_tokens=0,
            seed=11,
        ),
        device="cpu",
        log_fn=None,
    )

    assert summary["step"] == 1
    assert summary["best_checkpoint"] is not None
