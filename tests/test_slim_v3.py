"""Slim v3 controlled-ablation provenance and promotion coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import KiwiLM2Config, KiwiLM2SlimConfig, KiwiLM2SlimV3Config
from kiwilm.models import build_model
from kiwilm.slim_v3 import (
    select_slim_v3_candidate,
    validate_slim_v3_smoke_checkpoints,
)
from kiwilm.training import TrainConfig, train


class _TinyData:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.fingerprint = "a" * 64
        self.generator = torch.Generator(device="cpu").manual_seed(42)

    def get_batch(
        self,
        split: str,
        *,
        batch_size: int,
        context_length: int,
        device: str | torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert split in {"train", "validation"}
        source = generator if generator is not None else self.generator
        tokens = torch.randint(
            self.vocab_size,
            (batch_size, context_length + 1),
            generator=source,
        )
        return tokens[:, :-1].to(device), tokens[:, 1:].to(device)

    def state_dict(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state["fingerprint"] != self.fingerprint:
            raise ValueError("fingerprint mismatch")
        generator_state = state["generator_state"]
        assert isinstance(generator_state, torch.Tensor)
        self.generator.set_state(generator_state)


def _config(config_type: type[KiwiLM2Config], **overrides: object) -> KiwiLM2Config:
    values: dict[str, object] = {
        "vocab_size": 71,
        "context_length": 8,
        "d_model": 8,
        "num_query_heads": 2,
        "num_kv_heads": 1,
        "swiglu_dim": 12,
        "bigram_buckets": 17,
        "trigram_buckets": 19,
        "conv_kernel_sizes": (3, 5, 3, 5, 3, 5),
    }
    values.update(overrides)
    return config_type(**values)


def _checkpoints(tmp_path: Path) -> dict[str, Path]:
    configs = {
        "dense": _config(KiwiLM2Config),
        "slim_v2": _config(KiwiLM2SlimConfig),
        "h7s3": _config(KiwiLM2SlimV3Config, upper_swiglu_blocks=3),
        "h6s4": _config(KiwiLM2SlimV3Config, upper_swiglu_blocks=4),
    }
    train_config = {
        "max_steps": 3152,
        "batch_size": 8,
        "grad_accum_steps": 4,
        "lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_tokens": 1_000_000,
        "max_tokens": 50_000_000,
        "batch_mode": "packed",
        "precision": "bf16",
        "weight_decay": 0.1,
        "beta2": 0.95,
        "optimizer": "adamw",
        "grad_clip": 1.0,
        "seed": 42,
    }
    return {
        role: save_checkpoint(
            tmp_path / f"{role}.pt",
            model=build_model(config),
            step=3052,
            model_config=config,
            train_config=train_config,
            data_fingerprint="a" * 64,
            training_state={"tokens_seen": 50_000_000},
        )
        for role, config in configs.items()
    }


def test_smoke_checkpoint_provenance_accepts_only_the_frozen_four_way_run(
    tmp_path: Path,
) -> None:
    checkpoints = _checkpoints(tmp_path)
    result = validate_slim_v3_smoke_checkpoints(
        checkpoints,
        data_fingerprint="a" * 64,
        tokenizer_vocab_size=71,
    )
    assert result["checkpoints"]["h7s3"]["upper_swiglu_blocks"] == 3
    assert result["checkpoints"]["h6s4"]["upper_swiglu_blocks"] == 4
    with pytest.raises(ValueError, match="fingerprint"):
        validate_slim_v3_smoke_checkpoints(
            checkpoints,
            data_fingerprint="b" * 64,
            tokenizer_vocab_size=71,
        )


def test_slim_v3_selection_applies_both_gates() -> None:
    selected = select_slim_v3_candidate(
        dense_tokens_per_second=20_000,
        h7s3_validation_loss=4.80,
        h6s4_validation_loss=4.76,
        h6s4_tokens_per_second=22_500,
        h7s3_health_passed=True,
        h6s4_health_passed=True,
        h7s3_parity_passed=True,
        h6s4_parity_passed=True,
    )
    assert selected["selected"] == "h6s4"
    assert selected["loss_gate"]["passed"] is True
    assert selected["throughput_gate"]["passed"] is True

    cheaper = select_slim_v3_candidate(
        dense_tokens_per_second=20_000,
        h7s3_validation_loss=4.80,
        h6s4_validation_loss=4.78,
        h6s4_tokens_per_second=23_000,
        h7s3_health_passed=True,
        h6s4_health_passed=True,
        h7s3_parity_passed=True,
        h6s4_parity_passed=True,
    )
    assert cheaper["selected"] == "h7s3"

    blocked = select_slim_v3_candidate(
        dense_tokens_per_second=20_000,
        h7s3_validation_loss=4.80,
        h6s4_validation_loss=None,
        h6s4_tokens_per_second=23_000,
        h7s3_health_passed=True,
        h6s4_health_passed=True,
        h7s3_parity_passed=True,
        h6s4_parity_passed=True,
    )
    assert blocked["selected"] is None


@pytest.mark.parametrize(
    ("upper_swiglu_blocks", "gate_init"),
    [(3, None), (4, None), (4, 0.25), (4, 0.5)],
)
def test_slim_v3_tiny_training_and_resume(
    tmp_path: Path,
    upper_swiglu_blocks: int,
    gate_init: float | None,
) -> None:
    config = _config(
        KiwiLM2SlimV3Config,
        upper_swiglu_blocks=upper_swiglu_blocks,
        swiglu_residual_gate_init=gate_init,
    )
    settings = TrainConfig(
        max_steps=1,
        batch_size=1,
        eval_interval=0,
        checkpoint_interval=1,
        log_interval=0,
        sample_tokens=0,
    )
    first = train(
        config,
        _TinyData(config.vocab_size),
        tmp_path / "first",
        settings,
        device="cpu",
        log_fn=None,
    )
    resumed = train(
        config,
        _TinyData(config.vocab_size),
        tmp_path / "resumed",
        TrainConfig(**settings.to_dict()),
        device="cpu",
        resume_from=first["latest_checkpoint"],
        log_fn=None,
    )
    assert resumed["step"] == 1
    assert Path(resumed["latest_checkpoint"]).is_file()
