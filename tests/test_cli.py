"""CLI coverage for the active KiwiLM 2 workflows."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import kiwilm.cli as cli
from kiwilm.cli import build_parser
from kiwilm.config import (
    KiwiLM2Config,
    KiwiLM2SlimConfig,
    KiwiLM2SlimV3Config,
    ModelConfig,
)


def test_cli_exposes_only_active_architectures_and_clean_model_flags() -> None:
    parser = build_parser()
    dense = parser.parse_args(["train"])
    slim = parser.parse_args(
        [
            "train",
            "--architecture",
            "kiwilm2_slim",
            "--context-length",
            "256",
            "--d-model",
            "256",
            "--query-heads",
            "8",
            "--kv-heads",
            "4",
            "--swiglu-dim",
            "768",
            "--bigram-buckets",
            "8192",
            "--trigram-buckets",
            "4096",
        ]
    )
    slim_v3 = parser.parse_args(
        [
            "train",
            "--architecture",
            "kiwilm2_slim_v3",
            "--upper-swiglu-blocks",
            "3",
            "--swiglu-residual-gate-init",
            "0.25",
        ]
    )

    assert dense.architecture == "kiwilm2"
    assert dense.context_length == 512
    assert dense.d_model == 512
    assert dense.dropout == 0.0
    assert dense.query_heads == 8
    assert dense.kv_heads == 2
    assert dense.swiglu_dim == 1_536
    assert dense.compile_mode == "eager"
    assert slim.architecture == "kiwilm2_slim"
    assert slim.context_length == 256
    assert slim.kv_heads == 4
    assert slim.bigram_buckets == 8_192
    assert slim.trigram_buckets == 4_096
    assert slim.compile_mode == "eager"
    assert slim_v3.architecture == "kiwilm2_slim_v3"
    assert slim_v3.upper_swiglu_blocks == 3
    assert slim_v3.swiglu_residual_gate_init == 0.25

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--architecture", "historical_model"])
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--kiwilm2-context-length", "256"])


def test_cli_keeps_generic_data_training_and_evaluation_commands() -> None:
    parser = build_parser()
    expected = {
        "prepare",
        "prepare-smollm",
        "prepare-simplestories",
        "prepare-instruct",
        "export-tokenizer",
        "export-safetensors",
        "train",
        "profile-kiwilm2",
        "cpt",
        "sft",
        "evaluate",
        "generate",
        "compare",
        "sft-report",
    }
    subcommands = next(
        action.choices
        for action in parser._actions
        if getattr(action, "choices", None) is not None
    )
    assert set(subcommands) == expected


def test_generate_allows_explicit_cross_budget_tokenizer_use() -> None:
    args = build_parser().parse_args(
        ["generate", "--checkpoint", "latest.pt", "--prompt", "Hello"]
    )
    assert args.allow_data_mismatch is False

    args = build_parser().parse_args(
        [
            "generate",
            "--checkpoint",
            "latest.pt",
            "--prompt",
            "Hello",
            "--allow-data-mismatch",
        ]
    )
    assert args.allow_data_mismatch is True


def test_cli_sft_and_cpt_keep_checkpoint_initialization_contracts() -> None:
    parser = build_parser()
    sft = parser.parse_args(["sft", "--init-from", "runs/kiwilm2/best.pt"])
    cpt = parser.parse_args(["cpt", "--init-from", "runs/kiwilm2/best.pt"])

    assert sft.init_from == Path("runs/kiwilm2/best.pt")
    assert sft.resume is None
    assert sft.max_tokens == 10_000_000
    assert cpt.init_from == Path("runs/kiwilm2/best.pt")
    assert cpt.resume is None


@pytest.mark.parametrize(
    ("architecture", "config_type", "output_dir"),
    [
        ("kiwilm2", KiwiLM2Config, Path("runs/kiwilm2")),
        ("kiwilm2_slim", KiwiLM2SlimConfig, Path("runs/kiwilm2-slim")),
        (
            "kiwilm2_slim_v3",
            KiwiLM2SlimV3Config,
            Path("runs/kiwilm2-slim-v3"),
        ),
    ],
)
def test_train_command_builds_active_config_and_default_output(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    config_type: type[KiwiLM2Config],
    output_dir: Path,
) -> None:
    captured: dict[str, object] = {}
    fake_data = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=320))
    monkeypatch.setattr(cli, "PreparedTokenData", lambda *_args, **_kwargs: fake_data)

    def fake_train(model_config, data, destination, settings, **kwargs):
        captured.update(
            model_config=model_config,
            data=data,
            destination=destination,
            settings=settings,
            kwargs=kwargs,
        )
        return {"step": 0}

    monkeypatch.setattr(cli, "train", fake_train)

    result = cli.main(
        [
            "train",
            "--architecture",
            architecture,
            "--context-length",
            "16",
            "--d-model",
            "16",
            "--query-heads",
            "2",
            "--kv-heads",
            "1",
            "--swiglu-dim",
            "32",
            "--bigram-buckets",
            "16",
            "--trigram-buckets",
            "16",
            "--max-steps",
            "1",
        ]
    )

    assert result == 0
    model_config = captured["model_config"]
    assert isinstance(model_config, config_type)
    assert model_config.vocab_size == 320
    assert model_config.context_length == 16
    assert captured["destination"] == output_dir
    assert captured["kwargs"]["compile_model"] is False


def test_upper_swiglu_flag_is_restricted_to_slim_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_data = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=320))
    monkeypatch.setattr(cli, "PreparedTokenData", lambda *_args, **_kwargs: fake_data)
    with pytest.raises(ValueError, match="valid only for kiwilm2_slim_v3"):
        cli.main(["train", "--architecture", "kiwilm2", "--upper-swiglu-blocks", "3"])
    with pytest.raises(ValueError, match="valid only for kiwilm2_slim_v3"):
        cli.main(
            [
                "train",
                "--architecture",
                "kiwilm2_slim",
                "--swiglu-residual-gate-init",
                "0.25",
            ]
        )
    with pytest.raises(ValueError, match="valid only for kiwilm2_slim_v3"):
        cli.main(
            [
                "profile-kiwilm2",
                "--architecture",
                "kiwilm2",
                "--swiglu-residual-gate-init",
                "0.5",
            ]
        )


def test_muon_remains_dense_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_data = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=320))
    monkeypatch.setattr(cli, "PreparedTokenData", lambda *_args, **_kwargs: fake_data)
    with pytest.raises(ValueError, match="restricted to kiwilm2"):
        cli.main(
            [
                "train",
                "--architecture",
                "kiwilm2_slim",
                "--optimizer",
                "muon",
            ]
        )
    with pytest.raises(ValueError, match="restricted to kiwilm2"):
        cli.main(
            [
                "train",
                "--architecture",
                "kiwilm2_slim_v3",
                "--optimizer",
                "muon",
            ]
        )


def test_legacy_checkpoint_config_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="legacy branch"):
        ModelConfig.from_dict({"architecture": "historical_model"})
