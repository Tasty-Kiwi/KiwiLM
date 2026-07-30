"""CLI parser and checkpoint-loading coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import kiwilm.cli as cli
from kiwilm.checkpoint import CheckpointCompatibilityError, save_checkpoint
from kiwilm.cli import _load_trained_model, build_parser
from kiwilm.config import (
    CNNAttentionConfig,
    CNNFFNAttentionConfig,
    GatedCNNConfig,
    ModelXConfig,
    ModelYConfig,
    ModelZParallelConfig,
    TransformerConfig,
)
from kiwilm.models import build_model


def test_cli_defaults_select_fast_smoke_profile() -> None:
    parser = build_parser()
    prepare = parser.parse_args(["prepare"])
    prepare_instruct = parser.parse_args(
        ["prepare-instruct", "--tokenizer-from", "data/tinystories-750k"]
    )
    training = parser.parse_args(["train"])

    assert prepare.train_limit == 25_000
    assert prepare.validation_limit == 2_000
    assert prepare.vocab_size == 8_192
    assert prepare.tokenizer_from is None
    assert prepare_instruct.train_limit == 50_000
    assert prepare_instruct.validation_limit == 5_000
    assert prepare_instruct.tokenizer_from == Path("data/tinystories-750k")
    assert prepare_instruct.sft_format == "v1"
    assert prepare_instruct.required_word_weight == 3.0
    prepare_instruct_v2 = parser.parse_args(
        [
            "prepare-instruct",
            "--tokenizer-from",
            "data/tinystories-750k",
            "--sft-format",
            "v2",
            "--required-word-weight",
            "4",
        ]
    )
    assert prepare_instruct_v2.sft_format == "v2"
    assert prepare_instruct_v2.required_word_weight == 4.0
    exported = parser.parse_args(
        [
            "export-tokenizer",
            "--data-dir",
            "data/source",
            "--output-dir",
            "data/bundle",
        ]
    )
    assert exported.data_dir == Path("data/source")
    assert exported.output_dir == Path("data/bundle")
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


def test_cli_sft_defaults_and_requires_checkpoint() -> None:
    parser = build_parser()
    sft = parser.parse_args(["sft", "--init-from", "runs/model-x-750k/best.pt"])
    report = parser.parse_args(
        [
            "sft-report",
            "--checkpoints",
            "runs/model-x-instruct/latest.pt",
        ]
    )

    assert sft.data_dir == Path("data/tinystories-instruct-50k")
    assert sft.init_from == Path("runs/model-x-750k/best.pt")
    assert sft.resume is None
    assert sft.max_tokens == 10_000_000
    assert sft.warmup_tokens == 250_000
    assert sft.batch_size == 8
    assert sft.grad_accum_steps == 4
    assert sft.precision == "auto"
    assert report.data_dir == Path("data/tinystories-instruct-50k")
    assert report.suite == Path("eval/instruction-adherence-prompts.json")
    assert report.output_dir == Path("examples/comparisons/sft-adherence")
    assert report.cache == "off"

    with pytest.raises(SystemExit):
        parser.parse_args(["sft"])


def test_sft_cli_uses_checkpoint_architecture_and_weight_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    model_config = ModelXConfig(
        vocab_size=64,
        context_length=8,
        d_model=16,
        dropout=0.0,
        num_heads=2,
        swiglu_dim=32,
    )

    class DummyTokenizer:
        vocab_size = 64

    class DummyData:
        tokenizer = DummyTokenizer()

        @staticmethod
        def format_prompt(prompt: str) -> str:
            return "v2:" + prompt

    def fake_train(
        received_config: object,
        _data: object,
        output_dir: Path,
        settings: object,
        **kwargs: object,
    ) -> dict[str, object]:
        captured["model_config"] = received_config
        captured["output_dir"] = output_dir
        captured["settings"] = settings
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "PreparedSFTData", lambda *_args, **_kwargs: DummyData())
    monkeypatch.setattr(cli, "_checkpoint_model_config", lambda _path: model_config)
    monkeypatch.setattr(cli, "train", fake_train)
    args = build_parser().parse_args(
        ["sft", "--init-from", "runs/model-x-750k/best.pt"]
    )

    assert args.handler(args) == 0
    assert captured["model_config"] == model_config
    assert captured["output_dir"] == Path("runs/model-x-sft")
    assert captured["init_from"] == Path("runs/model-x-750k/best.pt")
    assert captured["resume_from"] is None
    settings = captured["settings"]
    assert settings.batch_mode == "sft"
    assert settings.eval_mode == "sft"
    assert settings.sample_prompt.startswith("v2:")


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


def test_generate_applies_prepared_sft_prompt_format(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, str] = {}

    class DummySFTData:
        fingerprint = "a" * 64
        tokenizer = object()

        @staticmethod
        def format_prompt(prompt: str) -> str:
            return "instruction\n" + prompt

    monkeypatch.setattr(cli, "PreparedSFTData", DummySFTData)
    monkeypatch.setattr(cli, "load_prepared_data", lambda *_args, **_kwargs: DummySFTData())
    monkeypatch.setattr(
        cli,
        "load_trained_model",
        lambda *_args, **_kwargs: (
            torch.nn.Identity(),
            type("Config", (), {"context_length": 8})(),
        ),
    )

    def fake_generate(_model: object, _tokenizer: object, prompt: str, **_kwargs: object) -> str:
        captured["prompt"] = prompt
        return "story"

    monkeypatch.setattr(cli, "generate", fake_generate)
    args = build_parser().parse_args(
        ["generate", "--checkpoint", "model.pt", "--prompt", "Features: Dialogue"]
    )

    assert args.handler(args) == 0
    assert captured["prompt"] == "instruction\nFeatures: Dialogue"
    assert capsys.readouterr().out == "story\n"


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
    assert comparison.output_dir == Path(
        "examples/comparisons/model-a-vs-model-b"
    )


def test_cli_accepts_models_c_d_and_n_way_comparison() -> None:
    parser = build_parser()
    model_c = parser.parse_args(["train", "--architecture", "cnn_dual_attention"])
    model_g = parser.parse_args(["train", "--architecture", "cnn_attention_ffn"])
    model_d = parser.parse_args(["train", "--architecture", "cnn_attention_mamba"])
    model_e = parser.parse_args(
        ["train", "--architecture", "cnn_interleaved_attention"]
    )
    model_f = parser.parse_args(
        ["train", "--architecture", "cnn_deep_interleaved_attention"]
    )
    transformer = parser.parse_args(["train", "--architecture", "transformer"])
    model_y = parser.parse_args(["train", "--architecture", "model_y"])
    model_x = parser.parse_args(["train", "--architecture", "model_x"])
    model_z = parser.parse_args(["train", "--architecture", "model_z_parallel"])
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
    assert model_g.architecture == "cnn_attention_ffn"
    assert model_d.mamba_inner_dim == 896
    assert model_d.mamba_state_dim == 16
    assert model_e.architecture == "cnn_interleaved_attention"
    assert model_f.architecture == "cnn_deep_interleaved_attention"
    assert transformer.architecture == "transformer"
    assert model_y.architecture == "model_y"
    assert model_y.model_y_swiglu_dim == 720
    assert model_x.architecture == "model_x"
    assert model_x.swiglu_dim == 640
    assert model_z.architecture == "model_z_parallel"
    assert model_z.model_z_swiglu_dim == 1280
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


def test_cli_loads_transformer_checkpoint(tmp_path: Path) -> None:
    config = TransformerConfig(
        vocab_size=64,
        context_length=8,
        d_model=16,
        dropout=0.0,
        num_layers=2,
        num_heads=2,
        feedforward_dim=32,
    )
    checkpoint = save_checkpoint(
        tmp_path / "transformer.pt",
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


def test_transformer_cli_uses_default_output_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        vocab_size = 64

    class DummyData:
        tokenizer = DummyTokenizer()

    def fake_train(
        model_config: object,
        _data: object,
        output_dir: Path,
        _train_config: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        captured["model_config"] = model_config
        captured["output_dir"] = output_dir
        return {}

    monkeypatch.setattr(cli, "PreparedTokenData", lambda *_args, **_kwargs: DummyData())
    monkeypatch.setattr(cli, "train", fake_train)
    args = cli.build_parser().parse_args(
        ["train", "--architecture", "transformer", "--max-steps", "1"]
    )

    assert args.handler(args) == 0
    assert isinstance(captured["model_config"], TransformerConfig)
    assert captured["output_dir"] == Path("runs/transformer")


def test_model_g_cli_uses_default_output_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        vocab_size = 64

    class DummyData:
        tokenizer = DummyTokenizer()

    def fake_train(
        model_config: object,
        _data: object,
        output_dir: Path,
        _train_config: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        captured["model_config"] = model_config
        captured["output_dir"] = output_dir
        return {}

    monkeypatch.setattr(cli, "PreparedTokenData", lambda *_args, **_kwargs: DummyData())
    monkeypatch.setattr(cli, "train", fake_train)
    args = cli.build_parser().parse_args(
        ["train", "--architecture", "cnn_attention_ffn", "--max-steps", "1"]
    )

    assert args.handler(args) == 0
    assert isinstance(captured["model_config"], CNNFFNAttentionConfig)
    assert captured["output_dir"] == Path("runs/model-g")


def test_model_x_cli_uses_default_output_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        vocab_size = 64

    class DummyData:
        tokenizer = DummyTokenizer()

    def fake_train(
        model_config: object,
        _data: object,
        output_dir: Path,
        _train_config: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        captured["model_config"] = model_config
        captured["output_dir"] = output_dir
        return {}

    monkeypatch.setattr(cli, "PreparedTokenData", lambda *_args, **_kwargs: DummyData())
    monkeypatch.setattr(cli, "train", fake_train)
    args = cli.build_parser().parse_args(
        ["train", "--architecture", "model_x", "--max-steps", "1"]
    )

    assert args.handler(args) == 0
    assert isinstance(captured["model_config"], ModelXConfig)
    assert captured["output_dir"] == Path("runs/model-x")


def test_model_y_cli_uses_default_output_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        vocab_size = 64

    class DummyData:
        tokenizer = DummyTokenizer()

    def fake_train(
        model_config: object,
        _data: object,
        output_dir: Path,
        _train_config: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        captured["model_config"] = model_config
        captured["output_dir"] = output_dir
        return {}

    monkeypatch.setattr(cli, "PreparedTokenData", lambda *_args, **_kwargs: DummyData())
    monkeypatch.setattr(cli, "train", fake_train)
    args = cli.build_parser().parse_args(
        ["train", "--architecture", "model_y", "--max-steps", "1"]
    )

    assert args.handler(args) == 0
    assert isinstance(captured["model_config"], ModelYConfig)
    assert captured["model_config"].swiglu_dim == 720
    assert captured["output_dir"] == Path("runs/model-y")


def test_model_z_parallel_cli_uses_default_output_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        vocab_size = 64

    class DummyData:
        tokenizer = DummyTokenizer()

    def fake_train(
        model_config: object,
        _data: object,
        output_dir: Path,
        _train_config: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        captured["model_config"] = model_config
        captured["output_dir"] = output_dir
        return {}

    monkeypatch.setattr(cli, "PreparedTokenData", lambda *_args, **_kwargs: DummyData())
    monkeypatch.setattr(cli, "train", fake_train)
    args = cli.build_parser().parse_args(
        ["train", "--architecture", "model_z_parallel", "--max-steps", "1"]
    )

    assert args.handler(args) == 0
    assert isinstance(captured["model_config"], ModelZParallelConfig)
    assert captured["model_config"].swiglu_dim == 1280
    assert captured["output_dir"] == Path("runs/model-z-parallel")
