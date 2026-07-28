"""Coverage for the alternating local/global Model X architecture."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import ModelConfig, ModelXConfig
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.generation import generate_token_stream, generate_tokens
from kiwilm.inference import load_trained_model
from kiwilm.models import (
    ModelXCache,
    ModelXLM,
    ResidualSwiGLUBlock,
    RMSAttentionBlock,
    RMSGatedCNNBlock,
    build_model,
)


def small_config(**overrides: object) -> ModelXConfig:
    values: dict[str, object] = {
        "vocab_size": 67,
        "context_length": 12,
        "d_model": 16,
        "dropout": 0.0,
        "kernel_size": 3,
        "cnn_dilations": (1, 2),
        "num_heads": 2,
        "swiglu_dim": 24,
    }
    values.update(overrides)
    return ModelXConfig(**values)


def test_default_shape_parameter_count_weight_tying_and_structure() -> None:
    config = ModelXConfig()
    model = build_model(config)
    logits = model(torch.randint(config.vocab_size, (1, 8)))

    assert logits.shape == (1, 8, config.vocab_size)
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_387_520
    assert isinstance(model, ModelXLM)
    assert model.lm_head.weight is model.token_embedding.weight
    assert len(model.cnn_blocks) == 2
    assert len(model.attention_blocks) == 2
    assert len(model.feedforward_blocks) == 4
    assert all(isinstance(block, RMSGatedCNNBlock) for block in model.cnn_blocks)
    assert all(isinstance(block, RMSAttentionBlock) for block in model.attention_blocks)
    assert all(isinstance(block, ResidualSwiGLUBlock) for block in model.feedforward_blocks)
    assert isinstance(model.final_norm, nn.RMSNorm)
    assert [block.conv.conv.dilation[0] for block in model.cnn_blocks] == [1, 2]


def test_swiglu_uses_three_bias_free_projections() -> None:
    block = ResidualSwiGLUBlock(
        16,
        hidden_dim=24,
        dropout=0.0,
        rms_norm_eps=1e-5,
    )

    assert block.gate_projection.weight.shape == (24, 16)
    assert block.up_projection.weight.shape == (24, 16)
    assert block.down_projection.weight.shape == (16, 24)
    assert block.gate_projection.bias is None
    assert block.up_projection.bias is None
    assert block.down_projection.bias is None


def test_model_is_strictly_causal_deterministic_and_supports_backward() -> None:
    torch.manual_seed(71)
    config = small_config(dropout=0.5)
    model = ModelXLM(config).eval()
    original = torch.randint(config.vocab_size, (2, config.context_length))
    changed = original.clone()
    changed[:, 7:] = torch.randint(config.vocab_size, (2, 5))

    first = model(original)
    second = model(original)
    changed_logits = model(changed)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[:, :7], changed_logits[:, :7])
    first.mean().backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_config_round_trip_and_validation() -> None:
    config = small_config(
        kernel_size=5,
        cnn_dilations=(1, 3),
        num_heads=4,
        swiglu_dim=40,
        rms_norm_eps=1e-6,
        tie_embeddings=False,
    )
    serialized = config.to_dict()

    assert json.loads(json.dumps(serialized)) == serialized
    assert ModelXConfig.from_dict(serialized) == config
    assert ModelConfig.from_dict(serialized) == config
    with pytest.raises(ValueError, match="architecture"):
        small_config(architecture="transformer")
    with pytest.raises(ValueError, match="cnn_dilations"):
        small_config(cnn_dilations=(1,))
    with pytest.raises(ValueError, match="swiglu_dim"):
        small_config(swiglu_dim=0)
    with pytest.raises(ValueError, match="rms_norm_eps"):
        small_config(rms_norm_eps=0)
    with pytest.raises(ValueError, match="divisible"):
        small_config(d_model=18, num_heads=4)
    with pytest.raises(ValueError, match="even"):
        small_config(d_model=24, num_heads=8)


def test_cache_matches_forward_generation_stream_and_rollover() -> None:
    torch.manual_seed(73)
    config = small_config(context_length=8)
    model = ModelXLM(config).eval()
    prompt = torch.randint(config.vocab_size, (2, 4))

    logits, cache = model.prefill(prompt)
    torch.testing.assert_close(logits, model(prompt))
    assert len(cache.cnn) == 2
    assert len(cache.attention) == 2

    tokens = prompt
    for _ in range(7):
        next_token = torch.randint(config.vocab_size, (2, 1))
        tokens = torch.cat((tokens, next_token), dim=1)
        cached_logits, cache = model.decode_step(next_token, cache)
        expected = model(tokens[:, -config.context_length :])[:, -1:, :]
        torch.testing.assert_close(cached_logits[:, -1:, :], expected)

    uncached = generate_tokens(
        model,
        prompt[:1],
        max_new_tokens=12,
        temperature=0,
        cache="off",
    )
    cached = generate_tokens(
        model,
        prompt[:1],
        max_new_tokens=12,
        temperature=0,
        cache="auto",
    )
    streamed = torch.cat(
        (
            prompt[:1],
            *generate_token_stream(
                model,
                prompt[:1],
                max_new_tokens=12,
                temperature=0,
                cache="auto",
            ),
        ),
        dim=1,
    )
    assert torch.equal(cached, uncached)
    assert torch.equal(streamed, uncached)


def test_decode_step_rejects_malformed_cache() -> None:
    model = ModelXLM(small_config()).eval()
    _, cache = model.prefill(torch.randint(67, (2, 4)))

    with pytest.raises(ValueError, match="incompatible structure"):
        model.decode_step(
            torch.randint(67, (2, 1)),
            ModelXCache(
                token_ids=cache.token_ids,
                cnn=cache.cnn[:-1],
                attention=cache.attention,
            ),
        )

    key, value = cache.attention[0]
    with pytest.raises(ValueError, match="attention cache"):
        model.decode_step(
            torch.randint(67, (2, 1)),
            ModelXCache(
                token_ids=cache.token_ids,
                cnn=cache.cnn,
                attention=[
                    (key[:, :, :-1], value[:, :, :-1]),
                    cache.attention[1],
                ],
            ),
        )


def test_checkpoint_reconstruction_on_available_devices(tmp_path: Path) -> None:
    config = small_config(context_length=8)
    checkpoint = save_checkpoint(
        tmp_path / "model-x.pt",
        model=build_model(config),
        step=1,
        model_config=config,
        data_fingerprint="x" * 64,
    )
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    for device_name in devices:
        device = torch.device(device_name)
        model, loaded_config = load_trained_model(
            checkpoint,
            data_fingerprint="x" * 64,
            device=device,
        )
        input_ids = torch.randint(config.vocab_size, (2, 8), device=device)
        loss = model(input_ids).float().mean()
        loss.backward()

        assert loaded_config == config
        assert torch.isfinite(loss)


def test_reduced_model_x_smoke_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "benchmark"
    prepare_from_stories(
        data_dir,
        [
            "Once there was a small green bird.",
            "The bird found a kind friend.",
        ]
        * 6,
        ["They went home before dark."] * 4,
        vocab_size=300,
        min_frequency=1,
    )
    data = PreparedTokenData(data_dir)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "suite_version": 1,
                "max_new_tokens": 1,
                "sampling_profiles": [
                    {
                        "id": "focused",
                        "temperature": 0,
                        "top_k": 0,
                        "seed": 42,
                    }
                ],
                "prompts": [{"id": "once", "prompt": "Once"}],
            }
        ),
        encoding="utf-8",
    )
    scripts_dir = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    module_path = scripts_dir / "run_model_x_smoke_benchmark.py"
    spec = importlib.util.spec_from_file_location("model_x_smoke", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--suite",
            str(suite),
            "--device",
            "cpu",
            "--expected-data-fingerprint",
            data.fingerprint,
            "--max-steps",
            "1",
            "--batch-size",
            "2",
            "--grad-accum-steps",
            "1",
            "--precision",
            "fp32",
            "--context-length",
            "8",
            "--d-model",
            "8",
            "--attention-heads",
            "1",
            "--feedforward-dim",
            "16",
            "--swiglu-dim",
            "12",
            "--warmup-steps",
            "0",
            "--eval-interval",
            "1",
            "--eval-batches",
            "1",
            "--post-eval-batches",
            "1",
            "--checkpoint-interval",
            "1",
            "--log-interval",
            "0",
            "--sample-tokens",
            "0",
            "--generation-tokens",
            "1",
            "--generation-repeats",
            "1",
        ]
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert summary["models"]["model_b"]["training"]["step"] == 1
    assert summary["models"]["transformer"]["training"]["step"] == 1
    assert summary["models"]["model_x"]["training"]["step"] == 1
    assert summary["comparison"]["generation_count"] == 3
    assert summary["settings"]["training_targets_per_model"] == 16
    assert (output_dir / "comparison" / "report.md").is_file()
    assert (output_dir / "comparison" / "results.jsonl").is_file()
