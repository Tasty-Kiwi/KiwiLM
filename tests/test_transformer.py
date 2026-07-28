"""Coverage for the controlled GPT-style Transformer baseline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import ModelConfig, TransformerConfig
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.generation import generate_token_stream, generate_tokens
from kiwilm.inference import load_trained_model
from kiwilm.models import TransformerCache, TransformerLM, build_model


def small_config(**overrides: object) -> TransformerConfig:
    values: dict[str, object] = {
        "vocab_size": 67,
        "context_length": 12,
        "d_model": 16,
        "dropout": 0.0,
        "num_layers": 4,
        "num_heads": 2,
        "feedforward_dim": 32,
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_default_shape_parameter_count_weight_tying_and_depth() -> None:
    config = TransformerConfig()
    model = build_model(config)
    logits = model(torch.randint(config.vocab_size, (1, 8)))

    assert logits.shape == (1, 8, config.vocab_size)
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_264_896
    assert isinstance(model, TransformerLM)
    assert model.lm_head.weight is model.token_embedding.weight
    assert len(model.blocks) == 4


def test_model_is_strictly_causal_deterministic_and_supports_backward() -> None:
    torch.manual_seed(47)
    config = small_config(dropout=0.5)
    model = TransformerLM(config).eval()
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
        num_layers=6,
        num_heads=4,
        feedforward_dim=48,
        tie_embeddings=False,
    )
    serialized = config.to_dict()

    assert json.loads(json.dumps(serialized)) == serialized
    assert TransformerConfig.from_dict(serialized) == config
    assert ModelConfig.from_dict(serialized) == config
    with pytest.raises(ValueError, match="architecture"):
        small_config(architecture="cnn_attention")
    with pytest.raises(ValueError, match="num_layers"):
        small_config(num_layers=0)
    with pytest.raises(ValueError, match="num_heads"):
        small_config(num_heads=0)
    with pytest.raises(ValueError, match="feedforward_dim"):
        small_config(feedforward_dim=0)
    with pytest.raises(ValueError, match="divisible"):
        small_config(d_model=18, num_heads=4)
    with pytest.raises(ValueError, match="even"):
        small_config(d_model=24, num_heads=8)


def test_cache_matches_forward_generation_stream_and_rollover() -> None:
    torch.manual_seed(53)
    config = small_config(context_length=8)
    model = TransformerLM(config).eval()
    prompt = torch.randint(config.vocab_size, (2, 4))

    logits, cache = model.prefill(prompt)
    torch.testing.assert_close(logits, model(prompt))
    assert len(cache.attention) == 4

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
    model = TransformerLM(small_config()).eval()
    _, cache = model.prefill(torch.randint(67, (2, 4)))

    with pytest.raises(ValueError, match="incompatible structure"):
        model.decode_step(
            torch.randint(67, (2, 1)),
            TransformerCache(
                token_ids=cache.token_ids,
                attention=cache.attention[:-1],
            ),
        )

    key, value = cache.attention[0]
    malformed_attention = [
        (key[:, :, :-1], value[:, :, :-1]),
        *cache.attention[1:],
    ]
    with pytest.raises(ValueError, match="attention cache"):
        model.decode_step(
            torch.randint(67, (2, 1)),
            TransformerCache(
                token_ids=cache.token_ids,
                attention=malformed_attention,
            ),
        )


def test_checkpoint_reconstruction_on_available_devices(tmp_path: Path) -> None:
    config = small_config(context_length=8)
    checkpoint = save_checkpoint(
        tmp_path / "transformer.pt",
        model=build_model(config),
        step=1,
        model_config=config,
        data_fingerprint="e" * 64,
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
            data_fingerprint="e" * 64,
            device=device,
        )
        input_ids = torch.randint(config.vocab_size, (2, 8), device=device)
        loss = model(input_ids).float().mean()
        loss.backward()

        assert loaded_config == config
        assert torch.isfinite(loss)


def test_reduced_smoke_benchmark_runner(tmp_path: Path) -> None:
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
    module_path = Path(__file__).parents[1] / "scripts" / "run_transformer_smoke_benchmark.py"
    spec = importlib.util.spec_from_file_location("transformer_smoke", module_path)
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
            "--context-length",
            "8",
            "--d-model",
            "8",
            "--attention-heads",
            "1",
            "--feedforward-dim",
            "16",
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
    assert summary["comparison"]["generation_count"] == 2
    assert (output_dir / "comparison" / "report.md").is_file()
    assert (output_dir / "comparison" / "results.jsonl").is_file()
