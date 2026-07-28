"""Coverage for Model G's per-convolution feed-forward architecture."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import CNNFFNAttentionConfig, ModelConfig
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.generation import generate_token_stream, generate_tokens
from kiwilm.inference import load_trained_model
from kiwilm.models import CNNAttentionCache, CNNFFNAttentionLM, build_model


def small_config(**overrides: object) -> CNNFFNAttentionConfig:
    values: dict[str, object] = {
        "vocab_size": 67,
        "context_length": 12,
        "d_model": 16,
        "dropout": 0.0,
        "num_heads": 2,
        "feedforward_dim": 32,
    }
    values.update(overrides)
    return CNNFFNAttentionConfig(**values)


def test_default_shape_parameter_count_weight_tying_and_order() -> None:
    config = CNNFFNAttentionConfig()
    model = build_model(config)
    logits = model(torch.randint(config.vocab_size, (1, 8)))

    assert logits.shape == (1, 8, config.vocab_size)
    assert sum(parameter.numel() for parameter in model.parameters()) == 8_417_536
    assert isinstance(model, CNNFFNAttentionLM)
    assert model.lm_head.weight is model.token_embedding.weight
    assert len(model.pre_attention_blocks) == 3
    assert len(model.pre_attention_ffn_blocks) == 3
    assert len(model.post_attention_blocks) == 3
    assert len(model.post_attention_ffn_blocks) == 3


def test_model_is_strictly_causal_and_supports_finite_backward() -> None:
    torch.manual_seed(59)
    config = small_config()
    model = CNNFFNAttentionLM(config).eval()
    original = torch.randint(config.vocab_size, (2, config.context_length))
    changed = original.clone()
    changed[:, 7:] = torch.randint(config.vocab_size, (2, 5))

    original_logits = model(original)
    changed_logits = model(changed)

    torch.testing.assert_close(original_logits[:, :7], changed_logits[:, :7])
    original_logits.mean().backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_config_round_trip_and_validation() -> None:
    config = small_config(
        kernel_size=5,
        pre_attention_dilations=(1, 3, 9),
        post_attention_dilations=(27, 54, 108),
        tie_embeddings=False,
    )
    serialized = config.to_dict()

    assert json.loads(json.dumps(serialized)) == serialized
    assert CNNFFNAttentionConfig.from_dict(serialized) == config
    assert ModelConfig.from_dict(serialized) == config
    with pytest.raises(ValueError, match="architecture"):
        small_config(architecture="cnn_attention")
    with pytest.raises(ValueError, match="feedforward_dim"):
        small_config(feedforward_dim=0)


def test_cache_matches_forward_generation_stream_and_rollover() -> None:
    torch.manual_seed(61)
    config = small_config(context_length=8)
    model = CNNFFNAttentionLM(config).eval()
    prompt = torch.randint(config.vocab_size, (2, 4))

    logits, cache = model.prefill(prompt)
    torch.testing.assert_close(logits, model(prompt))
    assert len(cache.pre_cnn) == 3
    assert len(cache.post_cnn) == 3

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
    model = CNNFFNAttentionLM(small_config()).eval()
    _, cache = model.prefill(torch.randint(67, (2, 4)))
    malformed = CNNAttentionCache(
        token_ids=cache.token_ids,
        pre_cnn=cache.pre_cnn[:-1],
        attention=cache.attention,
        post_cnn=cache.post_cnn,
    )

    with pytest.raises(ValueError, match="incompatible structure"):
        model.decode_step(torch.randint(67, (2, 1)), malformed)


def test_checkpoint_reconstruction_on_available_devices(tmp_path: Path) -> None:
    config = small_config(context_length=8)
    checkpoint = save_checkpoint(
        tmp_path / "model-g.pt",
        model=build_model(config),
        step=1,
        model_config=config,
        data_fingerprint="g" * 64,
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
            data_fingerprint="g" * 64,
            device=device,
        )
        input_ids = torch.randint(config.vocab_size, (2, 8), device=device)
        loss = model(input_ids).float().mean()
        loss.backward()

        assert loaded_config == config
        assert torch.isfinite(loss)


def test_reduced_model_g_smoke_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "model-g-benchmark"
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
    transformer_runner = _load_script(
        scripts_dir / "run_transformer_smoke_benchmark.py",
        "transformer_smoke_for_model_g",
    )
    common = [
        "--data-dir",
        str(data_dir),
        "--suite",
        str(suite),
        "--device",
        "cpu",
        "--expected-data-fingerprint",
        data.fingerprint,
    ]
    assert (
        transformer_runner.main(
            [
                *common,
                "--output-dir",
                str(reference_dir),
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
        == 0
    )
    model_g_runner = _load_script(
        scripts_dir / "run_model_g_smoke_benchmark.py",
        "model_g_smoke",
    )
    assert (
        model_g_runner.main(
            [
                *common,
                "--reference-dir",
                str(reference_dir),
                "--output-dir",
                str(output_dir),
                "--post-eval-batches",
                "1",
                "--generation-tokens",
                "1",
                "--generation-repeats",
                "1",
            ]
        )
        == 0
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["models"]["model_g"]["training"]["step"] == 1
    assert summary["comparison"]["generation_count"] == 3
    assert (output_dir / "comparison" / "report.md").is_file()


def _load_script(path: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
