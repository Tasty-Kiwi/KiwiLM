"""Coverage for the attention-only KiwiLM-SAN architecture."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.cli import build_parser
from kiwilm.config import KiwiLMSANConfig, ModelConfig
from kiwilm.generation import generate_token_stream, generate_tokens
from kiwilm.inference import load_trained_model
from kiwilm.models import (
    GroupedQueryAttention,
    KiwiLMSANBlock,
    KiwiLMSANCache,
    KiwiLMSANLM,
    ZeroCenteredRMSNorm,
    build_model,
)


def small_config(**overrides: object) -> KiwiLMSANConfig:
    values: dict[str, object] = {
        "vocab_size": 67,
        "context_length": 12,
        "d_model": 16,
        "dropout": 0.0,
        "num_layers": 3,
        "num_query_heads": 4,
        "num_kv_heads": 2,
    }
    values.update(overrides)
    return KiwiLMSANConfig(**values)


def test_default_recipe_parameter_count_and_initialization() -> None:
    torch.manual_seed(83)
    config = KiwiLMSANConfig()
    model = build_model(config)
    logits = model(torch.randint(config.vocab_size, (1, 8)))

    assert logits.shape == (1, 8, config.vocab_size)
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_260_560
    assert isinstance(model, KiwiLMSANLM)
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.embedding_scale == math.sqrt(config.d_model)
    assert len(model.blocks) == 16
    assert all(isinstance(block, KiwiLMSANBlock) for block in model.blocks)
    assert all(isinstance(block.attention, GroupedQueryAttention) for block in model.blocks)
    assert all(block.attention.num_query_heads == 8 for block in model.blocks)
    assert all(block.attention.num_kv_heads == 4 for block in model.blocks)
    assert all(block.attention.query_projection.bias is None for block in model.blocks)
    assert all(block.attention.key_projection.bias is None for block in model.blocks)
    assert all(block.attention.value_projection.bias is None for block in model.blocks)
    assert all(block.attention.output_projection.bias is None for block in model.blocks)
    assert all(block.residual_gate.item() == 0.0 for block in model.blocks)
    assert all(
        bool(torch.count_nonzero(norm.weight) == 0)
        for block in model.blocks
        for norm in (
            block.pre_norm,
            block.sandwich_norm,
            block.attention.query_norm,
            block.attention.key_norm,
        )
    )
    assert bool(torch.count_nonzero(model.final_norm.weight) == 0)
    expected_output_std = 0.02 / math.sqrt(2 * config.num_layers)
    for block in model.blocks:
        assert block.attention.output_projection.weight.std().item() == pytest.approx(
            expected_output_std,
            rel=0.03,
        )


def test_zero_centered_rms_norm_starts_at_unit_gain() -> None:
    norm = ZeroCenteredRMSNorm(4, eps=1e-6)
    values = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    expected = values / torch.sqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6)

    torch.testing.assert_close(norm(values), expected)
    assert norm.weight.requires_grad


def test_model_is_strictly_causal_deterministic_and_supports_backward() -> None:
    torch.manual_seed(89)
    config = small_config(dropout=0.5)
    model = KiwiLMSANLM(config).eval()
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
        num_query_heads=2,
        num_kv_heads=1,
        rms_norm_eps=1e-5,
        rope_base=20_000.0,
        tie_embeddings=False,
    )
    serialized = config.to_dict()

    assert json.loads(json.dumps(serialized)) == serialized
    assert KiwiLMSANConfig.from_dict(serialized) == config
    assert ModelConfig.from_dict(serialized) == config
    with pytest.raises(ValueError, match="architecture"):
        small_config(architecture="model_y")
    with pytest.raises(ValueError, match="num_layers"):
        small_config(num_layers=0)
    with pytest.raises(ValueError, match="num_query_heads"):
        small_config(num_query_heads=0)
    with pytest.raises(ValueError, match="num_kv_heads"):
        small_config(num_kv_heads=0)
    with pytest.raises(ValueError, match="d_model"):
        small_config(d_model=18, num_query_heads=4)
    with pytest.raises(ValueError, match="divisible by num_kv_heads"):
        small_config(num_query_heads=4, num_kv_heads=3)
    with pytest.raises(ValueError, match="even"):
        small_config(d_model=24, num_query_heads=8, num_kv_heads=4)
    with pytest.raises(ValueError, match="rms_norm_eps"):
        small_config(rms_norm_eps=0)
    with pytest.raises(ValueError, match="rope_base"):
        small_config(rope_base=float("inf"))


def test_cache_matches_forward_generation_stream_and_rollover() -> None:
    torch.manual_seed(97)
    config = small_config(context_length=8)
    model = KiwiLMSANLM(config).eval()
    prompt = torch.randint(config.vocab_size, (2, 4))

    logits, cache = model.prefill(prompt)
    torch.testing.assert_close(logits, model(prompt))
    assert len(cache.attention) == config.num_layers
    for key, value in cache.attention:
        assert key.shape == (2, config.num_kv_heads, 4, 4)
        assert value.shape == (2, config.num_kv_heads, 4, 4)

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
    model = KiwiLMSANLM(small_config()).eval()
    _, cache = model.prefill(torch.randint(67, (2, 4)))

    with pytest.raises(ValueError, match="incompatible structure"):
        model.decode_step(
            torch.randint(67, (2, 1)),
            KiwiLMSANCache(
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
            KiwiLMSANCache(
                token_ids=cache.token_ids,
                attention=malformed_attention,
            ),
        )


def test_checkpoint_reconstruction_on_available_devices(tmp_path: Path) -> None:
    config = small_config(context_length=8, num_layers=1)
    checkpoint = save_checkpoint(
        tmp_path / "kiwilm-san.pt",
        model=build_model(config),
        step=1,
        model_config=config,
        data_fingerprint="a" * 64,
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
            data_fingerprint="a" * 64,
            device=device,
        )
        input_ids = torch.randint(config.vocab_size, (2, 8), device=device)
        loss = model(input_ids).float().mean()
        loss.backward()

        assert loaded_config == config
        assert torch.isfinite(loss)


def test_cli_exposes_san_training_defaults() -> None:
    args = build_parser().parse_args(["train", "--architecture", "kiwilm_san"])

    assert args.architecture == "kiwilm_san"
    assert args.san_layers == 16
    assert args.attention_heads == 8
    assert args.san_kv_heads == 4
    assert args.san_rms_norm_eps == 1e-6
    assert args.san_rope_base == 10_000.0
