"""Coverage for Model F's deep interleaved CNN-attention architecture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import CNNDeepInterleavedAttentionConfig, ModelConfig
from kiwilm.generation import generate_token_stream, generate_tokens
from kiwilm.inference import load_trained_model
from kiwilm.models import (
    CNNDeepInterleavedAttentionLM,
    CNNInterleavedAttentionCache,
    build_model,
)


def small_config(**overrides: object) -> CNNDeepInterleavedAttentionConfig:
    values: dict[str, object] = {
        "vocab_size": 67,
        "context_length": 12,
        "d_model": 16,
        "dropout": 0.0,
        "num_heads": 2,
        "feedforward_dim": 32,
    }
    values.update(overrides)
    return CNNDeepInterleavedAttentionConfig(**values)


def test_default_shape_parameter_count_weight_tying_and_order() -> None:
    config = CNNDeepInterleavedAttentionConfig()
    model = build_model(config)

    logits = model(torch.randint(config.vocab_size, (1, 8)))
    dilations = [
        block.conv.conv.dilation[0]
        for group in model.cnn_groups
        for block in group
    ]

    assert logits.shape == (1, 8, config.vocab_size)
    assert sum(parameter.numel() for parameter in model.parameters()) == 8_023_296
    assert isinstance(model, CNNDeepInterleavedAttentionLM)
    assert model.lm_head.weight is model.token_embedding.weight
    assert [len(group) for group in model.cnn_groups] == [2, 2, 5]
    assert len(model.attention_blocks) == 3
    assert dilations == [1, 2, 4, 8, 16, 32, 1, 2, 4]


def test_model_is_strictly_causal_and_supports_finite_backward() -> None:
    torch.manual_seed(41)
    config = small_config()
    model = CNNDeepInterleavedAttentionLM(config).eval()
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
        dilations=(1, 3, 9, 27, 54, 108),
        refinement_dilations=(1, 5, 25),
        tie_embeddings=False,
    )
    serialized = config.to_dict()

    assert json.loads(json.dumps(serialized)) == serialized
    assert CNNDeepInterleavedAttentionConfig.from_dict(serialized) == config
    assert ModelConfig.from_dict(serialized) == config
    with pytest.raises(ValueError, match="architecture"):
        small_config(architecture="cnn_interleaved_attention")
    with pytest.raises(ValueError, match="exactly 3"):
        small_config(refinement_dilations=(1, 2))
    with pytest.raises(ValueError, match="refinement_dilations"):
        small_config(refinement_dilations=(1, 2, 0))


def test_three_attention_cache_matches_forward_generation_stream_and_rollover() -> None:
    torch.manual_seed(43)
    config = small_config(context_length=8)
    model = CNNDeepInterleavedAttentionLM(config).eval()
    prompt = torch.randint(config.vocab_size, (2, 4))

    logits, cache = model.prefill(prompt)
    torch.testing.assert_close(logits, model(prompt))
    assert [len(group) for group in cache.cnn_groups] == [2, 2, 5]
    assert len(cache.attention) == 3

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
    model = CNNDeepInterleavedAttentionLM(small_config()).eval()
    _, cache = model.prefill(torch.randint(67, (2, 4)))
    malformed = CNNInterleavedAttentionCache(
        token_ids=cache.token_ids,
        cnn_groups=[*cache.cnn_groups[:-1], cache.cnn_groups[-1][:-1]],
        attention=cache.attention,
    )

    with pytest.raises(ValueError, match="incompatible structure"):
        model.decode_step(torch.randint(67, (2, 1)), malformed)

    bad_attention = CNNInterleavedAttentionCache(
        token_ids=cache.token_ids,
        cnn_groups=cache.cnn_groups,
        attention=[
            (key[:, :, :-1], value[:, :, :-1])
            if index == 0
            else (key, value)
            for index, (key, value) in enumerate(cache.attention)
        ],
    )
    with pytest.raises(ValueError, match="attention cache"):
        model.decode_step(torch.randint(67, (2, 1)), bad_attention)


def test_checkpoint_reconstruction_on_available_devices(tmp_path: Path) -> None:
    config = small_config(context_length=8)
    checkpoint = save_checkpoint(
        tmp_path / "model-f.pt",
        model=build_model(config),
        step=1,
        model_config=config,
        data_fingerprint="f" * 64,
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
            data_fingerprint="f" * 64,
            device=device,
        )
        input_ids = torch.randint(config.vocab_size, (2, 8), device=device)
        loss = model(input_ids).float().mean()
        loss.backward()

        assert loaded_config == config
        assert torch.isfinite(loss)
