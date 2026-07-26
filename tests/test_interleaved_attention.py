"""Coverage for Model E's interleaved CNN-attention architecture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import CNNInterleavedAttentionConfig, ModelConfig
from kiwilm.generation import generate_tokens
from kiwilm.inference import load_trained_model
from kiwilm.models import CNNInterleavedAttentionLM, build_model


def small_config(**overrides: object) -> CNNInterleavedAttentionConfig:
    values: dict[str, object] = {
        "vocab_size": 67,
        "context_length": 12,
        "d_model": 16,
        "dropout": 0.0,
        "num_heads": 2,
        "feedforward_dim": 32,
    }
    values.update(overrides)
    return CNNInterleavedAttentionConfig(**values)


def test_default_shape_parameter_count_weight_tying_and_order() -> None:
    config = CNNInterleavedAttentionConfig()
    model = build_model(config)

    logits = model(torch.randint(config.vocab_size, (1, 8)))
    dilations = [
        block.conv.conv.dilation[0]
        for group in model.cnn_groups
        for block in group
    ]

    assert logits.shape == (1, 8, config.vocab_size)
    assert sum(parameter.numel() for parameter in model.parameters()) == 6_050_816
    assert isinstance(model, CNNInterleavedAttentionLM)
    assert model.lm_head.weight is model.token_embedding.weight
    assert [len(group) for group in model.cnn_groups] == [2, 2, 2]
    assert len(model.attention_blocks) == 2
    assert dilations == [1, 2, 4, 8, 16, 32]


def test_model_is_strictly_causal_and_supports_backward() -> None:
    torch.manual_seed(31)
    config = small_config()
    model = CNNInterleavedAttentionLM(config).eval()
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


def test_config_round_trip() -> None:
    config = small_config(
        kernel_size=5,
        dilations=(1, 3, 9, 27, 54, 108),
        tie_embeddings=False,
    )
    serialized = config.to_dict()

    assert json.loads(json.dumps(serialized)) == serialized
    assert CNNInterleavedAttentionConfig.from_dict(serialized) == config
    assert ModelConfig.from_dict(serialized) == config


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"architecture": "cnn_attention"}, "architecture"),
        ({"kernel_size": 0}, "kernel_size"),
        ({"dilations": (1, 2, 4)}, "exactly 6"),
        ({"dilations": (1, 2, 4, 8, 16, 0)}, "dilation"),
        ({"num_heads": 0}, "num_heads"),
        ({"d_model": 15, "num_heads": 2}, "divisible"),
        ({"d_model": 24, "num_heads": 8}, "even"),
        ({"feedforward_dim": 0}, "feedforward_dim"),
    ],
)
def test_config_validation(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        small_config(**overrides)


def test_incremental_cache_matches_forward_generation_and_rollover() -> None:
    torch.manual_seed(37)
    config = small_config(context_length=8)
    model = CNNInterleavedAttentionLM(config).eval()
    prompt = torch.randint(config.vocab_size, (2, 4))

    logits, cache = model.prefill(prompt)
    torch.testing.assert_close(logits, model(prompt))
    assert len(cache.cnn_groups) == 3
    assert [len(group) for group in cache.cnn_groups] == [2, 2, 2]
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
    assert torch.equal(cached, uncached)


def test_decode_step_validates_input_and_cache_batch() -> None:
    model = CNNInterleavedAttentionLM(small_config()).eval()
    _, cache = model.prefill(torch.randint(67, (2, 4)))

    with pytest.raises(ValueError, match="shape"):
        model.decode_step(torch.randint(67, (2, 2)), cache)
    with pytest.raises(ValueError, match="batch size"):
        model.decode_step(torch.randint(67, (1, 1)), cache)


def test_checkpoint_reconstruction_and_available_accelerators(
    tmp_path: Path,
) -> None:
    config = small_config(context_length=8)
    checkpoint = save_checkpoint(
        tmp_path / "model-e.pt",
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
