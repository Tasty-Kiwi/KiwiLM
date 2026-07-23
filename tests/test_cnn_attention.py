"""Tests for the CNN-attention-CNN language-model variant."""

from __future__ import annotations

import json

import pytest
import torch

from kiwilm.config import CNNAttentionConfig, ModelConfig
from kiwilm.models import CNNAttentionLM, build_model


def test_default_model_shape_parameter_count_and_weight_tying() -> None:
    config = CNNAttentionConfig()
    model = build_model(config)
    logits = model(torch.randint(config.vocab_size, (2, 31)))

    assert logits.shape == (2, 31, config.vocab_size)
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_261_056
    assert isinstance(model, CNNAttentionLM)
    assert model.lm_head.weight is model.token_embedding.weight


def test_model_is_strictly_causal_and_deterministic_in_eval() -> None:
    torch.manual_seed(7)
    config = CNNAttentionConfig(
        vocab_size=97,
        context_length=24,
        d_model=16,
        dropout=0.5,
        num_heads=2,
        feedforward_dim=32,
    )
    model = CNNAttentionLM(config).eval()
    original = torch.randint(config.vocab_size, (2, 24))
    changed = original.clone()
    changed[:, 13:] = torch.randint(config.vocab_size, (2, 11))

    with torch.no_grad():
        first = model(original)
        second = model(original)
        changed_logits = model(changed)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[:, :13], changed_logits[:, :13])


def test_attention_exposes_a_global_causal_information_path() -> None:
    torch.manual_seed(5)
    config = CNNAttentionConfig(
        vocab_size=41,
        context_length=160,
        d_model=16,
        dropout=0.0,
        num_heads=2,
        feedforward_dim=32,
    )
    model = CNNAttentionLM(config)
    captured: list[torch.Tensor] = []

    def retain_embedding_output(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        output.retain_grad()
        captured.append(output)

    hook = model.token_embedding.register_forward_hook(retain_embedding_output)
    try:
        logits = model(torch.randint(config.vocab_size, (1, 160)))
        logits[0, -1, 0].backward()
    finally:
        hook.remove()

    assert captured[0].grad is not None
    assert captured[0].grad[0, 0].abs().sum().item() > 0.0


def test_config_plain_dict_round_trip() -> None:
    config = CNNAttentionConfig(
        vocab_size=512,
        context_length=64,
        d_model=48,
        dropout=0.25,
        tie_embeddings=False,
        kernel_size=5,
        pre_attention_dilations=(1, 3, 9),
        post_attention_dilations=(27, 54, 108),
        num_heads=6,
        feedforward_dim=192,
    )
    serialized = config.to_dict()

    assert json.loads(json.dumps(serialized)) == serialized
    assert CNNAttentionConfig.from_dict(serialized) == config
    assert ModelConfig.from_dict(serialized) == config


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"architecture": "gated_cnn"}, "architecture"),
        ({"kernel_size": 0}, "kernel_size"),
        ({"pre_attention_dilations": (1, 2)}, "exactly 3"),
        ({"post_attention_dilations": (8, 16, 0)}, "dilation"),
        ({"num_heads": 0}, "num_heads"),
        ({"d_model": 15, "num_heads": 2}, "divisible"),
        ({"d_model": 24, "num_heads": 8}, "even"),
        ({"feedforward_dim": 0}, "feedforward_dim"),
    ],
)
def test_config_validation(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CNNAttentionConfig(**overrides)


def test_forward_backward_on_available_accelerators() -> None:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    config = CNNAttentionConfig(
        vocab_size=64,
        context_length=8,
        d_model=16,
        dropout=0.0,
        num_heads=2,
        feedforward_dim=32,
    )
    for device in devices:
        model = CNNAttentionLM(config).to(device)
        input_ids = torch.randint(config.vocab_size, (2, 8), device=device)
        loss = model(input_ids).float().mean()
        loss.backward()
        assert torch.isfinite(loss)
