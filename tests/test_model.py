"""Tests for the gated-convolution language-model baseline."""

from __future__ import annotations

import json

import pytest
import torch

from kiwilm.config import GatedCNNConfig, ModelConfig
from kiwilm.models import GatedCNNLM, build_model


def test_default_model_shape_parameter_count_and_weight_tying() -> None:
    config = GatedCNNConfig()
    model = build_model(config)
    input_ids = torch.randint(config.vocab_size, (2, 31))

    logits = model(input_ids)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert logits.shape == (2, 31, config.vocab_size)
    assert parameter_count == 5_259_776
    assert isinstance(model, GatedCNNLM)
    assert model.lm_head.weight is model.token_embedding.weight


def test_default_initialization_has_well_scaled_logits_and_backward() -> None:
    torch.manual_seed(11)
    config = GatedCNNConfig()
    model = GatedCNNLM(config)
    input_ids = torch.randint(config.vocab_size, (1, 8))
    targets = torch.randint(config.vocab_size, (1, 8))

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )
    loss.backward()

    assert model.token_embedding.weight.std().item() == pytest.approx(0.02, abs=0.001)
    assert logits.std().item() < 1.0
    assert torch.isfinite(loss)
    assert loss.item() < 20.0
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_model_is_strictly_causal() -> None:
    torch.manual_seed(7)
    config = GatedCNNConfig(
        vocab_size=97,
        context_length=24,
        d_model=16,
        dropout=0.0,
        num_layers=3,
        dilations=(1, 2, 4),
    )
    model = GatedCNNLM(config).eval()
    original = torch.randint(config.vocab_size, (2, 24))
    changed = original.clone()
    changed[:, 13:] = torch.randint(config.vocab_size, (2, 11))

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed)

    torch.testing.assert_close(original_logits[:, :13], changed_logits[:, :13])


def test_config_plain_dict_round_trip() -> None:
    config = GatedCNNConfig(
        vocab_size=512,
        context_length=64,
        d_model=48,
        dropout=0.25,
        tie_embeddings=False,
        num_layers=4,
        kernel_size=5,
        dilations=(1, 3, 9, 27),
    )

    serialized = config.to_dict()

    assert json.loads(json.dumps(serialized)) == serialized
    assert serialized["dilations"] == [1, 3, 9, 27]
    assert GatedCNNConfig.from_dict(serialized) == config
    assert ModelConfig.from_dict(serialized) == config


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"vocab_size": 0}, "vocab_size"),
        ({"context_length": 0}, "context_length"),
        ({"d_model": -1}, "d_model"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
        ({"tie_embeddings": 1}, "tie_embeddings"),
        ({"num_layers": 0}, "num_layers"),
        ({"kernel_size": 0}, "kernel_size"),
        ({"dilations": (1, 2)}, "dilations"),
        ({"dilations": (1, 2, 4, 8, 16, 32, 64, 0)}, "dilation"),
        ({"architecture": "attention"}, "architecture"),
    ],
)
def test_config_validation(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GatedCNNConfig(**overrides)


def test_model_rejects_invalid_input_shape_and_context_overflow() -> None:
    model = GatedCNNLM(
        GatedCNNConfig(
            vocab_size=32,
            context_length=4,
            d_model=8,
            num_layers=1,
            dilations=(1,),
        )
    )

    with pytest.raises(ValueError, match="shape"):
        model(torch.ones(4, dtype=torch.long))
    with pytest.raises(ValueError, match="context length"):
        model(torch.ones((1, 5), dtype=torch.long))
