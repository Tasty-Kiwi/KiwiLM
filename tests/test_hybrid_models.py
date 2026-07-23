"""Coverage for Models C and D and the portable selective state space."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import (
    CNNAttentionMambaConfig,
    CNNDualAttentionConfig,
    ModelConfig,
)
from kiwilm.inference import load_trained_model
from kiwilm.models import (
    CNNAttentionMambaLM,
    CNNDualAttentionLM,
    SelectiveStateSpace,
    build_model,
)


@pytest.mark.parametrize(
    ("config", "model_type", "parameter_count"),
    [
        (CNNDualAttentionConfig(), CNNDualAttentionLM, 6_050_816),
        (CNNAttentionMambaConfig(), CNNAttentionMambaLM, 6_027_648),
    ],
)
def test_default_shape_parameter_count_and_weight_tying(
    config: CNNDualAttentionConfig | CNNAttentionMambaConfig,
    model_type: type[torch.nn.Module],
    parameter_count: int,
) -> None:
    model = build_model(config)
    logits = model(torch.randint(config.vocab_size, (1, 8)))

    assert logits.shape == (1, 8, config.vocab_size)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
    assert isinstance(model, model_type)
    assert model.lm_head.weight is model.token_embedding.weight


@pytest.mark.parametrize(
    "config",
    [
        CNNDualAttentionConfig(
            vocab_size=71,
            context_length=16,
            d_model=16,
            dropout=0.0,
            num_heads=2,
            feedforward_dim=32,
        ),
        CNNAttentionMambaConfig(
            vocab_size=71,
            context_length=16,
            d_model=16,
            dropout=0.0,
            num_heads=2,
            feedforward_dim=32,
            mamba_inner_dim=24,
            mamba_state_dim=4,
            mamba_conv_kernel=3,
            mamba_dt_rank=4,
        ),
    ],
)
def test_models_are_strictly_causal_and_support_backward(
    config: CNNDualAttentionConfig | CNNAttentionMambaConfig,
) -> None:
    torch.manual_seed(13)
    model = build_model(config).eval()
    original = torch.randint(config.vocab_size, (1, 16))
    changed = original.clone()
    changed[:, 9:] = torch.randint(config.vocab_size, (1, 7))

    original_logits = model(original)
    changed_logits = model(changed)
    torch.testing.assert_close(original_logits[:, :9], changed_logits[:, :9])

    original_logits.mean().backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_selective_state_space_full_scan_matches_step_updates() -> None:
    torch.manual_seed(17)
    layer = SelectiveStateSpace(6, state_dim=3, dt_rank=2)
    layer.reset_parameters()
    values = torch.randn(2, 11, 6)

    full_outputs, full_state = layer(values)
    state = layer.initial_state(2, device=values.device)
    step_outputs = []
    for token in values.unbind(dim=1):
        output, state = layer.step(token, state)
        step_outputs.append(output)

    torch.testing.assert_close(full_outputs, torch.stack(step_outputs, dim=1))
    torch.testing.assert_close(full_state, state)


def test_mamba_model_is_finite_at_full_context() -> None:
    config = CNNAttentionMambaConfig(
        vocab_size=64,
        context_length=256,
        d_model=16,
        dropout=0.0,
        num_heads=2,
        feedforward_dim=32,
        mamba_inner_dim=24,
        mamba_state_dim=4,
        mamba_conv_kernel=3,
        mamba_dt_rank=4,
    )
    model = build_model(config).eval()

    with torch.no_grad():
        logits = model(torch.randint(config.vocab_size, (1, 256)))

    assert bool(torch.isfinite(logits).all())


@pytest.mark.parametrize(
    "config",
    [
        CNNDualAttentionConfig(),
        CNNAttentionMambaConfig(),
    ],
)
def test_config_json_round_trip(
    config: CNNDualAttentionConfig | CNNAttentionMambaConfig,
) -> None:
    serialized = config.to_dict()
    assert json.loads(json.dumps(serialized)) == serialized
    assert ModelConfig.from_dict(serialized) == config


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mamba_inner_dim": 0}, "mamba_inner_dim"),
        ({"mamba_state_dim": 0}, "mamba_state_dim"),
        ({"mamba_conv_kernel": 0}, "mamba_conv_kernel"),
        ({"mamba_dt_rank": 0}, "mamba_dt_rank"),
        ({"architecture": "cnn_attention"}, "architecture"),
    ],
)
def test_mamba_config_validation(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CNNAttentionMambaConfig(**overrides)


@pytest.mark.parametrize(
    "config",
    [
        CNNDualAttentionConfig(
            vocab_size=64,
            context_length=8,
            d_model=16,
            dropout=0.0,
            num_heads=2,
            feedforward_dim=32,
        ),
        CNNAttentionMambaConfig(
            vocab_size=64,
            context_length=8,
            d_model=16,
            dropout=0.0,
            num_heads=2,
            feedforward_dim=32,
            mamba_inner_dim=24,
            mamba_state_dim=4,
            mamba_conv_kernel=3,
            mamba_dt_rank=4,
        ),
    ],
)
def test_checkpoint_reconstruction_and_available_accelerators(
    config: CNNDualAttentionConfig | CNNAttentionMambaConfig,
    tmp_path: Path,
) -> None:
    checkpoint = save_checkpoint(
        tmp_path / f"{config.architecture}.pt",
        model=build_model(config),
        step=1,
        model_config=config,
        data_fingerprint="c" * 64,
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
            data_fingerprint="c" * 64,
            device=device,
        )
        input_ids = torch.randint(64, (1, 8), device=device)
        loss = model(input_ids).float().mean()
        loss.backward()
        assert loaded_config == config
        assert torch.isfinite(loss)
