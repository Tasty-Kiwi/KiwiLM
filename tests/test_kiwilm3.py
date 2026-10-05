"""M3 qualifies an untrained bidirectional encoder, not diffusion or training speed."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from kiwilm.config import ModelConfig
from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.encoder import BidirectionalEncoder
from kiwilm.models.kiwilm2 import SwiGLU
from kiwilm.models.kiwilm3 import FullBidirectionalAttention, GatedBidirectionalConv
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.profile import profile_encoder
from kiwilm.v3.validation import qualify_encoder
from kiwilm.v3.weights import load_encoder_weights, save_encoder_weights


def tiny(**overrides):
    values = dict(
        vocab_size=97,
        context_length=16,
        d_model=16,
        num_heads=2,
        swiglu_dim=48,
        noise_embedding_dim=16,
    )
    values.update(overrides)
    return KiwiLM3Config(**values)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("depth", [12, 16])
def test_baseline_and_json(depth):
    config = KiwiLM3Config(num_blocks=depth)
    assert config.d_model == config.context_length == 512
    assert config.vocab_size == 32000 and config.num_heads == 8 and config.swiglu_dim == 2048
    assert config.mixer_schedule == tuple(
        ("attention", "biconv", "biconv")[i % 3] for i in range(depth)
    )
    assert config.conv_kernel_sizes == (31, 63) * (depth // 3)
    assert KiwiLM3Config.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    # V2 dispatch and its causal interface must not silently reinterpret an encoder.
    with pytest.raises(ValueError):
        ModelConfig.from_dict(config.to_dict())


@pytest.mark.parametrize(
    "overrides",
    [
        {"architecture": "kiwilm2"},
        {"schema_version": 2},
        {"d_model": 15},
        {"num_heads": 3},
        {"num_blocks": 0},
        {"vocab_size": True},
        {"noise_embedding_dim": 3},
        {"pad_token_id": 97},
        {"pad_token_id": True},
        {"dropout": float("nan")},
        {"dropout": 1},
        {"dropout": True},
        {"rms_norm_eps": 0},
        {"rope_base": float("inf")},
        {"mixer_schedule": "attention"},
        {"mixer_schedule": ("gqa",) * 12},
        {"mixer_schedule": ("biconv",)},
        {"conv_kernel_sizes": (2,) * 8},
        {"conv_kernel_sizes": (31,)},
        {"conv_kernel_sizes": "31"},
    ],
)
def test_config_refuses_invalid_baselines(overrides):
    with pytest.raises(ValueError):
        tiny(**overrides)


@pytest.mark.parametrize("depth", [12, 16])
def test_exact_blocks_shapes_tied_head_and_backward(depth):
    torch.manual_seed(42)
    config = tiny(num_blocks=depth)
    model = build_encoder(config)
    assert isinstance(model, BidirectionalEncoder) and not isinstance(model, CausalLanguageModel)
    assert model.reconstruction_head.weight is model.token_embedding.weight
    assert not hasattr(model, "generate") and not hasattr(model, "ngrams")
    kernels = iter(config.conv_kernel_sizes)
    for kind, block in zip(config.mixer_schedule, model.blocks, strict=True):
        assert isinstance(block.mlp, SwiGLU)
        if kind == "attention":
            assert isinstance(block.mixer, FullBidirectionalAttention)
            assert block.mixer.query.out_features == block.mixer.key.out_features == 16
        else:
            assert isinstance(block.mixer, GatedBidirectionalConv)
            assert block.mixer.kernel_size == next(kernels)
            assert block.mixer.depthwise.padding == (block.mixer.kernel_size // 2,)
    ids = torch.tensor([[2, 5, 7, 3, 0], [2, 9, 8, 3, 0]])
    hidden = model.encode(ids, noise_level=torch.tensor([0.25, 0.75]))
    logits = model(ids, noise_level=torch.tensor([0.25, 0.75]))
    assert hidden.shape == (2, 5, 16) and logits.shape == (2, 5, 97)
    torch.testing.assert_close(logits, model.reconstruction_head(hidden))
    logits[:, :4].square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


@pytest.mark.parametrize("schedule", [("attention",), ("biconv",), None])
def test_right_context_influences_earlier_positions(schedule):
    torch.manual_seed(141)
    config = tiny() if schedule is None else tiny(num_blocks=1, mixer_schedule=schedule)
    model = build_encoder(config).eval()
    ids = torch.tensor([[2, 4, 5, 6, 3]])
    changed = ids.clone()
    changed[0, 3] = 7
    assert not torch.allclose(model.encode(ids)[:, 1], model.encode(changed)[:, 1], atol=1e-7)
    # A left-position prediction depends differentiably on a right-only token embedding.
    model(ids)[0, 1, 11].backward()
    assert model.token_embedding.weight.grad[6].norm() > 0


def test_biconv_is_symmetric_with_finite_local_radius():
    conv = GatedBidirectionalConv(1, 3, dropout=0).eval()
    with torch.no_grad():
        conv.input.weight.fill_(1)
        conv.depthwise.weight.fill_(1)
        conv.depthwise.bias.zero_()
        conv.output.weight.fill_(1)
    values = torch.ones(1, 7, 1)
    mask = torch.ones(1, 7, dtype=torch.bool)
    baseline = conv(values, mask)
    left, right, far = (values.clone() for _ in range(3))
    left[:, 2] = 2
    right[:, 4] = 2
    far[:, 5] = 2
    torch.testing.assert_close(conv(left, mask)[:, 3], conv(right, mask)[:, 3])
    assert (conv(right, mask)[:, 3] > baseline[:, 3]).all()
    torch.testing.assert_close(conv(far, mask)[:, 3], baseline[:, 3])


def test_padding_isolation_all_padded_rows_and_authoritative_mask():
    model = build_encoder(tiny()).eval()
    ids = torch.tensor([[2, 4, 5, 0, 0], [0, 0, 0, 0, 0]])
    mask = ids != 0
    changed = ids.clone()
    changed[~mask] = 11
    expected = model(ids)
    torch.testing.assert_close(expected, model(changed, attention_mask=mask))
    assert torch.isfinite(expected).all() and expected[~mask].count_nonzero() == 0
    assert model.encode(ids)[~mask].count_nonzero() == 0
    expected.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    all_valid = torch.ones_like(ids, dtype=torch.bool)
    assert model(ids, attention_mask=all_valid)[1].norm() > 0
    no_padding = build_encoder(tiny(pad_token_id=None))
    assert no_padding(ids)[1].norm() > 0
    # Right padding does not alter valid representations; centered conv sees zeros.
    torch.testing.assert_close(model.encode(ids[:1, :3]), model.encode(ids[:1])[:, :3])


def test_noise_conditioning_scalar_batch_and_gradients():
    model = build_encoder(tiny()).eval()
    ids = torch.tensor([[2, 4, 3], [2, 5, 3]])
    torch.testing.assert_close(model(ids), model(ids, noise_level=0.0))
    torch.testing.assert_close(
        model(ids, noise_level=0.25), model(ids, noise_level=torch.full((2,), 0.25))
    )
    assert not torch.allclose(model(ids, noise_level=0.25), model(ids, noise_level=0.75))
    level = torch.tensor([0.25, 0.75], requires_grad=True)
    model(ids, noise_level=level).square().mean().backward()
    assert torch.isfinite(level.grad).all() and level.grad.norm() > 0


@pytest.mark.parametrize(
    "noise",
    [
        True,
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        torch.tensor([0, 1]),
        torch.zeros(3),
        torch.zeros(2, 1),
    ],
)
def test_noise_validation(noise):
    with pytest.raises(ValueError, match="noise_level"):
        build_encoder(tiny())(torch.ones(2, 3, dtype=torch.long), noise_level=noise)


@pytest.mark.parametrize(
    "ids",
    [
        torch.zeros(0, 3, dtype=torch.long),
        torch.zeros(1, 0, dtype=torch.long),
        torch.zeros(1, 17, dtype=torch.long),
        torch.ones(1, 3),
        torch.ones(3, dtype=torch.long),
        torch.tensor([[-1]]),
        torch.tensor([[97]]),
    ],
)
def test_input_validation(ids):
    with pytest.raises(ValueError, match="input_ids"):
        build_encoder(tiny())(ids)


@pytest.mark.parametrize("mask", [torch.ones(1, 3), torch.ones(3, dtype=torch.bool)])
def test_padding_mask_validation(mask):
    with pytest.raises(ValueError, match="attention_mask"):
        build_encoder(tiny())(torch.ones(1, 3, dtype=torch.long), attention_mask=mask)


def test_residual_init_and_parameter_replacing_device_transfer():
    config = tiny(d_model=64, num_heads=8, swiglu_dim=192)
    model = build_encoder(config)
    expected = 0.02 / math.sqrt(2 * config.num_blocks)
    for block in model.blocks:
        assert block.mlp.down.weight.std().item() == pytest.approx(expected, rel=0.08)
        assert block.mixer.output.weight.std().item() == pytest.approx(expected, rel=0.08)
        assert torch.equal(block.mlp_norm.weight, torch.ones(64))
    flag = torch.__future__.get_overwrite_module_params_on_conversion()
    try:
        torch.__future__.set_overwrite_module_params_on_conversion(True)
        model.to(dtype=torch.float64)
        assert model.reconstruction_head.weight is model.token_embedding.weight
        assert model(torch.tensor([[2, 4, 3]])).dtype == torch.float64
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(flag)


@pytest.mark.parametrize("depth", [12, 16])
@pytest.mark.parametrize(
    "suffix,dtype", [(".pt", "fp32"), (".safetensors", "fp32"), (".safetensors", "bf16")]
)
def test_weights_roundtrip(tmp_path, depth, suffix, dtype):
    config = tiny(num_blocks=depth)
    model = build_encoder(config).eval()
    path = tmp_path / ("encoder" + suffix)
    tokenizer_sha = "a" * 64
    save_encoder_weights(model, path, dtype=dtype, tokenizer_sha256=tokenizer_sha)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    restored, restored_config = load_encoder_weights(
        path,
        expected_config=config,
        expected_tokenizer_sha256=tokenizer_sha,
        expected_sha256=digest,
    )
    assert restored_config == config and not restored.training
    assert restored.reconstruction_head.weight is restored.token_embedding.weight
    for name, value in model.state_dict().items():
        expected = value if dtype == "fp32" else value.to(torch.bfloat16).float()
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    reference = build_encoder(config).eval()
    reference.load_state_dict(restored.state_dict())
    ids = torch.tensor([[2, 4, 5, 3, 0]])
    torch.testing.assert_close(restored(ids, noise_level=0.5), reference(ids, noise_level=0.5))
    if dtype == "fp32":
        torch.testing.assert_close(restored(ids), model(ids), rtol=0, atol=0)
    if suffix == ".safetensors":
        assert "reconstruction_head.weight" not in load_file(path)
    with pytest.raises(FileExistsError):
        save_encoder_weights(model, path)
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_encoder_weights(path, expected_config=replace(config, dropout=0.1))
    with pytest.raises(ValueError, match="tokenizer checksum mismatch"):
        load_encoder_weights(path, expected_tokenizer_sha256="b" * 64)
    with pytest.raises(ValueError, match="weights checksum mismatch"):
        load_encoder_weights(path, expected_sha256="b" * 64)


def test_weights_storage_defaults_and_invalid_artifacts(tmp_path):
    model = build_encoder(tiny())
    path = tmp_path / "encoder.safetensors"
    save_encoder_weights(model, path)
    with safe_open(path, framework="pt") as stream:
        assert stream.metadata()["weights_dtype"] == "bf16"
    bad = tmp_path / "causal.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, bad)
    with pytest.raises(ValueError, match="different loaders"):
        load_encoder_weights(bad)
    torch.save([], bad)
    with pytest.raises(ValueError, match="encoder weight artifact"):
        load_encoder_weights(bad)
    with pytest.raises(ValueError, match="SHA256"):
        save_encoder_weights(model, tmp_path / "bad.pt", tokenizer_sha256="bad")
    with torch.no_grad():
        model.token_embedding.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        save_encoder_weights(model, tmp_path / "nan.pt")
    assert not (tmp_path / "nan.pt").exists()


def test_profile_accounting_and_full_attention_cost():
    config = tiny()
    profile = profile_encoder(config, sequence_length=8)
    assert profile["parameters"] == sum(p.numel() for p in build_encoder(config).parameters())
    assert profile["embedding_parameters"] == 97 * 16
    assert profile["attention_blocks"] == 4 and profile["biconv_blocks"] == 8
    attention = profile["blocks"][0]
    assert attention["forward_flops_per_token"] == 8 * 16**2 + 4 * 8 * 16 + 6 * 16 * 48
    assert profile["kv_cache"] is None and not profile["is_throughput_benchmark"]
    twelve, sixteen = (profile_encoder(KiwiLM3Config(num_blocks=n)) for n in (12, 16))
    assert twelve["parameters"] == 65_123_840
    assert sixteen["parameters"] == 81_430_016
    assert twelve["forward_flops_per_token"] < sixteen["forward_flops_per_token"]
    with pytest.raises(ValueError):
        profile_encoder(config, sequence_length=17)


@pytest.mark.parametrize("depth", [12, 16])
def test_local_qualification_does_not_train_or_change_rng(depth):
    before = torch.get_rng_state().clone()
    report = qualify_encoder(depth)
    assert report["passed"] and not report["trained"] and report["device"] == "cpu"
    assert report["qualification_config"]["num_blocks"] == depth
    assert report["full_width_static_profile"]["config"]["d_model"] == 512
    assert torch.equal(before, torch.get_rng_state())
